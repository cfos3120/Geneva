import math
import numpy as np
import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch.nn import functional as F
from torch.nn import GELU, ReLU, Tanh, Sigmoid
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

#from data_utils.utils import MultipleTensors
class MultipleTensors():
    def __init__(self, x):
        self.x = x

    def to(self, device):
        self.x = [x_.to(device) for x_ in self.x]
        return self

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item]
    
ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU()}

'''
    A simple MLP class, includes at least 2 layers and n hidden layers
'''
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', gating=False):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            self.act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.linear_pre = nn.Linear(n_input, n_hidden)
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_layers)])
        self.gating = gating

        # self.bns = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layers)])

    def forward(self, x):
        x = self.act(self.linear_pre(x))
        for i in range(self.n_layers):
            x = self.act(self.linears[i](x)) + x
            # x = self.act(self.bns[i](self.linears[i](x))) + x

        x = self.linear_post(x)
        return x
    
class rollout_RNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers=1, act='gelu', gating=False):
        super(rollout_RNN, self).__init__()

        if act in ACTIVATION.keys():
            self.act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.rnn = nn.RNN(n_input, n_hidden, num_layers=n_layers, nonlinearity='tanh', bias=True, batch_first=True)

        # self.bns = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layers)])

    def forward(self, x):
        #TODO: Extract time dimension from batch again.
        hn = None
        for i in range(x.shape[1]):
            output, hn = self.rnn(x[:,i,:,:], hn)
        x = output
        return x

class MoEGPTConfig():
    """ base GPT config, params common to all GPT versions """
    def __init__(self,attn_type='linear', embd_pdrop=0.0, resid_pdrop=0.0,attn_pdrop=0.0, n_embd=128, n_head=1, n_layer=3, block_size=128, n_inner=4,act='gelu',n_experts=2,space_dim=1,branch_sizes=None,n_inputs=1, rnn_input_i=None):
        self.attn_type = attn_type
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.n_embd = n_embd  # 64
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.n_inner = n_inner * self.n_embd
        self.act = act
        self.n_experts = n_experts
        self.space_dim = space_dim
        self.branch_sizes = branch_sizes
        self.n_inputs = n_inputs

        # For Flash Attention
        self.causal = False
        self.window_size = (-1, -1)

        # For RNN Rollout of Input
        self.rnn_input_i = rnn_input_i

class FlashAttention(nn.Module):
    """
    Using Flash Attention V2
    Features Cross and Self-Attention
    """

    def __init__(self, config, cross=False, rnn_input_i=None):
        super(FlashAttention, self).__init__()
        assert config.n_embd % config.n_head == 0

        self.cross = cross
        # key, query, value projections for all heads
        self.query = nn.Linear(config.n_embd, config.n_embd)
        if self.cross:
            self.keys = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_inputs)])
            self.values = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_inputs)])

            # override for index of input_f that is a timeseries
            if rnn_input_i is not None:
                self.keys[rnn_input_i] = rollout_RNN(config.n_embd, config.n_embd)
                self.values[rnn_input_i] = rollout_RNN(config.n_embd, config.n_embd)

        else:
            self.keys = nn.Linear(config.n_embd, config.n_embd)
            self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_inputs = config.n_inputs
        self.causal = config.causal
        self.window_size = config.window_size
        self.dropout_p = config.attn_pdrop

    '''
        Linear Attention and Linear Cross Attention
    '''
    def forward(self, x, y=None, layer_past=None):
        
        #y = x if y is None else y
        B, T1, C = x.size()
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T1, self.n_head, C // self.n_head)
        out = q

        if self.cross and y is not None:
            for i in range(self.n_inputs):
                T2 = y[i].shape[-2]
                k = self.keys[i](y[i]).view(B, T2, self.n_head, C // self.n_head).bfloat16()
                v = self.values[i](y[i]).view(B, T2, self.n_head, C // self.n_head).bfloat16()
                out = out + flash_attn_func(q.bfloat16(), k, v, self.dropout_p, causal=self.causal, window_size=self.window_size, deterministic=False).float()
        else:
            k = self.keys(x).view(B, T1, self.n_head, C // self.n_head).bfloat16()
            v = self.value(x).view(B, T1, self.n_head, C // self.n_head).bfloat16()
            out = flash_attn_func(q.bfloat16(), k, v, self.dropout_p, causal=self.causal, window_size=self.window_size, deterministic=False).float()
        
        # output projection
        out = rearrange(out, 'b n h d -> b n (h d)')
        out = self.proj(out)
        return out
    
'''
Self and Cross Attention block for CGPT, contains  a cross attention block and a self attention block
'''
class MIOECrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super(MIOECrossAttentionBlock, self).__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2_branch = nn.ModuleList([nn.LayerNorm(config.n_embd) for _ in range(config.n_inputs)])
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.ln4 = nn.LayerNorm(config.n_embd)
        self.ln5 = nn.LayerNorm(config.n_embd)
        if config.attn_type == 'flash':
            self.selfattn = FlashAttention(config)
            self.crossattn = FlashAttention(config, cross=True, rnn_input_i=config.rnn_input_i)
        else:
            raise NotImplementedError('Geneva currently only supports Flash Attention')

        if config.act == 'gelu':
            self.act = GELU
        elif config.act == "tanh":
            self.act = Tanh
        elif config.act == 'relu':
            self.act = ReLU
        elif config.act == 'sigmoid':
            self.act = Sigmoid

        self.resid_drop1 = nn.Dropout(config.resid_pdrop)
        self.resid_drop2 = nn.Dropout(config.resid_pdrop)

        self.n_experts = config.n_experts
        self.n_inputs = config.n_inputs

        self.moe_mlp1 = nn.ModuleList([nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, config.n_embd),
        ) for _ in range(self.n_experts)])

        self.moe_mlp2 = nn.ModuleList([nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, config.n_embd),
        ) for _ in range(self.n_experts)])

        self.gatenet = nn.Sequential(
            nn.Linear(config.space_dim, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, self.n_experts)
        )


    def ln_branchs(self, y):
        return MultipleTensors([self.ln2_branch[i](y[i]) for i in range(self.n_inputs)])

    '''
        x: [B, T1, C], y:[B, T2, C], pos:[B, T1, n]
    '''
    def forward(self, x, y, pos):
        gate_score = F.softmax(self.gatenet(pos),dim=-1).unsqueeze(2)    # B, T1, 1, m
        
        x = x + self.resid_drop1(self.crossattn(self.ln1(x), self.ln_branchs(y)))
        
        x_moe1 = torch.stack([self.moe_mlp1[i](x) for i in range(self.n_experts)],dim=-1) # B, T1, C, m
        x_moe1 = (gate_score*x_moe1).sum(dim=-1,keepdim=False)
       
        x = x + self.ln3(x_moe1)
        x = x + self.resid_drop2(self.selfattn(self.ln4(x)))
        
        x_moe2 = torch.stack([self.moe_mlp2[i](x) for i in range(self.n_experts)],dim=-1) # B, T1, C, m
        x_moe2 = (gate_score*x_moe2).sum(dim=-1,keepdim=False)
        
        x = x + self.ln5(x_moe2)
        return x
    
'''
Cross Attention GPT neural operator
Trunck Net: geom + RNN
'''

class GenevaNOT(nn.Module):
    def __init__(self,
                 trunk_size=2,
                 branch_sizes=[1],
                 space_dim=2,
                 time_dim=None,
                 output_size=3,
                 n_layers=3,
                 n_hidden=64,
                 n_head=1,
                 n_experts = 2,
                 n_inner = 4,
                 mlp_layers=2,
                 attn_type='linear',
                 act = 'gelu',
                 ffn_dropout=0.0,
                 attn_dropout=0.0,
                 horiz_fourier_dim = 0,
                 gating = True,
                 rnn_input_i=None,
                 rollout_steps=None,
                 in_timesteps=None,
                 **kwargs
                 ):
        super(GenevaNOT, self).__init__()

        # For RNN
        self.rnn_input_i = rnn_input_i
        self.in_timesteps = in_timesteps
        if rollout_steps is not None:
            trunk_size = trunk_size - 1
            print('WARNING: This GenevaNOT is hardcoded wrapped for GNOT input quieres')

        self.gating = gating
        self.horiz_fourier_dim = horiz_fourier_dim
        self.trunk_size = trunk_size * (4*horiz_fourier_dim + 3) if horiz_fourier_dim>0 else trunk_size
        self.branch_sizes = [bsize * (4*horiz_fourier_dim + 3) for bsize in branch_sizes] if horiz_fourier_dim > 0 else branch_sizes
        self.n_inputs = len(self.branch_sizes)
        self.output_size = output_size
        self.space_dim = space_dim

        # Get Layers
        self.gpt_config = MoEGPTConfig(attn_type=attn_type,embd_pdrop=ffn_dropout, resid_pdrop=ffn_dropout, attn_pdrop=attn_dropout,n_embd=n_hidden, n_head=n_head, n_layer=n_layers,
                                       block_size=128,act=act, n_experts=n_experts,space_dim=space_dim, branch_sizes=branch_sizes,n_inputs=len(branch_sizes),n_inner=n_inner,
                                       rnn_input_i=rnn_input_i)
  
        self.trunk_mlp = MLP(self.trunk_size, n_hidden, n_hidden, n_layers=mlp_layers,act=act)
        self.branch_mlps = nn.ModuleList([MLP(bsize, n_hidden, n_hidden, n_layers=mlp_layers,act=act) for bsize in self.branch_sizes])

        self.blocks = nn.Sequential(*[MIOECrossAttentionBlock(self.gpt_config) for _ in range(self.gpt_config.n_layer)])

        self.out_mlp = MLP(n_hidden, n_hidden, output_size, n_layers=mlp_layers)
        

        # For Decoder and Propagator
        self.rollout_steps = rollout_steps
        self.out_step = 1 #(only 1 timestep at a time for timebatched rollout)
        self.decoding_depth = 1
        self.rolling_checkpoint = False #trades memory for compute time (does not work with grad)
        n_hidden_prop = int(2*n_hidden) # expand features
        
        # Decoder Layers
        self.expand_feat = nn.Linear(n_hidden, n_hidden_prop)
        
        self.propagator = nn.ModuleList([
               nn.ModuleList([nn.LayerNorm(n_hidden_prop),
               nn.Sequential(
                    nn.Linear(n_hidden_prop + 1, n_hidden_prop, bias=False), # change the add 1 depending on what we are adding (e.g. space[2] or time[1] or both[3])
                    nn.GELU(),
                    nn.Linear(n_hidden_prop, n_hidden_prop, bias=False),
                    nn.GELU(),
                    nn.Linear(n_hidden_prop, n_hidden_prop, bias=False))])
            for _ in range(self.decoding_depth)])

        self.to_out = nn.Sequential(
            nn.LayerNorm(n_hidden_prop),
            nn.Linear(n_hidden_prop, n_hidden, bias=False),
            nn.GELU(),
            nn.Linear(n_hidden, n_hidden, bias=False),
            nn.GELU(),
            nn.Linear(n_hidden, self.output_size * self.out_step, bias=True))
        
        # self.apply(self._init_weights)

        self.__name__ = 'MIOEGPT_Geneva'


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def propagate(self, h, propagate_pos):
        for layer in self.propagator:
            norm_fn, ffn = layer
            h = h + ffn(torch.concat((norm_fn(h), propagate_pos), dim=-1))
        return h

    def decode(self, h):
        h = self.to_out(h)
        return h

    def rollout(self, h, propagate_pos, forward_steps):

        history = []
        
        h = self.expand_feat(h)

        assert self.out_step == 1, 'rollout() does not support batched out steps yet'
        for step in range(forward_steps//self.out_step):
            if self.rolling_checkpoint and self.training:
                h = checkpoint(self.propagate, h, propagate_pos[...,[step]])
                h_out = checkpoint(self.decode, h)
            else:
                h = self.propagate(h, propagate_pos[...,[step]])
                h_out = self.decode(h)
            h_out = rearrange(h_out, 'b n (t c) -> b (n t) c', c=self.output_size, t=self.out_step)
            history.append(h_out)

            x = torch.cat(history, axis=-2) # put time dim in nodes (aligns with our FLASH GNOT)
        return x

    def forward(self, x, inputs=None, in_timesteps=None, rollout_steps=None):
        
        if rollout_steps is not None and self.rollout_steps is not None:
            assert rollout_steps == self.rollout_steps, 'Currently GenevaNOT is wrapped for GNOT input data, rollout steps need to be the same as dataset'
            
        if rollout_steps is None and self.rollout_steps is not None:
            rollout_steps = self.rollout_steps
        
        if in_timesteps is None and self.in_timesteps is not None:
            in_timesteps = self.in_timesteps

        # cut down query field for x if rolling out instead
        if rollout_steps is not None:
            nodes_total = x.shape[1]
            
            # extract time_dim from query:
            x_t = torch.cat([x[:,i*(nodes_total//rollout_steps),-1].unsqueeze(-1) for i in range(rollout_steps)],dim=1)
            x_t = x_t.unsqueeze(1).repeat(1,(nodes_total//rollout_steps),1)
            x = x[:,:nodes_total//rollout_steps,:-1] 

        # unfold time dimension:
        if self.rnn_input_i is not None:
            assert in_timesteps is not None, 'Need to specify the in_timesteps when using the RNN encoder'
            b, n, c = inputs[self.rnn_input_i].shape
            inputs[self.rnn_input_i] = inputs[self.rnn_input_i].reshape(b,in_timesteps,int(n/in_timesteps),c)
        
        if self.gating:
            pos = x[:,:,0:self.space_dim]
        else:
            pos = None
        
        # if self.horiz_fourier_dim > 0:
        #     x = horizontal_fourier_embedding(x, self.horiz_fourier_dim)
        #     z = horizontal_fourier_embedding(z, self.horiz_fourier_dim)

        x = self.trunk_mlp(x)
        if self.n_inputs:
            z = MultipleTensors([self.branch_mlps[i](inputs[i]) for i in range(self.n_inputs)])
        else:
            z = MultipleTensors([x])

        for block in self.blocks:
            x = block(x, z, pos)
        
        if rollout_steps is not None:
            x = self.rollout(h=x, propagate_pos=x_t, forward_steps=rollout_steps)
        else:
            x = self.out_mlp(x)

        return x
    
