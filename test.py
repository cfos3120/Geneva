import torch
import numpy as np
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
import torch.nn as nn
from einops import repeat, rearrange
import time

from nets.gnot_flash import GNOT, MoEGPTConfig, LinearAttention, FlashAttention, MultipleTensors


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#_______________________________________________________________________
def test_attention_flash(batch_size, 
                        seq_length, 
                        nheads, 
                        embedding_dim, 
                        require_grad, 
                        device,
                        causal=False,
                        local=False,
                        dtype=torch.float16):
    
    seqlen_q = seq_length
    seqlen_k = seq_length
    d = embedding_dim

    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=require_grad)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=require_grad)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=require_grad)

    mem_alloc_1 = torch.cuda.memory_allocated(device=device)
    out = flash_attn_func(q, k, v, 0.0, causal=causal, window_size=window_size, deterministic=False)
    mem_alloc_2 = torch.cuda.memory_allocated(device=device)
    print(f'Sequence {seqlen_q:.1E} Before {mem_alloc_1/1E+09:.4f} | After  {mem_alloc_2/1E+09:.4f} | Change {(mem_alloc_2-mem_alloc_1)/1E+09:.4f} | Shape {out.shape}')

def test_attention_base(batch_size, seq_length, nheads, embedding_dim, require_grad, device, dtype=torch.float16):
    seqlen_q = seq_length
    seqlen_k = seq_length
    d = embedding_dim

    # default torch attention layer
    multihead_attn = nn.MultiheadAttention(embed_dim=d, num_heads=nheads, dtype=dtype).to(device)
    q = torch.randn(batch_size, seqlen_q, d, device=device, dtype=dtype, requires_grad=require_grad).to(device)
    k = torch.randn(batch_size, seqlen_k, d, device=device, dtype=dtype, requires_grad=require_grad).to(device)
    v = torch.randn(batch_size, seqlen_k, d, device=device, dtype=dtype, requires_grad=require_grad).to(device)

    mem_alloc_1 = torch.cuda.memory_allocated(device=device)
    out,__ = multihead_attn(query=q, key=k, value=v)
    mem_alloc_2 = torch.cuda.memory_allocated(device=device)
    
    print(f'Sequence {seqlen_q:.1E} Before {mem_alloc_1/1E+09:.4f} | After  {mem_alloc_2/1E+09:.4f} | Change {(mem_alloc_2-mem_alloc_1)/1E+09:.4f} | Shape {out.shape}')

def test_attention_gnot(batch_size, seq_length, nheads, embedding_dim, require_grad, attn_type, device, dtype=torch.float16):
    seqlen_q = seq_length
    seqlen_k = seq_length
    d = embedding_dim

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=require_grad)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=require_grad)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=require_grad)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    mem_alloc_1 = torch.cuda.memory_allocated(device=device)
    if attn_type == 'l1':
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)   #
        k_cumsum = k.sum(dim=-2, keepdim=True)
        D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)       # normalized
    elif attn_type == "galerkin":
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)  #
        D_inv = 1. / seqlen_k                                 # galerkin
    elif attn_type == "l2":                                   # still use l1 normalization
        q = q / q.norm(dim=-1,keepdim=True, p=1)
        k = k / k.norm(dim=-1,keepdim=True, p=1)
        k_cumsum = k.sum(dim=-2, keepdim=True)
        D_inv = 1. / (q * k_cumsum).abs().sum(dim=-1, keepdim=True)  # normalized
    else:
        raise NotImplementedError

    context = k.transpose(-2, -1) @ v
    out = (q @ context) * D_inv + q
    out = rearrange(out, 'b h n d -> b n (h d)')
    mem_alloc_2 = torch.cuda.memory_allocated(device=device)

    print(f'Sequence {seqlen_q:.1E} Before {mem_alloc_1/1E+09:.4f} | After  {mem_alloc_2/1E+09:.4f} | Change {(mem_alloc_2-mem_alloc_1)/1E+09:.4f} | Shape {out.shape}')

def test_gnot(batch_size, n_points, embedding_dim, attn_type, dtype, require_grad=False):

    input_x = torch.randn(batch_size, 
                          n_points,  
                          2, # U,V,P
                          device=device, 
                          dtype=dtype, 
                          requires_grad=require_grad
                          ).to(device)
    
    input_f = [torch.randn(batch_size, 
                          1,  
                          1,
                          device=device, 
                          dtype=dtype, 
                          requires_grad=require_grad
                          ).view(batch_size,1,1).to(device)
                    for _ in range(2)]
    
    print(len(input_f))
    print(input_f[0].shape)
    
    model = GNOT(branch_sizes=[1],
                n_layers=1,
                n_hidden=embedding_dim,
                attn_type=attn_type).half().to(device)

    mem_alloc_1 = torch.cuda.memory_allocated(device=device)
    out = model(x=input_x,inputs=input_f)
    mem_alloc_2 = torch.cuda.memory_allocated(device=device)
    
    print(f'Sequence {n_points:.1E} Before {mem_alloc_1/1E+09:.4f} | After  {mem_alloc_2/1E+09:.4f} | Change {(mem_alloc_2-mem_alloc_1)/1E+09:.4f} | Shape {out.shape}')

    # Write config
    # Write input data
    # Forward Pass 
    return

def test_eigenvalues(attn_type, embedding_dim):

    config = MoEGPTConfig(attn_type=attn_type, 
                          embd_pdrop=0.0, 
                          resid_pdrop=0.0,
                          attn_pdrop=0.0, 
                          n_embd=embedding_dim, 
                          n_head=1, 
                          n_layer=3, 
                          block_size=embedding_dim, 
                          n_inner=4,
                          act='gelu',
                          n_experts=2,
                          space_dim=2,
                          branch_sizes=[1],
                          n_inputs=1
                          )
    # Check out yannicks video on transformers and eigenvalue
    # take only the self-attention modules (base and flash) from GNOT
    # Plot eigenvalue curves to demonstrate
    # looks like we will need a trained model to do this
    return


if __name__ == "__main__":

    seq_length = int(1e+03)
    require_grad = False
    nheads = 1
    batch_size = 1
    embedding_dim = 32
    attn_type_for_test_4 = 'flash' # 'linear' or 'flash
    
    
    test_nums = [4]

    if 1 in test_nums:
        test_attention_base(batch_size=batch_size,
                            seq_length=seq_length,
                            nheads=nheads,
                            embedding_dim=embedding_dim,
                            require_grad=require_grad,
                            device=device,
                            dtype=torch.float16)
    
    if 2 in test_nums:
        test_attention_flash(batch_size=batch_size,
                            seq_length=seq_length,
                            nheads=nheads,
                            embedding_dim=embedding_dim,
                            require_grad=require_grad,
                            device=device,
                            dtype=torch.float16)
        
    if 3 in test_nums:
        test_attention_gnot(batch_size=batch_size,
                            seq_length=seq_length,
                            nheads=nheads,
                            embedding_dim=embedding_dim,
                            require_grad=require_grad,
                            attn_type = 'l2',
                            device=device,
                            dtype=torch.float16)
        
    if 4 in test_nums:
        test_gnot(batch_size, 
                  n_points=seq_length, 
                  dtype=torch.float16,
                  embedding_dim=embedding_dim,
                  attn_type=attn_type_for_test_4, 
                  require_grad=require_grad)
        
    