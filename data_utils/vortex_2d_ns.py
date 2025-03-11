import h5py
from glob import glob
import os
import torch
import numpy as np
import yaml
from einops import rearrange
from torch.utils.data import Dataset,DataLoader

class Cylinder2DDataset(Dataset):
    """
    Dataset: Cylinder 2D Dataset
    Source: Custom
    This is a folder dataset (preprocessed).
    Folder:
    - Files (*.h5py):
        - Keys:
            - 'cell_data' Array (251, 36800, 3) (time-steps, cell centres, channels)
            - 'time_data' List (251,)
            - 'nu' value
            - 'force_coeffs' NA
    """
    def __init__(self, 
                 dataset_params, 
                 Re_index = None,
                 train = False
                 ):

        # basic parameters:
        self.seq_start = dataset_params['dataset_args']['seq_start']
        self.seq_end = dataset_params['dataset_args']['seq_end']
        self.reduced_resolution_t = dataset_params['dataset_args']['reduced_resolution_t']
        self.density = dataset_params['dataset_args']['cell_sample_density']
        self.data_split = dataset_params['dataset_args']['data_split']
        self.train = train
        self.how_to_concat_t = 'nodes' # same as HAMLET paper
        self.in_dim = dataset_params['dataset_args']['in_dim']
        self.out_dim = dataset_params['dataset_args']['out_dim']
        self.dt_size = dataset_params['dataset_args']['dt_size']

        # Load data list (and sort range by Re)
        self.data_paths_all = sorted(glob(os.path.join(dataset_params['dir'], 'cylinder_2D*.h5py')))
        self.sorted_path = sorted(self.data_paths_all, key=lambda x: int(x.partition('RE_')[2].partition('.')[0]))  

        # Load Grid
        with h5py.File(f'{dataset_params["dir"]}/cell_centre_coordinates.h5py','r') as f:
            self.cell_centres = torch.from_numpy(f["cell_data"][()]).type(torch.float32)
        
        if Re_index is not None:
            self.select_one_sample_only(Re_index)
            self.non_dimensionalize()

        self.x_node_n = int(self.cell_centres.shape[0]*self.density)
        self.input_f_node_n = int(self.input_f.shape[1]*self.density)

    def select_one_sample_only(self,Re_index):
        self.sorted_path = self.sorted_path[Re_index]
        print(f'Sampling only Dataset {os.path.realpath(self.sorted_path).split("/")[-1]} to be split up...')

        self._load_sample(self.sorted_path)
        self.input_f, self.output_y, self.time_step_mapping = self.sliding_temporal_window(in_dim=self.in_dim, out_dim=self.out_dim, dt_size=self.dt_size)
        self.dataset_len = self.output_y.shape[0]

        split_indices = self.train_split_indices(sliding_window=True)
        self.dataset_len = len(split_indices)
        
        self.output_y = self.output_y[split_indices,...]
        self.input_f = self.input_f[split_indices,...]

    def train_split_indices(self, sliding_window = True):
        random_samples = np.random.default_rng(42).choice(self.dataset_len, size=int(self.data_split*self.dataset_len), replace=False)
        
        if self.train:            
            return random_samples
        else:
            return np.delete(np.arange(self.dataset_len),random_samples)

    def non_dimensionalize(self, U=1, L=2):

        # Space and Time
        self.cell_centres[...,0] = self.cell_centres[...,0]/L # x
        self.cell_centres[...,1] = self.cell_centres[...,1]/L # y
        self.cell_centres[...,2] = self.cell_centres[...,2]/(L/U) 

        self.output_y[...,:2] = self.output_y[...,:2]/U
        self.output_y[...,-1] = self.output_y[...,-1]/(U**2)

        self.input_f[...,:2] = self.input_f[...,:2]/U
        self.input_f[...,2] = self.input_f[...,2]/(U**2) # check this, might have 4 channels with one being time
        self.input_f[...,3] = self.input_f[...,3]/(L/U) # t (based on concat) might throw error if called before

        print('Dataset Non-dimensionalized')

    def cap_reynolds_range(self):
        raise NotImplementedError
        self.sorted_path = self.sorted_path[dataset_params['dataset_range'][0]:dataset_params['dataset_range'][1]]
        print(f"Loading Re Range: {(self.sorted_path[0].partition('RE_')[2].partition('.')[0])}-{(self.sorted_path[-1].partition('RE_')[2].partition('.')[0])}")

    def _gen_subsample_indices(self, density=1.0):
        assert density <= 1.0

        if density < 1.0:
            self.indices = np.random.default_rng(self.seed).choice(self.cell_centres.shape[0], size=int(density*self.cell_centres.shape[0]), replace=False)
            #print(f'Mesh Randomly Sampled from {self.cell_centres.shape[0]:n} Cells to {len(self.indices):n} Cells ({density*100:.1f}%)')
        else:
            self.indices = np.arange(self.cell_centres.shape[0])

    def _load_sample(self,data_path):
        with h5py.File(data_path,'r') as f:
            self.u = torch.from_numpy(f["cell_data"][()]).type(torch.float32)[self.seq_start:,...]
            self.grid_t = torch.from_numpy(f["time_data"][()]).type(torch.float32)[self.seq_start::self.reduced_resolution_t]

    def sliding_temporal_window(self, in_dim=5, out_dim=5, dt_size=1, predict_ic=False):
        #if out_dim !=1: raise NotImplementedError('Sliding window shape not supported')
        # NOTE: dt_size here is index spacing not dimensional time

        '''
        TODO: We need to create a similar function but for period not fixed dt spacing.
        '''
        
        n_nodes = self.u.shape[1]
        out_nodes = self.cell_centres.shape[0]
        channels = self.u.shape[2]
        t_total = self.u.shape[0]
        
        # create empty batched tensor to fill
        batched_input_tensor = torch.empty((t_total - dt_size*(in_dim+out_dim-1), in_dim, n_nodes, channels))
        batched_output_tensor = torch.empty((t_total - dt_size*(in_dim+out_dim-1), out_dim, n_nodes, channels))
        time_step_mapping = torch.empty((t_total - in_dim, in_dim))
        dt = (self.grid_t[1]-self.grid_t[0])*dt_size

        print('Sliding window in time constructed for batches with:')
        print(f' - Time index gap:                  {dt_size}')
        print(f' - Input Shape:                     {batched_input_tensor.shape}')
        print(f' - Output Shape:                    {batched_input_tensor.shape}')
        print(f' - Temporal dt:                     {dt.item():.5f}')
        print(f' - First Output is Last Input:      {predict_ic}')
        
        for i in range(batched_input_tensor.shape[0]):

            batched_input_tensor[i,...] = self.u[i:i+in_dim*dt_size:dt_size,...].reshape(in_dim,n_nodes,channels)
            #batched_input_tensor[i,...] = self.u[i:i+in_dim, ...].reshape(in_dim,n_nodes,channels)
            #batched_output_tensor[i,...] = self.u[i+in_dim:i+in_dim+out_dim, ...].reshape(out_dim,n_nodes,channels)
            batched_output_tensor[i,...] = self.u[i+in_dim*dt_size:i+in_dim*dt_size+out_dim*dt_size:dt_size, ...].reshape(out_dim,n_nodes,channels)
            time_step_mapping[i,...] = self.grid_t[i:i+in_dim*dt_size:dt_size] # we can make this dt here
        
        if in_dim > 1:
            batched_input_tensor = torch.concat([batched_input_tensor,torch.zeros([batched_input_tensor.shape[0],in_dim,out_nodes,1])],dim=-1)
            for t in range(in_dim):
                batched_input_tensor[:,t,...,-1] = (in_dim-t-1)*(-dt)
            
        if out_dim > 1:
            # add time dimension
            self.cell_centres = self.cell_centres.unsqueeze(0).repeat(out_dim,1,1)
            # add time coordinate channel
            self.cell_centres = torch.concat([self.cell_centres,torch.zeros([out_dim,out_nodes,1])],dim=-1)
            #batched_output_tensor = torch.concat([batched_output_tensor,torch.zeros([batched_input_tensor.shape[0],out_dim,out_nodes,1])],dim=-1)

            # set time values as dts
            for t in range(out_dim):
                self.cell_centres[t,...,-1] = t*dt
                #batched_output_tensor[:,t,...,-1] = t*dt
            
            # rearrange
            self.cell_centres = self.cell_centres.permute(1,0,2)

        print(' input shape: ',batched_input_tensor.shape , 'output shape:',batched_output_tensor.shape )
        return batched_input_tensor, batched_output_tensor, time_step_mapping

    def _gen_subsample_indices(self, density):
        if density < 1.0:
            indices = np.random.default_rng(42).choice(self.cell_centres.shape[0], size=int(density*self.cell_centres.shape[0]), replace=False)
            #print(f'Mesh Randomly Sampled from {self.cell_centres.shape[0]:n} Cells to {len(indices):n} Cells ({density*100:.1f}%)')
        else:
            
            indices = np.arange(self.cell_centres.shape[0])
        return indices

    def one_sample_sliding_window_batch(self,idx):
        indices = self._gen_subsample_indices(self.density)

        y = self.output_y[idx,:,indices,:]
        input_f = self.input_f[idx,:,indices,:]
        x = self.cell_centres[indices,...]

        # Concatenate only if output time dim
        if len(self.cell_centres.shape) == 3:

            if self.how_to_concat_t == 'nodes':
                x = rearrange(x, 'n t c -> (t n) c')
                y = rearrange(y, 't n c -> (t n) c')
            elif self.how_to_concat_t == 'channels':
                x = rearrange(x, 'n t c -> n (t c)')
                y = rearrange(y, 't n c -> n (t c)')
        else:
            y = y.squeeze(0)

        # NOTE Up to here

        # Concatenate only if input time dim > 1
        if self.input_f.shape[1] > 1:

            # Old:
            # t_in_dim = self.time_step_mapping.shape[1]
            # t_in = self.time_step_mapping[idx,...].reshape(1,t_in_dim,1,1).repeat(1,1,len(indices),1) # change this to dt
            # input_f = torch.concat([input_f.unsqueeze(0),t_in],dim=-1)
            
            if self.how_to_concat_t == 'nodes':
                input_f = rearrange(input_f.unsqueeze(0), 'b t n c -> b (t n) c')
            elif self.how_to_concat_t == 'channels':
                input_f = rearrange(input_f.unsqueeze(0), 'b t n c -> b n (c t)')
            else:
                raise KeyError(f'{self.how_to_concat_t} is not valid, must be "nodes" or "channels"')
        
        return x, input_f, y

    def _prepare(self):
        raise NotImplementedError

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):

        if isinstance(self.sorted_path, str):
            x, input_f, y = self.one_sample_sliding_window_batch(idx)
            input_f = tuple(input_f)
        else:
            raise NotImplementedError
            data_path = self.data_paths[idx]
            self._load_sample(data_path)

        return x, input_f, y
    
    def _get_test(self):
        print(self.u.shape)

if __name__ == '__main__':
    config = '/home/n.foster@acfr.usyd.edu.au/Geneva/configs/multi_t_out.yaml'
    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    train_dataset = Cylinder2DDataset(config['dataset_params'], Re_index=0, train=True)
    test_dataset = Cylinder2DDataset(config['dataset_params'], Re_index=0, train=False)
    
    train_loader = DataLoader(
                            dataset=train_dataset,
                            batch_size=4,
                            shuffle=True,
                            generator=torch.Generator().manual_seed(42)
                        )
    
    for x, input_f, y in train_loader:
        break
    #x, input_f, y = dataset[0]
    print('x        ', x.shape)
    print('input_f  ', input_f[0].shape)
    print('y        ', y.shape,'\n')

    nodes = x.shape[1]
    print('concat test (x-coord)', x[0,0,0], x[0,9200,0])
    print('concat test (y-coord)', x[0,0,1], x[0,9200,1])
    print('concat test (t-coord)', x[0,0,2], x[0,9200,2])
    # input = torch.rand((4,2,100,4))#input_f[0]
    # module = torch.nn.Linear(4,32)
    # print(input.shape)
    # out = module(input)
    # print(out.shape)


    input_window_size = 5
    n_nodes = input_f[0].shape[1]
    n_nodes_per_t = int(n_nodes/input_window_size)
    last_window = input_f[0][:,4*n_nodes_per_t:5*n_nodes_per_t,:]
    print(last_window.shape)
    print(torch.min(last_window[...,-1]),torch.max(last_window[...,-1]))

    B = input_f[0].shape[0]
    N = input_f[0].shape[1]
    C = input_f[0].shape[2]
    input_f_reshape = input_f[0].reshape(B,input_window_size,int(N/input_window_size),C)
    print(torch.min(input_f_reshape[:,0,:,-1]),torch.max(input_f_reshape[:,0,:,-1]))
    print(torch.min(input_f_reshape[:,1,:,-1]),torch.max(input_f_reshape[:,1,:,-1]))
    print(torch.min(input_f_reshape[:,2,:,-1]),torch.max(input_f_reshape[:,2,:,-1]))
    print(torch.min(input_f_reshape[:,3,:,-1]),torch.max(input_f_reshape[:,3,:,-1]))
    print(torch.min(input_f_reshape[:,4,:,-1]),torch.max(input_f_reshape[:,4,:,-1]))

    # wierd outputs:
    #torch.Size([4, 18400, 3])
    # torch.Size([4, 18400, 4])
    # torch.Size([4, 2, 9200, 3])
    # torch.Size([4, 2, 100, 4])
    # torch.Size([4, 2, 100, 32])

    # should we concat time into dim, does it make a difference in attention
    