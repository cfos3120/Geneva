import h5py
from glob import glob
import os
import torch
import numpy as np
import yaml
from einops import rearrange
from torch.utils.data import Dataset,DataLoader
from scipy.signal import find_peaks

def pad_with_interpolated_values(arr, num):
    # Calculate the number of new elements to be added
    new_length = num * (len(arr) - 1) + 1
    
    # Create a new array with the new length
    padded_array = np.zeros(new_length, dtype=int)

    # Fill the new array with origiexcenal and interpolated values
    padded_array[::num] = arr
    for i in range(len(arr) - 1):
        for j in range(num - 1):
            padded_array[i*(num)+j+1] = arr[i] + (j+1)*((arr[i+1]-arr[i]) / num)
    
    return padded_array

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
                 train = False,
                 bc=False
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
        self.periodicity = True
        self.methods = dataset_params['dataset_args']['methods']

        # Load data list (and sort range by Re)
        self.data_paths_all = sorted(glob(os.path.join(dataset_params['dir'], 'cylinder_2D*.h5py')))
        self.sorted_path = sorted(self.data_paths_all, key=lambda x: int(x.partition('RE_')[2].partition('.')[0]))  

        # Load Grid
        with h5py.File(f'{dataset_params["dir"]}/cell_centre_coordinates.h5py','r') as f:
            self.cell_centres = torch.from_numpy(f["cell_data"][()]).type(torch.float32)
        
        if bc:
            self.bc_dir = f'{dataset_params["dir"]}/mesh_data.h5py'
            

        self.N = self.cell_centres.shape[0]
        self.C = dataset_params['output_dim']

        self.filter_cases(Re_index=dataset_params['dataset_args']['Re_range'])
        self.dataset = self.train_split()
        self.non_dimensionalize()
        self.dataset_len = self.dataset[0].shape[0]
    
    def non_dimensionalize(self, U=1, L=2):

        # Space
        self.cell_centres = self.cell_centres/L

        # Time
        self.dataset[2] = self.dataset[2]/(L/U)
        self.dataset[3] = self.dataset[3]/(L/U)

        # Output 
        self.dataset[1][...,:2] = self.dataset[1][...,:2]/U
        self.dataset[1][...,-1] = self.dataset[1][...,-1]/(U**2)

        # Input
        self.dataset[0][...,:2] = self.dataset[0][...,:2]/U
        self.dataset[0][...,2] = self.dataset[0][...,2]/(U**2)
        
        print('Dataset Non-dimensionalized')

    def filter_cases(self, Re_index=None):
        if Re_index is None:
            return
        elif isinstance(Re_index,list):
            self.sorted_path  = [path for path in self.sorted_path 
                                    if int(path.partition('RE_')[2].partition('.')[0]) >= Re_index[0] 
                                    and int(path.partition('RE_')[2].partition('.')[0]) <= Re_index[1]]
        else:
            self.sorted_path  = [path for path in self.sorted_path 
                                    if int(path.partition('RE_')[2].partition('.')[0]) == Re_index]
    
    '''
    TODO: What do we want to do here:
    1. Get a mix of Reynolds numbers and sliding window batches.
    2. Sliding window batches are periods (uneven per case)
    3. We need to split into train and testing:
        a. only unseen reynolds numbers in testing
        b. only unseen windows in testing
        c. both (seperate datasets for validation)

    Idea:
    1. Iterate over cases, split into training and testing
    2. rearrange into minibatches
    3a. concatenate onto main batches (nodes, channels , time)in/out)
        should all still be the same, just more batches for higher frequency results
    3b. we know period decreases with increasing reynolds number, so cap
        number of batches at max batch size for lower end (ensures model doesn't
        skew towards the higher reynolds numbers)
    '''
    def print_train_split_info(self, dataset_set):
        print(f'\n {"Training" if self.train else "Testing"} Dataset Info:')
        print(f'    Case Split:             {self.data_split*100 if self.train else (1-self.data_split)*100 :.0f}%')             
        print(f'    Batches:                {dataset_set[0].shape[0]}')
        print(f'    Reynolds Numbers:       {", ".join([f"{i.item():.0f}" for i in torch.unique(dataset_set[4])])}')
        print(f'    Nodes (not sampled):    {dataset_set[0].shape[2]}')
        print(f'    Input Window:           {dataset_set[0].shape[1]}')
        print(f'    Output Window:          {dataset_set[1].shape[1]}')
        print(f'    Methods:                {", ".join([method for method in self.methods])}\n')

    def train_split(self):

        random_case_samples = np.random.default_rng(42).choice(len(self.sorted_path), size=int(self.data_split*len(self.sorted_path)), replace=False)
        
        # Loop Over Reynolds Number Cases
        for i, case_path in enumerate(self.sorted_path):
            case_batches_in, case_batches_out, time_range_in, time_range_out = self.sliding_window(case_path)
            
            # Cap Batches (ensure equal batches per Case)
            if i == 0:
                max_batches = case_batches_in.shape[0]

                # TODO: not implemented yet, but these indices will split out the windows for training and testing
                random_train_batch_samples = np.random.default_rng().choice(max_batches, size=int(self.data_split*max_batches), replace=False)
                random_test_batch_samples = np.delete(np.arange(max_batches),random_train_batch_samples) 

            case_batches_in = case_batches_in[:max_batches,...]
            case_batches_out = case_batches_out[:max_batches,...]
            time_range_in = time_range_in[:max_batches,...]
            time_range_out = time_range_out[:max_batches,...]
            
            Re = torch.tensor([float(case_path.partition('RE_')[2].partition('.')[0])]).reshape(1,1).repeat(max_batches,1)

            # Get training Reynolds Numbers
            if i in random_case_samples and self.train: # This train will be wrong if we do sliding window (to get test scope of training too)
                if 'case_batches_in_all_train' not in locals():
                    case_batches_in_all_train = case_batches_in
                    case_batches_out_all_train = case_batches_out
                    time_range_in_all_train = time_range_in
                    time_range_out_all_train = time_range_out
                    Re_list_train = Re
                else:
                    case_batches_in_all_train = torch.cat((case_batches_in_all_train, case_batches_in), dim=0)
                    case_batches_out_all_train = torch.cat((case_batches_out_all_train, case_batches_out), dim=0)
                    time_range_in_all_train = torch.cat((time_range_in_all_train, time_range_in), dim=0)
                    time_range_out_all_train = torch.cat((time_range_out_all_train, time_range_out), dim=0)
                    Re_list_train = torch.cat((Re_list_train, Re), dim=0)
                
                if max_batches > 1 and ('test all Re' in self.methods or 'peak2peak' in self.methods):
                    '''TODO: This may need to be put before the concatenation to ensure not the same random windows are taken for each case
                             The split dataset needs to be put into the _test variables
                    '''
                    raise NotImplementedError('Spliting a time window into train/test is not supported yet')

            # Get testing Reynolds Numbers 
            elif i not in random_case_samples and not self.train: 
                if 'case_batches_in_all_test' not in locals():
                    case_batches_in_all_test = case_batches_in
                    case_batches_out_all_test = case_batches_out
                    time_range_in_all_test = time_range_in
                    time_range_out_all_test = time_range_out
                    Re_list_test = Re
                else:
                    case_batches_in_all_test = torch.cat((case_batches_in_all_test, case_batches_in), dim=0)
                    case_batches_out_all_test = torch.cat((case_batches_out_all_test, case_batches_out), dim=0)
                    time_range_in_all_test = torch.cat((time_range_in_all_test, time_range_in), dim=0)
                    time_range_out_all_test = torch.cat((time_range_out_all_test, time_range_out), dim=0)
                    Re_list_test = torch.cat((Re_list_test, Re), dim=0)
                
        # if option 3a. see if "i" is training case and append into buckets
        # if option 3b. split into training and testing batches
        # if option 3c. do both (3b. then 3a.)

        if self.train:
            training_set = [case_batches_in_all_train, case_batches_out_all_train,
                            time_range_in_all_train, time_range_out_all_train,
                            Re_list_train]
            self.print_train_split_info(training_set)
            return training_set
        else:
            testing_set = [case_batches_in_all_test, case_batches_out_all_test,
                            time_range_in_all_test, time_range_out_all_test,
                            Re_list_test]
            self.print_train_split_info(testing_set)
            return testing_set
            

    def sliding_window(self,case_path, dt_size=None):

        # load in case
        with h5py.File(case_path,'r') as f:
            u = torch.from_numpy(f["cell_data"][()]).type(torch.float32)[self.seq_start:,...]
            grid_t = torch.from_numpy(f["time_data"][()]).type(torch.float32)[self.seq_start::self.reduced_resolution_t]
            coeff = f['force_coeffs']['Cl'][()]
            
        if self.periodicity:
            peaks, _ = find_peaks(coeff, height=0)
            troughs, _ = find_peaks(coeff*(-1), height=0)
    
        # Cap peaks here. for Reynolds number no. periods consistincy

        # now we have a set of peaks we need to get the indices bettween them.
        if 'peak2peak' in self.methods:
            # need to reduce interp number by 1 so you still get fixed input
            
            subset_i_in = pad_with_interpolated_values(peaks, num=self.in_dim-1)
            t_total_in = len(grid_t[subset_i_in])
            if self.in_dim != self.out_dim:
                subset_i_out = pad_with_interpolated_values(peaks, num=self.in_dim-1)
                #t_total_out = len(grid_t[subset_i_out])
            else:
                subset_i_out = subset_i_in
                #t_total_out = t_total_in

            number_batches = len(peaks)-2
            batched_input_tensor = torch.empty((number_batches, self.in_dim, self.N, self.C))
            batched_output_tensor = torch.empty((number_batches, self.out_dim, self.N, self.C))
            time_range_in = torch.empty((number_batches, self.in_dim))
            time_range_out = torch.empty((number_batches, self.out_dim))

            for i in range(number_batches):
                batched_input_tensor[i,...] = u[subset_i_in[(i*self.in_dim):(i+1)*self.in_dim],...]
                batched_output_tensor[i,...] = u[subset_i_out[(i+1)*self.out_dim-1:(i+2)*self.out_dim+1],...]
                time_range_in[i,...] = grid_t[subset_i_in[(i*self.in_dim):(i+1)*self.in_dim]]
                time_range_out[i,...] = grid_t[subset_i_out[(i+1)*self.out_dim-1:(i+2)*self.out_dim+1]]
                #print(subset_i_in[(i*self.in_dim):(i+1)*self.in_dim][-1], '=',subset_i_out[(i+1)*self.out_dim-1:(i+2)*self.out_dim+1][0])
                #print(subset_i_out[(i+1)*self.out_dim-1:(i+2)*self.out_dim+1][-1], '=',subset_i_out[-1])
                
                # centre time at zero (IC):
                time_range_in[i,...] = time_range_in[i,...] - time_range_in[i,-1]
                time_range_out[i,...] = time_range_out[i,...] - time_range_out[i,0]
                 
            '''NOTE: This by default predicts the initial condition'''

        elif 'phase2phase' in self.methods:
            phase_i_size = peaks[-1] - peaks[-2]
            n_time_steps = u.shape[0]
            all_indices = np.arange(n_time_steps)

            number_batches = int(np.floor((n_time_steps - 2*phase_i_size -1)/self.dt_size + 1))
            batched_input_tensor = torch.empty((number_batches, self.in_dim, self.N, self.C))
            batched_output_tensor = torch.empty((number_batches, self.out_dim, self.N, self.C))
            time_range_in = torch.empty((number_batches, self.in_dim))
            time_range_out = torch.empty((number_batches, self.out_dim))

            for i in range(number_batches):
                
                in_phase_indices = all_indices[[i*self.dt_size, i*self.dt_size + phase_i_size]]
                out_phase_indices = all_indices[[i*self.dt_size + phase_i_size , i*self.dt_size + 2*phase_i_size ]]

                subset_i_in = pad_with_interpolated_values(in_phase_indices, num=self.in_dim-1)
                subset_i_out = pad_with_interpolated_values(out_phase_indices, num=self.in_dim-1)

                # print('batch',i,': in',in_phase_indices, 'to', out_phase_indices,'|  intep: in',subset_i_in, 'to', subset_i_out,
                #       '|match in: ', subset_i_in[-1]==(subset_i_in[0] + phase_i_size), '|match out: ', subset_i_out[-1]==(subset_i_out[0] + phase_i_size))

                batched_input_tensor[i,...] = u[subset_i_in,...]
                batched_output_tensor[i,...] = u[subset_i_out,...]
                time_range_in[i,...] = grid_t[subset_i_in]
                time_range_out[i,...] = grid_t[subset_i_out]

                # centre time at zero (IC):
                time_range_in[i,...] = time_range_in[i,...] - time_range_in[i,-1]
                time_range_out[i,...] = time_range_out[i,...] - time_range_out[i,0]
        
        if 'generalize peak/trough' in self.methods:

            subset_i_in = pad_with_interpolated_values(troughs, num=self.in_dim-1)
            if self.in_dim != self.out_dim:
                subset_i_out = pad_with_interpolated_values(troughs, num=self.in_dim-1)
            else:
                subset_i_out = subset_i_in

            raise NotImplementedError('generalize peak/trough')
        
        return batched_input_tensor, batched_output_tensor, time_range_in, time_range_out

        # perfrom type of slide 
        # 4a. Equidistant dt index
        # 4b. Period
        # 4c. Both but for either multi-input or multi-output


    def _gen_subsample_indices(self, density):
        if density < 1.0:
            indices = np.random.default_rng().choice(self.cell_centres.shape[0], size=int(density*self.cell_centres.shape[0]), replace=False)
        else:   
            indices = np.arange(self.cell_centres.shape[0])
        return indices

    def add_bc_nodes(self):
        self.all_bc_points = np.array()
        previous_points = int(self.N*self.density) # start as internal points

        with h5py.File(self.bc_dir,'r') as f:
            self.bc_global_dict = dict.fromkeys(f.keys())

            for i,patch in enumerate(f.keys()):
                if i == 0:
                    patch_node_n = f[patch]["Vertices"][()].shape[0]
                    self.all_bc_points = f[patch]["Vertices"][()]
                    self.all_bc_normals = f[patch]["Normals"][()]
                    self.bc_global_dict[patch]
                else:
                    self.all_bc_points = np.concatenate((self.all_bc_points,f[patch]["Vertices"][()]),dim=0)
                    self.all_bc_normals = np.concatenate((self.all_bc_normals,f[patch]["Normals"][()]),dim=0)

                self.bc_global_dict[patch] = {"Vertices":f[patch]["Vertices"][()],
                                              "Normals":f[patch]["Normals"][()],
                                              "Indices":np.arange(patch_node_n) + previous_points}
                
                previous_points += patch_node_n
                


    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self,idx):
        input_seq   = self.dataset[0]
        output_seq  = self.dataset[1]
        input_t     = self.dataset[2]
        output_t    = self.dataset[3]
        Re          = self.dataset[4]

        indices = self._gen_subsample_indices(self.density)

        x = self.cell_centres[indices,:]
        y = output_seq[idx,:,indices,:]
        input_f1 = input_seq[idx,:,indices,:]
        input_f2 = Re[idx,...]
        input_t = input_t[idx,...]
        output_t = output_t[idx,...]
        
        # Concatenate time values:
        x = x.unsqueeze(0).repeat(output_t.shape[0],1,1)
        output_t = output_t.unsqueeze(-1).unsqueeze(-1).repeat(1,x.shape[1],1)
        x = torch.cat((x,output_t),dim=-1) # torch.Size([6, 5, 9200, 3])
        
        input_t = input_t.unsqueeze(-1).unsqueeze(-1).repeat(1,x.shape[1],1)
        input_f1 = torch.cat((input_f1,input_t),dim=-1)
        
        # Einops all tensors to be nodes only with time in nodes dim
        x = rearrange(x, 't n c -> (t n) c')
        y = rearrange(y, 't n c -> (t n) c')
        input_f1 = rearrange(input_f1, 't n c -> (t n) c')
        input_f2 = input_f2.reshape((1,1))

        return x, tuple([input_f1,input_f2]), y





    