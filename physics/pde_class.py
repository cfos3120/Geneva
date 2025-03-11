import torch
import numpy as np
from physics.temporal_2D_NS_autograd import ns_pde_autograd, input_output_temporal_lineage, derivative_window_sample
from physics.bc_functions import get_boundary_functions

class pde_calculator():

    def __init__(self, config, dataset, loss_function):
        self.config = config

        self.input_n = config['dataset_params']['dataset_args']['in_dim']
        self.output_n = config['dataset_params']['dataset_args']['out_dim']
        self.pressure = config['parameters']['pressure_correction']
        self.ic_lineage = config['parameters']['ic_lineage']
        self.boundary_conditions = config['parameters']['bc_loss']
        self.monitor_ns_only = config['parameters']['monitor_ns_only']
        self.x_node_n = int(dataset.N*dataset.density)
        self.ic_node_n = self.x_node_n # can adapt this later for multi size
        self.loss_function = loss_function

        self.loss_desc_list = ['Continuity', 'X-Momentum', 'Y-Momentum']
        self.cumu_pde_loss = None
        self.batch_iter_n = 0
                               
        if self.pressure:
            self.loss_desc_list.append('Pressure')

        if any(item in self.config['dataset_params']['dataset_args']['methods'] for item in ['peak2peak','phase2phase']) and self.ic_lineage:
            assert self.x_node_n == self.ic_node_n
            self.loss_desc_list += ['IC']\
            
        elif self.ic_lineage:
            assert self.x_node_n == self.ic_node_n, "output cell numbers need to match initial condition cell numbers"
            self.loss_desc_list += ['X-Momentum (in/out)', 'Y-Momentum (in/out)']

        if self.boundary_conditions:
            self.assign_boundaries(self, dataset.boundary_dict)
            raise NotImplementedError('Boundary condition losses not supported yet')


    def __call__(self,x, ic, Re, y, out):
        '''
        x   = input coordinates                 channels=(x,y,t)
        ic  = initial conditions (in input_f)   channels=(u,v,p,t)
        Re  = Reynolds number (in input_f)
        y   = ground_truth  -> aligns with x    channels=(u,v,p) 
        out = prediction    -> aligns with x    channels=(u,v,p) 
        '''

        output_loss_list = list()
        self.batch_iter_n += 1
        pde_losses = self.autograd_pde(x, out, Re)
        output_loss_list += pde_losses
        
        # Extract Time dim for stencils and lineage
        out = self.extract_time_dim(out)
        ic = self.extract_time_dim(ic)
        
        if any(item in self.config['dataset_params']['dataset_args']['methods'] for item in ['peak2peak','phase2phase']) and self.ic_lineage:
            output_loss_list += self.ic_first_last_match(out,ic)

        elif self.ic_lineage:
            for key in self.derivatives:
                self.derivatives[key] = self.extract_time_dim(self.derivatives[key].unsqueeze(-1))
        
            lineage_losses = input_output_temporal_lineage(out, ic, self.derivatives, Re, self.input_n, self.output_n)
            output_loss_list += lineage_losses

        if self.boundary_conditions:
            raise NotImplementedError('Boundary condition losses not supported yet')

        # Store for recording to wandb later
        if self.cumu_pde_loss is None:
            self.cumu_pde_loss = np.zeros(len(output_loss_list))
        self.cumu_pde_loss += np.array([pde_loss_n.item() for pde_loss_n in output_loss_list])

        if self.monitor_ns_only and len(output_loss_list)>3:
            output_loss_list = output_loss_list[3:]
        elif self.monitor_ns_only:
            output_loss_list = list()

        return output_loss_list
    
    def extract_time_dim(self, output):
        total_node_n = output.shape[1]
        channels = output.shape[-1]
        B = output.shape[0]
        return output.reshape(B,self.output_n,int(total_node_n/self.output_n),channels)
    
    def autograd_pde(self, x, out, Re):
        pde_losses, self.derivatives = ns_pde_autograd(x, out, Re, pressure=self.pressure)
        loss_list = list()
        for eqn in pde_losses:
            eqn_loss = self.loss_function(eqn, torch.zeros_like(eqn))
            loss_list += [eqn_loss]
        return loss_list

    def ic_first_last_match(self, output, input):
        loss = self.loss_function(output[:,0,...], input[:,-1,...,:-1])
        loss_list = list([loss])
        return loss_list
    
    def assign_boundaries(self, boundary_dict):
        self.bc_f_module_dict = dict.fromkeys(boundary_dict)
        for key in boundary_dict:
            self.bc_f_module_dict['key'] = get_boundary_functions(boundary_dict[key]['bc_types'])
        pass
        
    def exclude_bc(self,out, y):
        if self.boundary_conditions:
            '''TODO: AIM (all points in one inference)
                1. Get Indices per patch from dataset
                2. Parse boundary type into mesh_data.h5py (then parse further below)
                3. Create Boundary function types (i.e. fixed, zero_gradient, no_slip)
                4. Calculate loss on each boundary
                5. Optional list on yaml for which boundaries to enforce
                6. Ensure that
                    a. PDE satisfaction occurs at boundaries
                    b. Exclude boundaries from input/output comparison
            '''
            raise NotImplementedError('under construction')
        else:
            return out, y

    def boundary_conditions(self):
        '''
        TODO: This is a bit harder. We need to tag (through index) which nodes are boundary nodes.
        This is given we have stored the data as node aggregates and not cell centres. If we have cell centres.
        We need to create a boundary node dataset to pair with the cell centre data. Then we have a few options:
        1. Do a second pass through only with boundary nodes and pin these either:
            a. to the output data through strickly supervised learning on ground truth data
            b. to a combination of ground truth data for transient boundary values (pressure) and
                compare to input function values for fixed quantities (i.e. wall velocity)
        2. Combine the cell centre and vertex dataset, splitting out the comparison points based on index (will need
            to ensure the random cell sampling does not interfere)

        Question for Autograd: If we do two inferences and then combine the output datasets and coordinates (retaining the
        1:1 indexation), will we get an autograd result that encompasses the boundaries too?
        '''
        raise NotImplementedError
        self.boundary_normals 
        self.boundary_indicesS
    
    def wandb_record(self, log_wandb:dict):
        self.cumu_pde_loss = self.cumu_pde_loss / self.batch_iter_n
        
        # record to wandb
        for loss_name, loss_value in zip(self.loss_desc_list, self.cumu_pde_loss):
            log_wandb[f'{loss_name} Loss'] = loss_value

        # reset values for next epoch
        self.batch_iter_n = 0
        self.cumu_pde_loss = None


    

    
    





