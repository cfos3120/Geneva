import torch 
import numpy as np
import einsum

'''NOTE:THis only applied for time concatenated with nodes (not native Geneva form)'''
class get_boundary_functions():
    def __init__(self, bc_type_dict, loss_function, dim=2):
        self.loss_function = loss_function
        self.boundary_function_u = getattr(self, f"{bc_type_dict['U']['type']}")
        self.boundary_function_p = getattr(self, f"{bc_type_dict['p']['type']}")
        #self.normals = bc_type_dict['Normals']
        self.bc_type_dict = bc_type_dict
        self.dim = dim

    def no_slip(self,out,channel,**kwargs):
        # normal and tangent components are zero
        
        return self.loss_function(out)

    def zero_gradient(self,out,channel,derivatives):
        # normal gradient are zero
        if channel=='U':
            du = torch.cat((derivatives['u_x'].unsqueeze(-1),derivatives['u_y'].unsqueeze(-1)), dim=-1)
            dv = torch.cat((derivatives['v_x'].unsqueeze(-1),derivatives['v_y'].unsqueeze(-1)), dim=-1)
            du_normal = torch.einsum('ijk,jk->ij', du, self.normals)
            dv_normal = torch.einsum('ijk,jk->ij', dv, self.normals)

            loss = self.loss_function(du_normal)
            loss += self.loss_function(dv_normal)
            return loss/2
        
        elif channel=='p':
            dp = torch.cat((derivatives['p_x'].unsqueeze(-1),derivatives['p_y'].unsqueeze(-1)), dim=-1)
            dp_normal = torch.einsum('ijk,jk->ij', dp, self.normals)
            loss = self.loss_function(dp_normal)
            return loss
        else:
            raise NotImplementedError('BC only works for U and p')

    def fixed_value(self,out,channel,**kwargs):
        
        b = out.shape[0]
        n = out.shape[-2]
        t = out.shape[1]

        fixed_value_array = self.bc_type_dict[channel]['value']
        if not np.any(fixed_value_array):
            fixed_value_array = None
        else:
            assert len(fixed_value_array) == out.shape[-1], f'{len(fixed_value_array)} does not match {out.shape[-1]}'
            if len(out.shape) == 3:
                fixed_value_array = torch.tensor(fixed_value_array, dtype=torch.float32).reshape(1,1,len(fixed_value_array))
                fixed_value_array = fixed_value_array.repeat(b,n,1)
            elif len(out.shape) == 4:
                fixed_value_array = torch.tensor(fixed_value_array, dtype=torch.float32).reshape(1,1,1,len(fixed_value_array))
                fixed_value_array = fixed_value_array.repeat(b,t,n,1)
            else:
                raise NotImplementedError('Output Tensor has too many dimensions')
        return self.loss_function(out,fixed_value_array)

    def __call__(self, out, derivatives):
        # Replace ... with boundary indices (assigned at init)
        u_loss = self.boundary_function_u(out[...,:-1],channel='U', derivatives=derivatives)
        p_loss = self.boundary_function_p(out[...,[-1]],channel='p', derivatives=derivatives)

        return u_loss, p_loss