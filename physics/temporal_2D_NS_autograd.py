import torch
import numpy as np

# Autograd Calculation of Gradients and Navier-Stokes Equations
def ns_pde_autograd(model_input_coords, model_out, Re, pressure=False):

    # Stack and Repeat Re for tensor multiplication
    Re = Re.squeeze(-1)
    #print('Re', Re.item(), "Max:", model_input_coords.max().item(), "Min:",model_input_coords.min().item())

    u = model_out[..., 0]
    v = model_out[..., 1]
    p = model_out[..., 2] 

    # First Derivatives
    u_out = torch.autograd.grad(u.sum(), model_input_coords, create_graph=True, retain_graph=True)[0]
    v_out = torch.autograd.grad(v.sum(), model_input_coords, create_graph=True, retain_graph=True)[0]
    p_out = torch.autograd.grad(p.sum(), model_input_coords, create_graph=True, retain_graph=True)[0]

    u_x = u_out[..., 0] #...might need to check the order of model_input_coords to ensure normal pressure boundary
    u_y = u_out[..., 1]
    u_t = u_out[..., 2]

    v_x = v_out[..., 0]
    v_y = v_out[..., 1]
    v_t = u_out[..., 2]

    p_x = p_out[..., 0]
    p_y = p_out[..., 1]
    
    # Second Derivatives
    u_xx = torch.autograd.grad(u_x.sum(), model_input_coords, create_graph=True, retain_graph=True)[0][..., 0]
    u_yy = torch.autograd.grad(u_y.sum(), model_input_coords, create_graph=True, retain_graph=True)[0][..., 1]
    v_xx = torch.autograd.grad(v_x.sum(), model_input_coords, create_graph=True, retain_graph=True)[0][..., 0]
    v_yy = torch.autograd.grad(v_y.sum(), model_input_coords, create_graph=True, retain_graph=True)[0][..., 1]

    # Continuity equation
    f0 = u_x + v_y

    # Navier-Stokes equation
    f1 = u_t + u*u_x + v*u_y - (1/Re) * (u_xx + u_yy) + p_x
    f2 = v_t + u*v_x + v*v_y - (1/Re) * (v_xx + v_yy) + p_y

    derivatives = {
                   'u_t': u_t,'u_x':u_x, 'u_y':u_y, 'v_t': v_t,'v_x':v_x, 'v_y':v_y, 'p_x':p_x, 'p_y':p_y,
                   'u_xx':u_xx, 'u_yy':u_yy, 'v_xx':v_xx, 'v_yy':v_yy
                   }
    
    # Pressure correction (incomressible condition)
    if pressure:
        f3 = (u**2 + v**2)*1/2 - p
        return [f0,f1,f2,f3], derivatives
    else:
        return [f0,f1,f2], derivatives

def input_output_temporal_lineage(model_out, input_f, derivatives, Re, input_samples=1, output_samples=1, order=2):
    
    B = input_f[0].shape[0]
    N_in = input_f[0].shape[1]
    C_in = input_f[0].shape[2]
    input_solution = input_f[0].reshape(B,input_samples,int(N_in/input_samples),C_in)

    N_out = model_out.shape[1]
    C_out = model_out.shape[2]
    output_solution = model_out.reshape(B,output_samples,int(N_out/output_samples),C_out)

    # Reshape derivatives
    derivatives = derivative_window_sample(derivatives, output_samples, N_out, B)

    # assuming input and output are equally spaced in non-dimensional time
    dt = np.abs(input_solution[0,0,0,-1].item() - input_solution[0,1,0,-1].item())

    # second order stencil in time
    if order == 2:
        u_t = (output_solution[:,0,:,0] - input_solution[:,-1,:,0]) / dt
        v_t = (output_solution[:,0,:,1] - input_solution[:,-1,:,1]) / dt

        # Navier-Stokes equation
        f1 = u_t + output_solution[:,0,:,0]*derivatives['u_x'] + output_solution[:,0,:,1]*derivatives['u_y'] - (1/Re) * (derivatives['u_xx'] + derivatives['u_yy']) + derivatives['p_x']
        f2 = v_t + output_solution[:,0,:,0]*derivatives['v_x'] + output_solution[:,0,:,1]*derivatives['v_y'] - (1/Re) * (derivatives['v_xx'] + derivatives['v_yy']) + derivatives['p_y']

    else:
        raise NotImplementedError('Only second order stencil is implemented')
    
    return [f1,f2]

def derivative_window_sample(derivatives, output_samples, N_out, B, window_index=0):

    for key in derivatives:
        derivatives[key] = derivatives[key].reshape(B,output_samples,int(N_out/output_samples))[:,window_index,:]

    return derivatives


'''
TODO: This function is a catch all one. This may be better of as a class, init with the dataset,
to store things like normalization, boundary geometric normals and nodes etc.
'''
def all_pde_losses(model_input_coords, model_out, Re,
                    input_f, input_samples, output_samples, order=2, pressure=False,
                    core_autograd=True, input_output_time_stencil=False, boundary_conditions=False):
    
    if input_output_time_stencil: assert core_autograd

    output_loss_list = list()
    output_loss_list_desc = list()

    if core_autograd:
        pde_losses, derivatives = ns_pde_autograd(model_input_coords, model_out, Re, pressure=False)
        output_loss_list_desc += ['Continuity', 'X-Momentum', 'Y-Momentum']
        output_loss_list += pde_losses

    if input_output_time_stencil:
        lineage_losses = input_output_temporal_lineage(model_out, input_f, derivatives, Re, input_samples, output_samples, order)
        output_loss_list_desc += ['X-Momentum (in/out)', 'Y-Momentum (in/out)']
        output_loss_list += lineage_losses

    if boundary_conditions:
        raise NotImplementedError('Boundary condition losses not supported yet')
    
    return output_loss_list, output_loss_list_desc