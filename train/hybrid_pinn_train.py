import torch
import numpy as np

from utils.loss_functions import get_loss_metrics

def train_epoch_hybrid(model, dataloader, loss_function, optimizer, loss_balancer, pde_controller, device='cpu'):

    cumu_loss = 0
    for x, input_f, y in dataloader:
        x, input_f, y = x.to(device), [f.to(device) for f in input_f], y.to(device)
        x.requires_grad = True

        # Infer Model
        optimizer.zero_grad()
        out = model(x=x, inputs=input_f)

        # Physics Informed Loss
        pde_loss_list = pde_controller(x=x, ic=input_f[0], Re=input_f[1], y=y, out=out)
        out, y = pde_controller.exclude_bc(out=out, y=y)

        # Calculate Pointwise Model Loss
        supervised_loss = loss_function(out,y)
        cumu_loss += supervised_loss.item()
        
        # Get total loss (through optional Balance Losses):
        all_losses_list = [supervised_loss] + pde_loss_list
        if loss_balancer is not None:
            loss = loss_balancer(loss_list=all_losses_list)
        else:
            loss = sum(all_losses_list)/len(all_losses_list)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000)
        optimizer.step()

    return cumu_loss / len(dataloader)



def val_epoch(model, dataloader, loss_function, metrics_list=None, device='cpu'):
    raise NotImplementedError('Currently this function is the same as basic train')
    if metrics_list is not None:
        epoch_metrics_dict = dict.fromkeys(metrics_list)
    else: 
        epoch_metrics_dict = None
    
    with torch.no_grad():
        cumu_loss = 0
        for x, input_f, y in dataloader:
            x, input_f, y = x.to(device), [f.to(device) for f in input_f], y.to(device)

            out = model(x=x, inputs=input_f)
            loss = loss_function(out,y)
            cumu_loss += loss.item()
            
            if metrics_list is not None:
                batch_metrics_dict = get_loss_metrics
                for metric_name in metrics_list:
                    epoch_metrics_dict[metric_name].append(batch_metrics_dict[metric_name])

        if metrics_list is not None:
            for key in epoch_metrics_dict:
                epoch_metrics_dict[key] = np.mean(epoch_metrics_dict[key])

    return cumu_loss / len(dataloader), epoch_metrics_dict

