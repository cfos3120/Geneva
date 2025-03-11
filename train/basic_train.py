import torch
import numpy as np
import wandb

from utils.loss_functions import get_loss_metrics
from utils.visualize import figure_maker

def train_epoch(model, dataloader, loss_function, optimizer, device='cpu'):

    cumu_loss = 0
    for x, input_f, y in dataloader:
        x, input_f, y = x.to(device), [f.to(device) for f in input_f], y.to(device)
        
        optimizer.zero_grad()
        out = model(x=x, inputs=input_f)
        
        loss = loss_function(out,y)
        cumu_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000)
        optimizer.step()

    return cumu_loss / len(dataloader)

def val_epoch(model, dataloader, loss_function, metrics_list=None, device='cpu'):
    
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

def plot_example_val(model, dataset, case_n, save_path, device='cpu'):
    '''
    TODO: Need to Re as input.'''
    
    with torch.no_grad():

        for _ in range(case_n+1):
            x, input_f, y = next(iter(dataset))

        x, input_f, y = x.unsqueeze(0).to(device), [f.unsqueeze(0).to(device) for f in input_f], y.unsqueeze(0).to(device)
        out = model(x=x, inputs=input_f)

        # if 2D single time-step out
        if x.shape[-1] == 2:
            fig = figure_maker(Re=150, coord=x.cpu(), pred=out.cpu(), sol=y.cpu())
            frame = fig.get_frame(idx=0)
            frame.canvas.draw()
            data = np.frombuffer(frame.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(frame.canvas.get_width_height()[::-1] + (3,))
            # save frame to wandb
            image = wandb.Image(data, caption=f"Example result")
            wandb.log({"example": image})
        
        # if 2D multi time-step out
        else :
            Re=150
            save_file = f'{save_path}/example_re{Re:n}.gif'
            fig = figure_maker(Re=Re, coord=x.cpu(), pred=out.cpu(), sol=y.cpu())
            gif = fig.get_anim()
            gif.save(save_file, fps=2)
            print(f'Animation saved in: {save_file}')
            
            # save frame to wandb
            video = wandb.Video(save_file, caption=f"Example result")
            wandb.log({"example": video})

