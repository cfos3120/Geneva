import argparse
import json
import os 
import wandb
import shutil
import yaml
import torch
from torch.utils.data import DataLoader
import pprint

from utils.gpu_utils import peripheral_setup
from utils.globalise import sweep_agent_wrapper, setup_output_dir
from utils.loss_functions import get_loss_function
from utils.schedulers import get_schedulers
from nets.gnot_flash import GNOT
from nets.geneva import GenevaNOT
from train.basic_train import train_epoch, val_epoch, plot_example_val
from train.hybrid_pinn_train import train_epoch_hybrid
from data_utils.load_case import load_dataset_case
from utils.dynamic_loss_balancing import RELOBRALO, fixed_weight_balancer
from physics.pde_class import pde_calculator

# load configuration from json
parser = argparse.ArgumentParser("TRAIN THE FLASH GNOT TRANSFORMER")
parser.add_argument('--config', type=str, help="json configuration file")
parser.add_argument('--wandb', type=str, help="wandb mode", choices=['online','offline','disabled'], default='offline')
parser.add_argument('--sweep', type=str, help="sweep json file")
args = parser.parse_args()

def pipeline(config):
    params = config['parameters']
    dataset_params = config['dataset_params']

    # setup Model
    if config['parameters']['model_name'] == 'Geneva':
        model_class = GenevaNOT
    else:  
        model_class = GNOT
    model = GNOT(trunk_size=dataset_params['input_dim'],
                    branch_sizes=dataset_params['branch_sizes'],
                    space_dim=dataset_params['space_dims'],
                    output_size=dataset_params['output_dim'],
                    n_head=params['n_head'],
                    n_layers=params['n_layers'],
                    n_experts=params['n_experts'],
                    n_hidden=params['n_hidden'],
                    attn_dropout=params['attn_dropout'],
                    attn_type=params['attn_type'],
                    **{k: params[k] for k in ['rnn_input_i', 'rollout_steps', 'in_timesteps'] if k in params}
                    )
    
    if len(config['peripheral']['device']) > 1:
       print(f'Model being trained in data parallel')
       model = torch.nn.DataParallel(model)

    model = model.to(device)
    wandb.watch(model)
    
    # Set up Learning
    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), params['learning_rate'])
    else: raise NotImplementedError

    scheduler = get_schedulers(params, optimizer)
    
    # Dataset
    train_dataset, val_dataset = load_dataset_case(dataset_params)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=4,
                            shuffle=True,
                            generator=torch.Generator().manual_seed(config['peripheral']['seed']))
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=4,
                            shuffle=True,
                            generator=torch.Generator().manual_seed(config['peripheral']['seed']))
    
    # Loss function for training
    loss_function = get_loss_function(params['loss_function'])

    # Init Loss Balancing Class
    if params['dyn_loss_balancing']:
        loss_balancer = RELOBRALO(device=device)
    elif params['loss_weights']:
        loss_balancer = fixed_weight_balancer(device=device, loss_list=[params[k] for k in params['loss_list'] if k in params])

    if params['PINN']:
        pde_controller = pde_calculator(config,train_dataset,loss_function)

    # Epoch Training loop
    for epoch in range(params['epochs']):

        #Train Model
        if params['PINN']:
            loss = train_epoch_hybrid(model=model,
                                        dataloader=train_loader,
                                        loss_function=loss_function,
                                        optimizer=optimizer,
                                        pde_controller=pde_controller,
                                        loss_balancer=loss_balancer,
                                        device=device)                         
        else:
            loss = train_epoch(model=model,
                                dataloader=train_loader,
                                loss_function=loss_function,
                                optimizer=optimizer,
                                device=device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Validate Model
        val_loss, metrics = val_epoch(model=model,
                            dataloader=val_loader,
                            loss_function=loss_function,
                            device=device)
        
        # Record loss and accuracy
        log_wandb = {'Epoch': epoch}
        log_wandb['Learning Rate'] = current_lr
        log_wandb[f'Training Loss'] = loss
        log_wandb[f'Validation Loss'] = val_loss
        if params['PINN']:
            pde_controller.wandb_record(log_wandb)
        if params['dyn_loss_balancing']: 
            loss_balancer.wandb_record(log_wandb)
    
        print(f"Epoch: {epoch+1:5n}/{params['epochs']:5n} | Train: {loss:.5f}| Val: {val_loss:.5f}")
        if metrics is not None:
            for metric in metrics:
                log_wandb[f'Validation ({metric})'] = metrics[metric]
        
        wandb.log(log_wandb)

    
    setup_output_dir(config)

    # Plot figure
    plot_example_val(model=model,
                    dataset=val_dataset,
                    case_n=0,
                    save_path = config['peripheral']['out_dir'],
                    device=device)
    
    # save model checkpoint
    if config['peripheral']['save_model']:
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint, f"{config['peripheral']['out_dir']}/checkpoint_epoch_{epoch+1}.pth")

    
if __name__ == "__main__":
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    device = peripheral_setup(gpu_list=config['peripheral']['device'], seed=config['peripheral']['seed'])
    #device = torch.torch.device('cuda')
    
    if args.sweep is not None:
        with open(args.sweep, 'r') as file:
            sweep_config = yaml.safe_load(file)
            sweep_id = wandb.sweep(sweep_config, project=config['wandb']['project'], entity=config['wandb']['entity'])
            config['peripheral']['out_dir'] = sweep_id
            wandb_sweep_controller = sweep_agent_wrapper(config)
            wandb_sweep_controller.assign_function(pipeline)
            wandb.agent(sweep_id, wandb_sweep_controller.run, count=config['wandb']['sweep_n'])
    else: 
        wandb.init(config=config,project=config['wandb']['project'], entity=config['wandb']['entity'], mode=args.wandb)
        pipeline(config)

    
    '''TODO:
    1. Autograd                                 - [x]
    2. Weightings                               - [x]
    3. PDEs                                     - [x]
    4. Parallel Training                        - [x]
    5. Emedding                                 - [ ]
    6. Code up decoder and propagator           - [-]
    7. projecting high dimensions onto 2d       - [ ]
    '''
