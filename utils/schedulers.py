import torch.optim.lr_scheduler as schedulers

def get_schedulers(params, optimizer):

    scheduler_name = params['scheduler']

    if scheduler_name == 'LR':
        return schedulers.LambdaLR(optimizer, lr_lambda=lambda step: 0.85**step)
    elif scheduler_name == 'step':
        return schedulers.StepLR(optimizer, step_size=params['scheduler_step_size'], gamma=params['gamma'])
    elif scheduler_name == 'cycle':
        return schedulers.OneCycleLR(optimizer,
                                            max_lr=params['init_lr'],
                                            total_steps=params['epochs'],  # epoch --> step in the schedule
                                            **{k: params[k] for k in ['div_factor', 'pct_start', 'final_div_factor'] if k in params}
                                            )  
    elif scheduler_name == 'ReduceLROnPlateau':
        return schedulers.ReduceLROnPlateau(optimizer,
                                                mode='min',
                                                factor=params['lr_reduce_factor'],
                                                patience=params['lr_schedule_patience'],
                                                min_lr=params['min_lr'],
                                                verbose=True
                                                )
    else: raise NotImplementedError

