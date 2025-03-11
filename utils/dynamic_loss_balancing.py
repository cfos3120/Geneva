import torch

# Sourced from here: https://github.com/Khadrawi/ReLoBRaLo_PyTorch.git 
# Adapated from here:  https://github.com/rbischof/relative_balancing.git
# Adjustments made by Noah

class RELOBRALO():
    def __init__(self, T: float = 0.1, alpha: float = 0.999,
                 rho: float = 0.999, device: str = 'cuda') -> None:
        '''
        Initialization of ReloBRaLo, a loss balancing scheme used to improve the convergence
        of Physics-Informed Neural Networks (PINNS) proposed by Rafael Bischof
        Input:
            T: temperature parameter
            rho: expected value of the bernoulli random value 'rho' used for exponential decay
            alpha: exponential decay parameter for the weights of the losses
        Output:
            None
        '''
        
        self.T = torch.tensor(T, dtype=torch.float32, device=device)
        self.rho = torch.tensor(rho, device=device)
        self.alpha = alpha
        self.first_epoch_f = True
        self.device = device

    def set_l0(self, loss_list: list) -> None:
        '''This function is used to the save the values of the losses at the first epoch of
        training So it should be called as follows:
        '''
        assert all(isinstance(x, torch.Tensor) for x in loss_list), 'Each loss in loss_list'\
             ' must be a salar tensor'
        
        loss_list = loss_list.copy()
        
        # Assign each variable (in a list) per loss
        self.num_losses = len(loss_list)
        self.l0 ={"l0_"+str(i): torch.tensor([1.], device=self.device) for i in range(self.num_losses)}
        self.lam ={"lam_"+str(i): torch.tensor([1.], device=self.device) for i in range(self.num_losses)}
        self.l ={"l_"+str(i): torch.tensor([1.], device=self.device) for i in range(self.num_losses)}

        for i in range(self.num_losses):
            self.l0['l0_'+str(i)] = loss_list[i].reshape(1)

    def __call__(self, loss_list: list) -> torch.Tensor:
        '''This function returns the balanced loss of the muliobjective problem, where each loss
        in loss_list is multiplied by an adaptive weight.
        Call an instanciated object of this class after computing all the losses and pass a list of these losses
        '''

        loss_list = loss_list.copy()
        
        # initialize loss balancing
        if self.first_epoch_f:
            self.set_l0(loss_list=loss_list)
            self.first_epoch_f = False

        assert len(loss_list) == self.num_losses, 'Length of losses in the input list should'\
             ' be equal to self.num_losses'

        rho = torch.bernoulli(self.rho)
        lambs_hat = (torch.softmax(torch.cat([loss_list[i]/(self.l['l_'+str(i)]*self.T+1e-12) for i in range(self.num_losses)]),dim=0)*self.num_losses).detach()
        lambs0_hat = (torch.softmax(torch.cat([loss_list[i]/(self.l0['l0_'+str(i)]*self.T+1e-12) for i in range(self.num_losses)]),dim=0)*self.num_losses).detach()
        lambs = [rho*self.alpha*self.lam['lam_'+str(i)] + (1-rho)*self.alpha*lambs0_hat[i] + (1-self.alpha)*lambs_hat[i] for i in range(self.num_losses)]
        loss = torch.sum(torch.cat([lambs[i]*loss_list[i] for i in range(self.num_losses)]))
        for i in range(self.num_losses):
            self.lam['lam_'+str(i)] = lambs[i]
            self.l['l_'+str(i)] = loss_list[i].reshape(1)
        return loss
    
    def wandb_record(self, log_wandb):
        for i in range(self.num_losses):
            log_wandb[f'lam_{i}'] = self.lam['lam_'+str(i)].item()
            log_wandb[f'scaled_l_{i}'] = self.lam['lam_'+str(i)].item()*self.l['l_'+str(i)].item()

class fixed_weight_balancer():
    def __init__(self, loss_list: list, device: str = 'cuda') -> None:
        self.num_losses = len(loss_list)
        self.device = device
        self.lam ={"lam_"+str(i): torch.tensor([float(w)], device=self.device) for i,w in enumerate(loss_list)}
        self.l ={"l_"+str(i): torch.tensor([1.], device=self.device) for i in range(self.num_losses)}

    def set_l0(self, loss_list: list) -> None:
        pass

    def __call__(self, loss_list: list) -> torch.Tensor:
        assert len(loss_list) == self.num_losses, 'Length of losses in the input list should'\
             ' be equal to self.num_losses'

        loss = torch.sum(torch.cat([self.lam['lam_'+str(i)]*loss_list[i] for i in range(self.num_losses)]))
        for i in range(self.num_losses):
            self.l['l_'+str(i)] = loss_list[i].reshape(1)
        return loss
    
    def wandb_record(self, log_wandb):
        for i in range(self.num_losses):
            log_wandb[f'lam_{i}'] = self.lam['lam_'+str(i)].item()
            log_wandb[f'scaled_l_{i}'] = self.lam['lam_'+str(i)].item()*self.l['l_'+str(i)].item()
