import torch
import numpy as np

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y=None):
        # Custom adjustment here, if compared to zero we do absolute L2
        if y is None:
            return self.abs(x, torch.zeros_like(x))
        elif torch.all(y == 0):
            return self.abs(x, y)
        else:
            return self.rel(x, y)

def get_loss_function(name):
    if name == 'LPloss':
        return LpLoss()

def get_loss_metrics(pred, target, metrics_list):
    
    metric_dict = {}
    for metric_name in metrics_list:
        if metric_name in ['L2']:
            dict[metric_name] = torch.nn.MSELoss(reduction='mean')(pred, target).item()
        elif metric_name in ['L1']:
            dict[metric_name] = torch.nn.L1Loss(reduction='mean')(pred, target).item()
        elif metric_name in ['Rel_L2_Norm']:
            dict[metric_name] = LpLoss()(pred, target).item()
        else:
            raise NotImplementedError

    return metric_dict