from typing import Tuple, Callable
import torch
from torch.optim.optimizer import Optimizer
import numpy as np
from torch.nn.functional import cosine_similarity

# functions
def exists(val):
    return val is not None

# update functions with angle calculation
def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay
    p.data.mul_(1. - lr * wd)
    
    # Calculate raw update
    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1. - beta1)
    
    # Calculate sign update
    update_sign = update.sign()
    
    # Calculate angle between raw update and sign update
    angle = cosine_similarity(update.flatten(), update_sign.flatten(), dim=0, eps=1e-20).acos().item()
    
    # weight update using sign
    p.add_(update_sign, alpha=-lr)
    
    # decay the momentum running average coefficient
    exp_avg.mul_(beta2).add_(grad, alpha=1. - beta2)
    
    return angle

def rot_sampler(v, rot_angle):
    v = v/v.norm()
    gauss = torch.normal(torch.zeros(v.numel()), 0.1)
    sph_sample = (gauss / gauss.norm()).squeeze().to(v.device)
    sampled_angle = cosine_similarity(v, sph_sample, dim=0, eps=1e-20).acos()
    shift_rate = np.sin(rot_angle)/(np.sin(rot_angle)+(sampled_angle-rot_angle).sin())
    diff = sph_sample - v
    noised = v + diff * shift_rate
    noised = noised/noised.norm()
    return noised

# class
class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = False,
        decoupled_weight_decay: bool = False,
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        self._init_lr = lr
        self.decoupled_wd = decoupled_weight_decay
        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay
        )
        super().__init__(params, defaults)
        self.update_fn = update_fn
        self.update_angles = []  # To store angles
        self.mean_angle = 0
        if use_triton:
            from lion_pytorch.triton import update_fn as triton_update_fn
            self.update_fn = triton_update_fn

    @torch.no_grad()
    def step(
        self,
        closure: Callable = None
    ):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        self.update_angles = []  # Reset angles list
                
        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):
                grad, lr, wd, beta1, beta2, state, decoupled_wd, init_lr = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p], self.decoupled_wd, self._init_lr
                
                # maybe decoupled weight decay
                if decoupled_wd:
                    wd /= init_lr
                
                # init state - exponential moving average of gradient values
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                angle = self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    wd,
                    beta1,
                    beta2
                )
                self.update_angles.append(angle)
        
        # Calculate average angle if needed
        self.mean_angle = np.mean(self.update_angles) if self.update_angles else 0.0
                
        return loss
