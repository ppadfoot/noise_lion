import math
import inspect
import functools
from typing import Any, Callable

import torch
from torch import Tensor
from typing import List, Optional, Tuple
from torch.nn.functional import cosine_similarity

import numpy as np


def rot_sampler(v, rot_angle):
    v = v/v.norm()
    gauss = torch.normal(torch.zeros(v.numel()), 1)
    sph_sample = (gauss / gauss.norm()).squeeze().to(v.device)
    sampled_angle = cosine_similarity(v, sph_sample, dim=0, eps=1e-20).acos()
    shift_rate = np.sin(rot_angle)/(np.sin(rot_angle)+(sampled_angle-rot_angle).sin())
    diff = sph_sample - v
    noised = v + diff * shift_rate
    noised = noised/noised.norm()
    return noised

class Noise_Lion(torch.optim.Optimizer):
  def __init__(self,
               params,
               lr=1e-4,
               betas=(0.9, 0.99),
               weight_decay=0.0,
               rot_angle=1.4,
              ):
    defaults = dict(
        lr = lr,
        betas = betas,
        weight_decay = weight_decay
    )
    super().__init__(params, defaults)
    self.lr = lr
    self.wd_coef = weight_decay
    self.betas = betas
    self.rot_angle = rot_angle
    self.momentum = []
    self.update = []
    self.presign = []

  @torch.no_grad()
  def update_fn(self, p, grad, exp_avg, update, lr, wd, beta2):
    p.mul_(1 - lr * wd)
    p.add_(update)
    exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)
      
  @torch.no_grad()
  def get_cat_update(self, cat_grad: torch.Tensor, cat_momemtum: torch.Tensor) -> Tuple[torch.Tensor]:
    cat_prenoise = cat_momemtum.mul_(self.betas[0]).add_(cat_grad, alpha=(1-self.betas[0]))
    if self.rot_angle == None:
        self.rot_angle = cosine_similarity(cat_grad, cat_grad.sign(), dim=0, eps=1e-20).acos().item()
    noised = rot_sampler(cat_prenoise, self.rot_angle)
    cat_update = noised * (-self.lr) * np.sqrt(noised.numel())
    return cat_update, cat_prenoise
    
  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()
    
    self.lr = self.param_groups[0]['lr']
    self.momentum, self.presign, self.update = [], [], []
    grad_list, shapes = [], []
    for group in self.param_groups:
      for i, p in enumerate([p for p in group['params'] if p.requires_grad]):
        grad, state = p.grad, self.state[p]
        if len(state) == 0:
          state['exp_avg'] = torch.zeros_like(p)
        grad_list += [grad]
        self.momentum += [state['exp_avg']]
        shapes += [p.data.shape]
    cat_grad = torch.cat([g.flatten() for g in grad_list])
    cat_momentum = torch.cat([m.flatten() for m in self.momentum])
    cat_update, cat_prenoise = self.get_cat_update(cat_grad, cat_momentum)
    self.update = [u.view(shapes[i]) for i, u in 
                   enumerate(torch.split(cat_update, [s.numel() for s in shapes]))]
    self.presign = [p.view(shapes[i]) for i, p in 
                    enumerate(torch.split(cat_prenoise, [s.numel() for s in shapes]))]
    self.momentum = []



    for j, group in enumerate(self.param_groups):
        if j > 0:
            break
        for i, p in enumerate([p for p in group['params'] if p.requires_grad]):
            grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]
            self.momentum += [state['exp_avg']]
            self.update_fn(p,
                            grad,
                            state['exp_avg'],
                            self.update[i],
                            lr,
                            wd,
                            beta2)
    
    return loss
