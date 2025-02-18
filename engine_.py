import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def get_weight_norm(model: torch.nn.Module) -> float:
    """Calculate total weight norm for the model."""
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def get_grad_norm(model: torch.nn.Module) -> float:
    """Calculate average gradient norm for the model."""
    total_norm = 0.0
    count = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            count += 1
    if count == 0:
        return 0.0
    return (total_norm ** 0.5) / count


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # Add meters for weight and gradient norms
    metric_logger.add_meter('weight_norm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        # Calculate and log norms after optimizer step
        weight_norm = get_weight_norm(model)
        grad_norm = get_grad_norm(model)
        
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(weight_norm=weight_norm)
        metric_logger.update(grad_norm=grad_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
