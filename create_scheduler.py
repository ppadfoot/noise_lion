from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Заменяем функцию создания scheduler'а
def create_scheduler(args, optimizer):
    # Создаем warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=args.warmup_lr / args.lr,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
        verbose=False
    )
    
    # Создаем основной scheduler
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.min_lr,
        verbose=False
    )
    
    # Объединяем их в последовательный scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[args.warmup_epochs]
    )
    
    return scheduler, args.epochs
