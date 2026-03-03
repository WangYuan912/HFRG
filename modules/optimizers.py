import torch
import logging


def build_optimizer(args, model):
    """构建优化器 - 支持分层学习率"""
    # 分离视觉编码器和其他参数
    ve_params = list(map(id, model.visual_extractor.parameters()))
    ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())

    # 构建参数组
    param_groups = [
        {'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
        {'params': ed_params, 'lr': args.lr_ed}
    ]

    # 选择优化器
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
    elif args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
    else:
        # 使用getattr作为降级方案
        optimizer = getattr(torch.optim, args.optim)(
            param_groups,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )

    logging.info(f"Using {args.optim} optimizer with ve_lr={args.lr_ve:.2e}, ed_lr={args.lr_ed:.2e}")
    return optimizer


def build_lr_scheduler(args, optimizer):
    """构建学习率调度器 - 支持多种调度器"""

    if args.lr_scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
        logging.info(f"Using StepLR: step_size={args.step_size}, gamma={args.gamma}")

    elif args.lr_scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.step_size,
            T_mult=int(args.gamma)
        )
        logging.info(f"Using CosineAnnealingWarmRestarts: T_0={args.step_size}, T_mult={int(args.gamma)}")

    elif args.lr_scheduler == "CosineAnnealingLR":
        # 🔥 新增：支持CosineAnnealingLR调度器
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,  # 使用总epoch数作为周期
            eta_min=1e-6  # 最小学习率
        )
        logging.info(f"Using CosineAnnealingLR: T_max={args.epochs}, eta_min=1e-6")

    elif args.lr_scheduler == "ReduceLROnPlateau":
        # 🔥 新增：基于验证指标的自适应调度器
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # BLEU_4越大越好
            factor=0.5,  # 学习率衰减因子
            patience=3,  # 等待epoch数
            min_lr=1e-7,  # 最小学习率
            verbose=True  # 打印调整信息
        )
        logging.info(f"Using ReduceLROnPlateau: mode=max, factor=0.5, patience=3")

    elif args.lr_scheduler == "MultiStepLR":
        # 🔥 新增：多步长调度器
        milestones = [10, 20, 25]  # 在这些epoch调整学习率
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.gamma
        )
        logging.info(f"Using MultiStepLR: milestones={milestones}, gamma={args.gamma}")

    elif args.lr_scheduler == "ExponentialLR":
        # 🔥 新增：指数衰减调度器
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95  # 每个epoch乘以0.95
        )
        logging.info(f"Using ExponentialLR: gamma=0.95")

    else:
        # 🔥 降级方案：使用CosineAnnealingLR作为默认
        logging.warning(f"Unknown lr_scheduler '{args.lr_scheduler}', using CosineAnnealingLR as default")
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
        logging.info(f"Using default CosineAnnealingLR: T_max={args.epochs}, eta_min=1e-6")

    return lr_scheduler


# 🔥 新增：学习率调度器的step函数包装
def scheduler_step(scheduler, args, epoch=None, metric=None):
    """统一的调度器step函数"""

    if hasattr(scheduler, 'step'):
        # ReduceLROnPlateau需要传入metric
        if 'ReduceLROnPlateau' in str(type(scheduler)):
            if metric is not None:
                scheduler.step(metric)
            else:
                logging.warning("ReduceLROnPlateau scheduler needs metric, skipping step")
        else:
            # 其他调度器只需要step
            scheduler.step()
    else:
        logging.warning(f"Scheduler {type(scheduler)} has no step method")


# 🔥 新增：获取当前学习率的函数
def get_current_lr(optimizer):
    """获取当前学习率"""
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    return lrs


# 🔥 向后兼容函数（保持原有接口）
def build_optimizer_v1(args, model):
    """原版本的优化器构建函数"""
    ve_params = list(map(id, model.visual_extractor.parameters()))
    ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
    optimizer = getattr(torch.optim, args.optim)(
        [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
         {'params': ed_params, 'lr': args.lr_ed}],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer


def build_lr_scheduler_v1(args, optimizer):
    """原版本的学习率调度器构建函数"""
    if args.lr_scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)
    elif args.lr_scheduler == "cosine":
        print(f"Using CosineAnnealingWarmRestarts lr_scheduler, T_0={args.step_size}, T_mult={args.gamma}")
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=args.step_size,
                                                                            T_mult=int(args.gamma)
                                                                            )
    elif args.lr_scheduler == "CosineAnnealingLR":
        # 🔥 修复：添加对CosineAnnealingLR的支持
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
    else:
        raise NotImplementedError(f"Learning rate scheduler '{args.lr_scheduler}' not implemented")

    return lr_scheduler


def build_custom_lr_scheduler(args, optimizer):
    """构建自定义学习率调度器"""

    if args.lr_scheduler == 'CustomEpoch7':
        from custom_schedulers import Epoch7FocusedScheduler
        scheduler = Epoch7FocusedScheduler(optimizer, args)
        logging.info("Using Epoch7FocusedScheduler for fine-tuned training")

    elif args.lr_scheduler == 'MultiStepLR':
        milestones = getattr(args, 'milestones', [5, 8])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=getattr(args, 'gamma', 0.5)
        )
        logging.info(f"Using MultiStepLR: milestones={milestones}, gamma={getattr(args, 'gamma', 0.5)}")

    elif args.lr_scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=getattr(args, 'T_0', 4),
            T_mult=getattr(args, 'T_mult', 1),
            eta_min=1e-7
        )
        logging.info(f"Using CosineAnnealingWarmRestarts: T_0={getattr(args, 'T_0', 4)}")

    else:
        # 使用原有的调度器
        scheduler = build_lr_scheduler(args, optimizer)

    return scheduler


# 在 optimizers.py 中添加：

def build_custom_lr_scheduler(args, optimizer):
    """构建自定义学习率调度器"""

    if args.lr_scheduler == 'CustomEpoch7':
        from custom_schedulers import Epoch7FocusedScheduler
        scheduler = Epoch7FocusedScheduler(optimizer, args)
        logging.info("Using Epoch7FocusedScheduler for fine-tuned training")

    elif args.lr_scheduler == 'MultiStepLR':
        milestones = getattr(args, 'milestones', [5, 8])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=getattr(args, 'gamma', 0.5)
        )
        logging.info(f"Using MultiStepLR: milestones={milestones}, gamma={getattr(args, 'gamma', 0.5)}")

    elif args.lr_scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=getattr(args, 'T_0', 4),
            T_mult=getattr(args, 'T_mult', 1),
            eta_min=1e-7
        )
        logging.info(f"Using CosineAnnealingWarmRestarts: T_0={getattr(args, 'T_0', 4)}")

    else:
        # 使用原有的调度器
        scheduler = build_lr_scheduler(args, optimizer)

    return scheduler