import warnings
warnings.simplefilter("ignore", UserWarning)

import logging
import torch
import numpy as np
import platform

from modules.tokenizers import Tokenizer
from modules.dataloaders import LADataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
import models
from config import opts



#python main.py --version CLIP --use_clip --dataset_name mimic_cxr --clip_model_name "ViT-B/32"
#python main.py --version CLIP --use_clip --dataset_name iu_xray --clip_model_name "ViT-B/32"
#python main.py --version CLIP --use_clip --dataset_name LGK --clip_model_name "ViT-B/32"

def main():
    # parse arguments
    args = opts.parse_opt()

    # 设置CLIP相关的默认参数
    if args.version == "CLIP":
        args.use_clip = True
        logging.info("CLIP mode detected, enabling CLIP enhancement")

    logging.info(str(args))

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # ✅ 在 Windows 下也允许使用多进程，只要写好了 __main__
    logging.info(f"Platform: {platform.system()}, using num_workers = {args.num_workers}")

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = LADataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = LADataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = LADataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture - 根据是否使用CLIP选择模型
    if getattr(args, 'use_clip', False) or args.version == "CLIP":
        model_name = "LAMRGModel_vCLIP"
        logging.info(f"Using CLIP-enhanced model: {model_name}")

        # 确保必要的CLIP参数存在
        args.use_clip = getattr(args, 'use_clip', True)
        args.clip_weight = getattr(args, 'clip_weight', 0.25)
        args.global_align_weight = getattr(args, 'global_align_weight', 0.12)
        args.local_align_weight = getattr(args, 'local_align_weight', 0.12)
        args.clip_feature_dim = getattr(args, 'clip_feature_dim', 512)
        args.clip_model_name = getattr(args, 'clip_model_name', 'ViT-B/32')
        args.temperature_init = getattr(args, 'temperature_init', 0.07)
        args.freeze_clip = getattr(args, 'freeze_clip', True)
        args.device = getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')

        logging.info("CLIP Configuration:")
        logging.info(f"  - Model: {args.clip_model_name}")
        logging.info(f"  - Feature dim: {args.clip_feature_dim}")
        logging.info(f"  - Contrastive weight: {args.clip_weight}")
        logging.info(f"  - Global align weight: {args.global_align_weight}")
        logging.info(f"  - Local align weight: {args.local_align_weight}")
        logging.info(f"  - Temperature: {args.temperature_init}")
        logging.info(f"  - Freeze CLIP: {args.freeze_clip}")

    else:
        model_name = f"LAMRGModel_v{args.version}"
        logging.info(f"Using standard model: {model_name}")

    logging.info(f"Model name: {model_name} \tModel Layers: {args.num_layers}")

    # 检查模型类是否存在
    if not hasattr(models, model_name):
        logging.error(f"Model {model_name} not found in models module!")
        if model_name == "LAMRGModel_vCLIP":
            logging.error("Please ensure LAMRGModel_vCLIP is properly defined in models/lamrg.py")
        raise AttributeError(f"Model {model_name} not found")

    # 创建模型
    try:
        model = getattr(models, model_name)(args, tokenizer)
        logging.info(f"Model {model_name} created successfully")

        # 打印模型参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")

        # CLIP 初始化检查
        if getattr(args, 'use_clip', False) and hasattr(model, 'clip_alignment'):
            if model.clip_alignment is not None:
                logging.info("CLIP alignment module initialized successfully")
                if model.clip_alignment.clip_model is not None:
                    logging.info("CLIP model loaded successfully")
                else:
                    logging.warning("CLIP model failed to load, using zero features")
            else:
                logging.warning("CLIP alignment module is None")

    except Exception as e:
        logging.error(f"Failed to create model {model_name}: {e}")
        raise

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler,
                      train_dataloader, val_dataloader, test_dataloader)

    # 开始训练
    logging.info("Starting training...")
    try:
        trainer.train()
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

    logging.info(str(args))


# ✅ 这是必须的：Windows 多进程必须从这里启动
if __name__ == '__main__':
    main()
