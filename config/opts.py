from __future__ import print_function
import argparse
import os
import torch
import json
from datetime import datetime, timedelta
from misc.utils import set_logging
import logging


def parse_opt(prefix=None):
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/LGK/images/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/LGK/LGK_ann.json',
                        help='the path to the directory containing the data.')
    parser.add_argument('--label_path', type=str, default='data/labels/LGK/LGK_lab.csv',
                        help='the path to the directory containing the label.')
    parser.add_argument('--image_size', type=int, default=256, help='')
    parser.add_argument('--crop_size', type=int, default=224, help='')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='LGK',
                        choices=['iu_xray', 'mimic_cxr', 'covid', 'covidall', 'LGK'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=4, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('-v', '--version', type=str, default="CLIP", help='main model version')
    parser.add_argument('--visual_extractor', type=str, default='resnet101',
                        choices=['densenet', 'efficientnet', 'resnet101'],
                        help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,
                        help='whether to load the pretrained visual extractor')
    parser.add_argument('--pretrain_cnn_file', type=str, default='', help='the visual extractor to be used.')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=1280, help='for densenet = 1024, for efficientnet = 1280')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_memory_heads', type=int, default=8, help='the number of heads in memory.')
    parser.add_argument('--num_layers', type=int, default=6, help='the number of layers of Transformer.')
    parser.add_argument('--num_labels', type=int, default=14, help='the number of labels.')

    # 🔧 优化1: 增强正则化 - 提高dropout率防止过拟合
    parser.add_argument('--dropout', type=float, default=0.35, help='the dropout rate of Transformer (增强版).')

    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')

    # 🔧 优化2: 增强正则化 - 提高输出层dropout率
    parser.add_argument('--drop_prob_lm', type=float, default=0.15,
                        help='the dropout rate of the output layer (增强版).')

    # CLIP Enhancement Settings - 优化CLIP权重防止过拟合
    parser.add_argument('--use_clip', action='store_true', default=False,
                        help='whether to use CLIP enhancement')
    parser.add_argument('--clip_model_name', type=str, default='ViT-B/32',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                        help='CLIP model variant to use')
    parser.add_argument('--clip_feature_dim', type=int, default=512,
                        help='CLIP feature dimension')

    # 🔧 优化3: 降低CLIP权重防止过度依赖CLIP特征
    parser.add_argument('--clip_weight', type=float, default=0.2,
                        help='weight for CLIP contrastive loss (降低版)')
    parser.add_argument('--global_align_weight', type=float, default=0.1,
                        help='weight for global alignment loss (降低版)')
    parser.add_argument('--local_align_weight', type=float, default=0.1,
                        help='weight for local alignment loss (降低版)')

    parser.add_argument('--temperature_init', type=float, default=0.07,
                        help='initial temperature for contrastive learning')
    parser.add_argument('--freeze_clip', action='store_true', default=True,
                        help='whether to freeze CLIP parameters')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings - 优化训练参数
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=15, help='the number of training epochs (减少版).')  # 减少epoch数
    parser.add_argument('--save_dir', type=str, default='results/', help='the patch to save the models.')
    parser.add_argument('--expe_name', type=str, default='', help='extra experiment name')
    parser.add_argument('--record_dir', type=str, default='records/',
                        help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'],
                        help='whether to max or min the metric.')

    # 🔧 优化4: 改变监控指标为CIDEr，更好地反映生成质量
    parser.add_argument('--monitor_metric', type=str, default='CIDEr', help='the metric to be monitored (改为CIDEr).')

    # 🔧 优化5: 激进早停 - 从25改为3，快速停止过拟合
    parser.add_argument('--early_stop', type=int, default=3, help='the patience of training (激进早停).')

    # 🔧 优化6: 增强标签平滑
    parser.add_argument('--label_smoothing', type=float, default=0.15)

    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--grad_clip', type=float, default=5)
    parser.add_argument('--test_steps', type=int, default=0, help='the period of test in training')

    # Optimization - 优化学习率和正则化
    parser.add_argument('--label_loss', default=False, action='store_true')
    parser.add_argument('--rank_loss', default=False, action='store_true')
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')

    # 🔧 优化7: 降低学习率防止过拟合
    parser.add_argument('--lr_ve', type=float, default=2e-5,
                        help='the learning rate for the visual extractor (降低版).')
    parser.add_argument('--lr_ed', type=float, default=2e-5,
                        help='the learning rate for the remaining parameters (降低版).')

    # 🔧 优化8: 增强权重衰减
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='the weight decay (增强版).')

    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler - 优化学习率调度
    # 🔧 优化9: 使用ReduceLROnPlateau自适应调整学习率
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau',
                        choices=['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'],
                        help='the type of the learning rate scheduler (改为自适应).')

    parser.add_argument('--step_size', type=int, default=10, help='the step size of the learning rate scheduler.')

    # 🔧 优化10: 更激进的学习率衰减
    parser.add_argument('--gamma', type=float, default=0.5, help='the gamma of the learning rate scheduler (激进版).')

    # ReduceLROnPlateau相关参数
    parser.add_argument('--patience', type=int, default=2, help='patience for ReduceLROnPlateau.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--gpu', type=str, default='0', help='')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    # for Relational Memory
    parser.add_argument('--num_slots', type=int, default=60, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # config
    parser.add_argument('--cfg', type=str, default=None,
                        help='configuration; similar to what is used in detectron')
    parser.add_argument(
        '--set_cfgs', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]\n This has higher priority'
             'than cfg file but lower than other args. (You can only overwrite'
             'arguments that have alerady been defined in config file.)',
        default=[], nargs='+')

    # step 1: read cfg_fn
    args = parser.parse_args()
    if args.cfg is not None or args.set_cfgs is not None:
        from .config import CfgNode
        if args.cfg is not None:
            cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        else:
            cn = CfgNode()
        if args.set_cfgs is not None:
            cn.merge_from_list(args.set_cfgs)
        for k, v in cn.items():
            if not hasattr(args, k):
                print('Warning: key %s not in args' % k)
            setattr(args, k, v)
        args = parser.parse_args(namespace=args)

    # Check if args are valid
    assert args.d_model > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.d_vf > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.beam_size > 0, "beam_size should be greater than 0"

    # add training parameters to save dir
    if args.resume == None or args.expe_name != '':
        clip_suffix = "_CLIP" if args.use_clip else ""
        expe_name = f"V{args.version}_" \
                    + args.visual_extractor \
                    + clip_suffix \
                    + ("_labelloss" if args.label_loss else "") \
                    + ("_rankloss" if args.rank_loss else "") \
                    + ("_" + args.expe_name if args.expe_name != "" else "") \
                    + (datetime.now() + timedelta(hours=15)).strftime("_%Y%m%d-%H%M%S")
        expe_name = prefix + '_' + expe_name if prefix else expe_name
        args.save_dir = os.path.join(args.save_dir, args.dataset_name, expe_name)
    else:
        args.save_dir = os.path.split(args.resume)[0]
        expe_name = os.path.split(args.save_dir)[1]

    # Save config for reproduce
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    set_logging(os.path.join(args.save_dir, f'{expe_name}.log'))
    logging.info(f'Logging Dir: {args.save_dir}')

    # modify visual feature projection dimension
    if args.visual_extractor == "efficientnet" and args.d_vf != 1280:
        args.d_vf = 1280
    elif args.visual_extractor == "densenet" and args.d_vf != 1024:
        args.d_vf = 1024
    elif args.visual_extractor == "resnet101" and args.d_vf != 2048:
        args.d_vf = 2048
    logging.info(f"Visual Extractor:{args.visual_extractor}   d_vf: {args.d_vf}")

    # CLIP参数验证和设置
    if args.use_clip or args.version == "CLIP":
        args.use_clip = True  # 确保开启CLIP
        logging.info(f" CLIP Enhancement: Enabled with model {args.clip_model_name}")
        logging.info(
            f" CLIP weights (优化版) - contrastive: {args.clip_weight}, global: {args.global_align_weight}, local: {args.local_align_weight}")
        # 确保device参数存在
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        logging.info("CLIP Enhancement: Disabled")

    # 🔧 优化信息日志
    logging.info(f" Anti-Overfitting Optimizations Applied:")
    logging.info(f"   Early Stop: {args.early_stop} (激进早停)")
    logging.info(f"   Learning Rate: ve={args.lr_ve}, ed={args.lr_ed} (降低学习率)")
    logging.info(f"   Dropout: {args.dropout}, drop_prob_lm={args.drop_prob_lm} (增强正则化)")
    logging.info(f"   Weight Decay: {args.weight_decay} (增强权重衰减)")
    logging.info(f"   LR Scheduler: {args.lr_scheduler} (自适应调度)")
    logging.info(f"   Monitor Metric: {args.monitor_metric} (CIDEr监控)")

    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logging.info(f"==>> Set GPU devices: {args.gpu}")
    logging.info(f"==>> DataLoader num_workers: {args.num_workers}")

    return args


def add_eval_options(parser):
    pass


def add_diversity_opts(parser):
    pass


def add_eval_sample_opts(parser):
    pass


if __name__ == '__main__':
    import sys

    sys.argv = [sys.argv[0]]
    args = parse_opt()
    print(args)
    print()
    sys.argv = [sys.argv[0], '--cfg', 'covid.yml']
    args1 = parse_opt()
    print(dict(set(vars(args1).items()) - set(vars(args).items())))
    print()
    sys.argv = [sys.argv[0], '--cfg', 'covid.yml', '--visual_extractor', 'densenet']
    args2 = parse_opt()
    print(dict(set(vars(args2).items()) - set(vars(args1).items())))