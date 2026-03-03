#!/bin/bash

# 医学报告生成训练脚本 V3.0 - 保守优化版本
# 基于第一次实验成功经验的微调改进

python main.py \
    --use_clip \
    --visual_extractor resnet101 \
    --clip_model_name ViT-B/32 \
    --freeze_clip \
    \
    `# 保持第一次成功的CLIP权重设置` \
    --clip_weight 0.25 \
    --global_align_weight 0.12 \
    --local_align_weight 0.12 \
    --temperature_init 0.07 \
    \
    `# 温和的正则化改进（在原基础上微调）` \
    --dropout 0.25 \
    --drop_prob_lm 0.12 \
    --weight_decay 2e-4 \
    --label_smoothing 0.12 \
    \
    `# 优化学习率调度（保持CosineAnnealingLR但缩短周期）` \
    --lr_ve 4e-5 \
    --lr_ed 4e-5 \
    --lr_scheduler CosineAnnealingLR \
    --step_size 15 \
    --gamma 0.8 \
    \
    `# 适中的早停策略` \
    --epochs 20 \
    --early_stop 12 \
    --batch_size 14 \
    --monitor_metric CIDEr \
    \
    `# 保持原有成功的模型架构` \
    --num_memory_heads 8 \
    --rm_num_heads 8 \
    --d_ff 512 \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 6 \
    \
    `# 基础配置` \
    --dataset_name iu_xray \
    --beam_size 3 \
    --gpu 0
