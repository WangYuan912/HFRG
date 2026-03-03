#!/bin/bash

# 目标: BLEU_1>39.6%, BLEU_4>11.6%, CIDEr>0.116
# 策略: 基于失败案例分析的激进优化

python main.py \
    --version CLIP \
    --use_clip \
    --visual_extractor resnet101 \
    --visual_extractor_pretrained True \
    --clip_model_name ViT-B/32 \
    --freeze_clip \
    \
    `# === CLIP权重激进优化 ===` \
    --clip_weight 0.4 \
    --global_align_weight 0.2 \
    --local_align_weight 0.2 \
    --temperature_init 0.03 \
    --clip_feature_dim 512 \
    \
    `# === 超低学习率长期训练策略 ===` \
    --lr_ve 2.5e-5 \
    --lr_ed 2.5e-5 \
    --lr_scheduler CosineAnnealingLR \
    --step_size 15 \
    --gamma 0.8 \
    \
    `# === 强化正则化策略 ===` \
    --dropout 0.08 \
    --drop_prob_lm 0.05 \
    --weight_decay 5e-5 \
    --label_smoothing 0.05 \
    --grad_clip 1.5 \
    \
    `# === 超长训练策略 ===` \
    --epochs 100 \
    --early_stop 50 \
    --batch_size 8 \
    --monitor_mode max \
    --monitor_metric CIDEr \
    --save_period 1 \
    \
    `# === 强化多任务学习 ===` \
    --label_loss \
    --rank_loss \
    \
    `# === 激进生成策略 ===` \
    --beam_size 7 \
    --block_trigrams 1 \
    --sample_method beam_search \
    --decoding_constraint 0 \
    --output_logsoftmax 1 \
    --temperature 0.8 \
    \
    `# === 扩展模型架构 ===` \
    --num_memory_heads 16 \
    --rm_num_heads 16 \
    --d_ff 1024 \
    --d_model 768 \
    --num_heads 12 \
    --num_layers 10 \
    --num_slots 100 \
    --num_labels 14 \
    --rm_d_model 768 \
    \
    `# === 序列长度优化 ===` \
    --max_seq_length 100 \
    --threshold 1 \
    --bos_idx 0 \
    --eos_idx 0 \
    --pad_idx 0 \
    \
    `# === 数据处理优化 ===` \
    --crop_size 224 \
    --image_size 256 \
    --num_workers 4 \
    \
    `# === 优化器精调 ===` \
    --optim Adam \
    --amsgrad True \
    \
    `# === 基础配置 ===` \
    --dataset_name mimic_cxr \
    --gpu 0 \
    --seed 2024 \
    --n_gpu 1 \
    \
    `# === 高级参数优化 ===` \
    --use_bn 0 \
    --logit_layers 1 \
    --sample_n 1 \
    --group_size 1 \
    --test_steps 0 \
    --alpha 0.3 \
    \
    `# === 数据路径 ===` \
    --image_dir data/MIMIC-CXR/images/ \
    --ann_path data/MIMIC-CXR/annotation.json \
    --label_path data/labels/MIMIC/MIMIC_lab.csv
