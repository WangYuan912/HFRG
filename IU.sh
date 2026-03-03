#!/bin/bash



python main.py \
    --use_clip \
    --visual_extractor resnet101 \
    --clip_model_name ViT-B/32 \
    --freeze_clip \
    --clip_weight 0.25 \
    --global_align_weight 0.12 \
    --local_align_weight 0.12 \
    --temperature_init 0.07 \
    --dropout 0.25 \
    --drop_prob_lm 0.12 \
    --weight_decay 2e-4 \
    --label_smoothing 0.12 \
    --lr_ve 4e-5 \
    --lr_ed 4e-5 \
    --lr_scheduler CosineAnnealingLR \
    --step_size 15 \
    --gamma 0.8 \
    --epochs 20 \
    --early_stop 12 \
    --batch_size 14 \
    --monitor_metric CIDEr \
    --num_memory_heads 8 \
    --rm_num_heads 8 \
    --d_ff 512 \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 6 \
    --dataset_name iu_xray \
    --beam_size 3 \
    --gpu 0
