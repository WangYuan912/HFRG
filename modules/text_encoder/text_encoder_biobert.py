import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 全局禁用 Tokenizer 多线程
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import numpy as np

from modules.Transformer import MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, Encoder, \
    EncoderLayer, Embeddings, SublayerConnection, clones


class TextEncoder(nn.Module):
    def __init__(self, model_name='dmis-lab/biobert-v1.1', num_labels=14, freeze=True):
        super(TextEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 本地加载模型和 Tokenizer
        local_model_path = "modules/biobert-v1.1"
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        self.bert = AutoModel.from_pretrained(local_model_path).to(self.device)

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, text_list):
        if isinstance(text_list, torch.Tensor):
            text_list = [self.tokenizer.decode(text, skip_special_tokens=True) for text in text_list.tolist()]

        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)

        pooled_output = outputs.pooler_output
        labels = self.classifier(pooled_output)
        return outputs.last_hidden_state, pooled_output, labels


class MHA_FF(nn.Module):  # 定义多头自注意力和前馈网络的组合模块
    def __init__(self, d_model, d_ff, h, dropout=0.1):  # 初始化方法
        super(MHA_FF, self).__init__()  # 调用父类的初始化方法
        self.self_attn = MultiHeadedAttention(h, d_model)  # 初始化多头自注意力机制
        self.sublayer = SublayerConnection(d_model, dropout)  # 初始化子层连接，包含残差连接和层归一化

    def forward(self, x, feats, mask=None):  # 前向传播方法
        x = self.sublayer(x, lambda x: self.self_attn(x, feats, feats))  # 使用子层连接处理输入，通过多头自注意力机制获取特征
        return x  # 返回处理后的特征
