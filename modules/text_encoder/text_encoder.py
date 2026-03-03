import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import numpy as np

from modules.Transformer import MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, Encoder, \
    EncoderLayer, Embeddings, SublayerConnection, clones

class TextEncoder(nn.Module):
    # 初始化TextEncoder类
    def __init__(self, d_model, d_ff, num_layers, tgt_vocab, num_labels=14, h=3, dropout=0.1):
        super(TextEncoder, self).__init__()  # 调用父类的构造函数
        # TODO:
        #  将eos,pad的index改为参数输入
        self.eos_idx = 0
        self.pad_idx = 0  # 设置结束符的索引
        attn = MultiHeadedAttention(h, d_model)  # 创建多头注意力机制
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # 创建位置前馈网络
        position = PositionalEncoding(d_model, dropout)  # 创建位置编码
        self.classifier = nn.Linear(d_model, num_labels)  # 创建分类器，将模型输出映射到标签数量
        self.encoder = Encoder(EncoderLayer(d_model, attn, ff, dropout), num_layers)  # 创建编码器，包含多个编码层
        self.src_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), position)  # 创建源嵌入层，包含词嵌入和位置编码

    # 准备序列的掩码
    # 参数: seq: 输入的序列
    # 返回: 序列的掩码
    def prepare_mask(self, seq):  # 定义准备掩码的方法
        seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)  # 创建掩码，将序列中不等于结束符和填充符的位置设为1，其余位置设为0
        seq_mask[:, 0] = 1  # 将掩码的第一列设为1，表示开始符
        seq_mask = seq_mask.unsqueeze(-2)  # 增加一个维度，使掩码适合模型输入
        return seq_mask  # 返回准备好的掩码

    # 前向传播方法
    # 参数: src: 输入的源数据
    # 返回: 特征、池化输出和标签
    def forward(self, src):  # 定义前向传播方法



        src_mask = self.prepare_mask(src)  # 准备源数据的掩码
        feats = self.encoder(self.src_embed(src), src_mask)  # 将嵌入后的源数据和掩码输入编码器，获取特征
        pooled_output = feats[:, 0, :]  # 获取池化输出，即特征的第一个位置
        labels = self.classifier(pooled_output)  # 将池化输出输入分类器，获取标签

        # print(f"Input src shape: {src.shape}")  # 输入序列的维度   [16, 51]
        # print(f"src_mask shape: {src_mask.shape}")  # 掩码维度  [16, 1, 51
        # print(f"Encoder output feats shape: {feats.shape}")  # 编码器输出维度  [16, 51, 512]
        # print(f"Pooled output shape: {pooled_output.shape}")  # 池化输出维度  [16, 512]
        # print(f"Classifier output labels shape: {labels.shape}")  # 标签输出维度  [16, 14]

        return feats, pooled_output, labels  # 返回特征、池化输出和标签


class MHA_FF(nn.Module):  # 定义多头自注意力和前馈网络的组合模块
    def __init__(self, d_model, d_ff, h, dropout=0.1):  # 初始化方法
        super(MHA_FF, self).__init__()  # 调用父类的初始化方法
        self.self_attn = MultiHeadedAttention(h, d_model)  # 初始化多头自注意力机制
        self.sublayer = SublayerConnection(d_model, dropout)  # 初始化子层连接，包含残差连接和层归一化

    def forward(self, x, feats, mask=None):  # 前向传播方法
        x = self.sublayer(x, lambda x: self.self_attn(x, feats, feats))  # 使用子层连接处理输入，通过多头自注意力机制获取特征
        return x  # 返回处理后的特征
