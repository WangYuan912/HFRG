# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    # 初始化编码器层
    def __init__(self, size, self_attn, feed_forward, dropout):
        # 调用父类的初始化方法
        super(EncoderLayer, self).__init__()
        # 初始化自注意力机制
        self.self_attn = self_attn
        # 初始化前馈神经网络
        self.feed_forward = feed_forward
        # 创建两个子层连接，用于残差连接和层归一化
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # 存储层的大小
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        """Helper: Construct a model from hyperparameters."""  # 辅助函数：根据超参数构建模型
        c = copy.deepcopy  # 创建一个深拷贝函数的别名
        attn = MultiHeadedAttention(h, d_model, dropout)  # 创建多头注意力机制
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # 创建位置前馈网络
        position = PositionalEncoding(d_model, dropout)  # 创建位置编码
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),  # 构建编码器，包含N_enc个编码层
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N_dec),  # 构建解码器，包含N_dec个解码层
            lambda x: x,  # 用于源词汇的嵌入层，当前使用恒等函数
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))  # 用于目标词汇的嵌入层，包含嵌入层和位置编码
        )

        # This was important from their code.  # 这部分代码来自他们的实现，非常重要
        # Initialize parameters with Glorot / fan_avg.  # 使用Glorot初始化方法（也称为fan_avg）初始化模型参数
        for p in model.parameters():
            if p.dim() > 1:  # 只对维度大于1的参数进行初始化
                nn.init.xavier_uniform_(p)  # 使用Xavier均匀分布初始化参数
        return model  # 返回构建好的模型

    # 初始化Transformer模型
    def __init__(self, opt, tokenizer):
        super(TransformerModel, self).__init__(opt, tokenizer)  # 调用父类的初始化方法
        self.opt = opt  # 保存配置参数
        # self.config = yaml.load(open(opt.config_file))
        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)  # 获取编码器层数，如果未指定则使用opt.num_layers
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)  # 获取解码器层数，如果未指定则使用opt.num_layers
        self.d_model = getattr(opt, 'd_model', opt.d_model)  # 获取模型维度，如果未指定则使用opt.d_model
        self.d_ff = getattr(opt, 'd_ff', opt.d_ff)  # 获取前馈网络维度，如果未指定则使用opt.d_ff
        self.h = getattr(opt, 'num_heads', 8)  # 获取注意力头数，如果未指定则使用8
        self.dropout = getattr(opt, 'dropout', 0.1)  # 获取dropout概率，如果未指定则使用0.1
        tgt_vocab = self.vocab_size + 1  # 计算目标词汇表大小
        # 定义注意力嵌入层
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.d_model),
                 # nn.ReLU(),
                 nn.Dropout(self.dropout)) +
                ((nn.BatchNorm1d(self.d_model),) if self.use_bn == 2 else ())))
        self.embed = lambda x: x  # 定义嵌入层，这里使用lambda表达式
        self.fc_embed = lambda x: x  # 定义前馈嵌入层，这里使用lambda表达式
        self.logit = nn.Linear(self.d_model, tgt_vocab)  # 定义输出层，将模型维度映射到目标词汇表大小
        # 创建Transformer模型
        self.model = self.make_model(0, tgt_vocab,
                                     N_enc=self.N_enc,
                                     N_dec=self.N_dec,
                                     d_model=self.d_model,
                                     d_ff=self.d_ff,
                                     h=self.h,
                                     dropout=self.dropout)

    def init_hidden(self, bsz):
        return []

    # 准备特征的方法
    # 参数: fc_feats: 全连接特征，att_feats: 注意力特征，att_masks: 注意力掩码
    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats,
                                                                            att_masks)  # 调用前向特征准备方法，处理注意力特征和掩码
        memory = self.model.encode(att_feats, att_masks)  # 使用模型的编码器对处理后的注意力特征进行编码
        return fc_feats[..., :0], att_feats[..., :0], memory, att_masks  # 返回全连接特征的前0个元素，注意力特征的前0个元素，编码后的记忆和注意力掩码

    def _prepare_feature_forward(self, att_feats, att_masks=None,
                                 seq=None):  # att_feats: 注意力特征；att_masks: 注意力掩码seq: 序列，
        att_feats, att_masks = self.clip_att(att_feats, att_masks)  # 裁剪特征和掩码，确保它们的形状一致
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)  # 对特征进行嵌入处理
        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)  # 如果掩码为None，创建一个全1的掩码
        att_masks = att_masks.unsqueeze(-2)  # 增加一个维度以适应模型输入
        if seq is not None:
            seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)  # 创建序列掩码，排除结束符和填充符
            seq_mask[:, 0] = 1  # 确保开始符始终被包含
            seq_mask = seq_mask.unsqueeze(-2)  # 增加一个维度以适应模型输入
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)  # 生成后续掩码并将其与序列掩码结合

            seq_per_img = seq.shape[0] // att_feats.shape[0]  # 计算每个图像的序列数量
            if seq_per_img > 1:
                att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                                                            [att_feats, att_masks])  # 如果每个图像的序列数量大于1，重复特征和掩码
        else:
            seq_mask = None  # 如果序列为None，序列掩码也为None

        return att_feats, seq, att_masks, seq_mask  # 返回处理后的特征、序列、特征掩码和序列掩码

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        if seq.ndim == 3:  # 检查seq的维度，如果为3则重塑为2维
            seq = seq.reshape(-1, seq.shape[2])
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)  # 准备前向传播所需的特征和掩码
        out = self.model(att_feats, seq, att_masks, seq_mask)  # 通过模型进行前向传播
        outputs = F.log_softmax(self.logit(out), dim=-1)  # 应用log_softmax函数，计算输出的对数概率
        return outputs  # 返回计算后的输出
        # 返回拼接后的输出（注释掉的代码）
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]
