import torch
import torch.nn as nn
from transformers import AutoModel

from modules.Transformer import MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, Encoder, \
    EncoderLayer, Embeddings, SublayerConnection

class TextEncoder(nn.Module):
    def __init__(self, d_model, d_ff, num_layers, tgt_vocab, num_labels=14, h=8, dropout=0.1, bos_idx=0, eos_idx=0, pad_idx=0):
        super(TextEncoder, self).__init__()
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # Bio_ClinicalBERT 768维输出
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        # 降维：768 → d_model (512)
        self.proj = nn.Linear(768, d_model)

        # Transformer Encoder
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, attn, ff, dropout), num_layers)

        # 分类器（池化后用）
        self.classifier = nn.Linear(d_model, num_labels)

    def prepare_mask(self, seq):
        seq_mask = (seq != self.eos_idx) & (seq != self.pad_idx)
        seq_mask[:, 0] = 1  # 强制第一个位置是valid
        seq_mask = seq_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        return seq_mask  # (B, 1, 1, S)

    def forward(self, src, labels=None):
        print("Input src shape:", src.shape)  # (batch_size, seq_len)
        src_mask = self.prepare_mask(src)
        print("Src mask shape:", src_mask.shape)

        bert_outputs = self.bert(src)
        if isinstance(bert_outputs, dict):
            bert_feats = bert_outputs['last_hidden_state']  # (B, S, 768)
        else:
            bert_feats = bert_outputs[0]

        print("BERT output shape:", bert_feats.shape)

        proj_feats = self.proj(bert_feats)  # (B, S, d_model)
        print("Projected features shape:", proj_feats.shape)

        feats = self.encoder(proj_feats, src_mask)  # (B, S, d_model)
        print("Encoder output feats shape:", feats.shape)

        pooled_output = feats.mean(dim=1)  # 池化，得到 (B, d_model)
        print("Pooled output shape:", pooled_output.shape)

        # 分类器输出
        logits = self.classifier(pooled_output)  # (B, num_labels)

        return feats, pooled_output, labels if labels is not None else logits

class MHA_FF(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout=0.1):
        super(MHA_FF, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.sublayer = SublayerConnection(d_model, dropout)

    def forward(self, x, feats, mask=None):
        x = self.sublayer(x, lambda x_: self.self_attn(x_, feats, feats, mask))
        return x
