import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from modules.Transformer import MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, Encoder, \
    EncoderLayer, Embeddings, SublayerConnection, clones

class TextEncoder(nn.Module):
    def __init__(self, pretrained_model_name='emilyalsentzer/Bio_ClinicalBERT', num_labels=14, d_model=512):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        # 假设bert hidden_size不等于d_model，需加线性层映射
        self.d_model = d_model
        self.hidden_size = self.bert.config.hidden_size
        if self.hidden_size != d_model:
            self.linear_proj = nn.Linear(self.hidden_size, d_model)
        else:
            self.linear_proj = nn.Identity()
        self.classifier = nn.Linear(d_model, num_labels)

        # eos_idx和pad_idx可以从tokenizer获取
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.eos_idx = self.tokenizer.sep_token_id  # 通常BERT的[SEP]作为结束符
        self.pad_idx = self.tokenizer.pad_token_id

    def prepare_mask(self, seq):
        # seq shape: [batch_size, seq_len]
        seq_mask = (seq != self.eos_idx) & (seq != self.pad_idx)
        seq_mask[:, 0] = True  # 保证第一位mask为True
        # BERT expects mask shape [batch_size, seq_len], 转成 [batch_size, 1, seq_len] 保持兼容
        return seq_mask.unsqueeze(1)

    def forward(self, src):
        # src shape: [batch_size, seq_len], dtype=torch.long
        #print(f"Input src shape: {src.shape}")  # [16, 51]

        src_mask = self.prepare_mask(src)
        #print(f"src_mask shape: {src_mask.shape}")  # [16, 1, 51]

        # BERT的attention_mask是[batch_size, seq_len]且bool型或0/1
        bert_attention_mask = src_mask.squeeze(1).to(src.device).long()  # [batch_size, seq_len]

        outputs = self.bert(input_ids=src, attention_mask=bert_attention_mask)
        # outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
        feats = outputs.last_hidden_state
        feats = self.linear_proj(feats)  # 映射到d_model
        #print(f"Encoder output feats shape: {feats.shape}")  # [16, 51, 512]

        pooled_output = feats[:, 0, :]  # 取[CLS]对应位置作为池化输出
        #print(f"Pooled output shape: {pooled_output.shape}")  # [16, 512]

        labels = self.classifier(pooled_output)
        #print(f"Classifier output labels shape: {labels.shape}")  # [16, 14]

        return feats, pooled_output, labels

class MHA_FF(nn.Module):  # 定义多头自注意力和前馈网络的组合模块
    def __init__(self, d_model, d_ff, h, dropout=0.1):  # 初始化方法
        super(MHA_FF, self).__init__()  # 调用父类的初始化方法
        self.self_attn = MultiHeadedAttention(h, d_model)  # 初始化多头自注意力机制
        self.sublayer = SublayerConnection(d_model, dropout)  # 初始化子层连接，包含残差连接和层归一化

    def forward(self, x, feats, mask=None):  # 前向传播方法
        x = self.sublayer(x, lambda x: self.self_attn(x, feats, feats))  # 使用子层连接处理输入，通过多头自注意力机制获取特征
        return x  # 返回处理后的特征