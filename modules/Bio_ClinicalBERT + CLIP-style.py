import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    def __init__(self, num_labels=14):
        super(TextEncoder, self).__init__()

        # 加载Bio_ClinicalBERT
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        hidden_size = self.bert.config.hidden_size  # 通常是768

        # 分类器（保持你原来结构）
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, src, attention_mask=None):
        if attention_mask is None:
            attention_mask = (src != 0).long()

        outputs = self.bert(input_ids=src, attention_mask=attention_mask)
        feats = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        pooled_output = feats[:, 0, :]     # 取CLS位 (batch_size, hidden_size)
        labels = self.classifier(pooled_output)

        return feats, pooled_output, labels


class CLIPContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(CLIPContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 相似度矩阵
        logits_per_image = image_features @ text_features.t() / self.temperature
        logits_per_text = logits_per_image.t()

        # 标签是对角线
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size).to(image_features.device)

        # 双向交叉熵
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)

        return (loss_i + loss_t) / 2


# 假设你有图像特征提取器 (用ResNet/Swin/CLIP融合好的)
class DummyImageEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super(DummyImageEncoder, self).__init__()
        self.encoder = nn.Linear(2048, output_dim)  # 假设原特征2048维，降到768

    def forward(self, image_input):
        # image_input: (batch_size, 2048)
        return self.encoder(image_input)


# =============================
# ✅ 整体组装
# =============================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模块初始化
    text_encoder = TextEncoder(num_labels=14).to(device)
    image_encoder = DummyImageEncoder(output_dim=768).to(device)
    contrastive_loss_fn = CLIPContrastiveLoss(temperature=0.07)

    # 假设输入
    batch_size = 8
    seq_len = 128
    vocab_size = 30522  # BERT默认词表大小
    image_feat_dim = 2048  # ResNet/Swin输出维度

    # 文本输入（token ids）
    src = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # 图像输入（假设已提取好的原始特征）
    image_input = torch.randn(batch_size, image_feat_dim).to(device)

    # =============================
    # ✅ 前向传播
    # =============================
    # 文本特征提取
    feats, pooled_output, labels = text_encoder(src)

    # 图像特征提取
    image_features = image_encoder(image_input)  # (batch_size, 768)

    # =============================
    # ✅ CLIP-style 对比损失
    # =============================
    clip_loss = contrastive_loss_fn(image_features, pooled_output)

    print(f"CLIP Contrastive Loss: {clip_loss.item():.4f}")

    # =============================
    # ✅ labels 也可以直接用作下游分类损失
    # =============================
    dummy_gt_labels = torch.randint(0, 2, (batch_size, 14)).float().to(device)
    classification_loss = F.binary_cross_entropy_with_logits(labels, dummy_gt_labels)

    print(f"Classification Loss: {classification_loss.item():.4f}")

    # =============================
    # ✅ 总Loss
    # =============================
    total_loss = clip_loss + classification_loss
    print(f"Total Loss: {total_loss.item():.4f}")