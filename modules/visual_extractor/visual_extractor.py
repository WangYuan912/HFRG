import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# ResNet-101 作为 Backbone
# 预训练的 ResNet-101 作为特征提取器，去掉最后的全连接层，只保留 conv5 层的特征。
# 输出大小为 (16, 2048, 7, 7)。
# 添加 CoTAttention
# 采用 CoTAttention 进行上下文建模，增强局部和全局的特征提取能力。
#
#特征转换
# patch_feats：ResNet + CoT 之后的 (16, 2048, 7, 7) 重新排列为 (16, 49, 2048)
# avg_feats：全局平均池化为 (16, 2048)
# labels：通过 FC 层输出 (16, 14)

class CoTAttention(nn.Module):
    def __init__(self, dim=2048, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape

        k1 = self.key_embed(x)  # 静态上下文 key
        v = self.value_embed(x).view(bs, c, -1)  # Value 矩阵

        y = torch.cat([k1, x], dim=1)  # 拼接 key 和 query

        att = self.attention_embed(y)
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)

        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2


class VisualExtractor(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.resnet = models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # 移除全连接层 保留卷积特征提取部分
        self.cot = CoTAttention(dim=2048)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        num_classes = 14
        print(f"num_classes: {num_classes}")
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # 输入: (16, 3, 224, 224)
        x = self.resnet(x)  # (16, 2048, 7, 7)
        x = self.cot(x)  # CoTAttention 处理后的特征 (16, 2048, 7, 7)
        bs, c, h, w = x.shape  # (16, 2048, 7, 7)
        patch_feats = x.view(bs, c, h * w).permute(0, 2, 1)  # (16, 49, 2048)   (batch_size=16, num_patches=49, feature_dim=2048)
        avg_feats = self.avg_pool(x).view(bs, -1)  # (16, 2048)  (batch_size=16, feature_dim=2048)
        labels = self.fc(avg_feats)  # (16, 14)  (batch_size=16, 14)
        return patch_feats, avg_feats, labels
