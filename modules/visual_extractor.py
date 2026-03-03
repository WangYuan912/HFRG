import torch
import torch.nn as nn
from medclip import MedCLIPModel, MedCLIPVisionModel
from torch.nn import functional as F
import os

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
    def __init__(self, args):
        super().__init__()
        self.cot_layer = CoTAttention(dim=2048)  # 注册 CoT 模块
        num_classes = getattr(args, 'num_classes', 14)  #从 args 中读取 num_classes 参数，若未定义则默认为 14 类

        # 初始化 MedCLIP 模型
        self.medclip_model = MedCLIPModel(vision_cls=MedCLIPVisionModel) #初始化一个 MedCLIP 模型，使用其内置的视觉编码器类 MedCLIPVisionModel
        #加载本地保存的 MedCLIP 预训练模型权重（.bin 格式 PyTorch 模型文件）
        ckpt_path = os.path.expanduser("C:\\Users\\25318\\.cache\\medclip\\medclip-pretrained\\pytorch_model.bin")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        #strict=False 允许加载不完全匹配的权重（避免版本差异或未使用的组件报错）。
        self.medclip_model.load_state_dict(state_dict, strict=False)

        # 冻结视觉编码器参数
        # for p in self.medclip_model.vision_model.parameters():
        #     p.requires_grad = False
        # 自定义池化和分类头
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #将 7×7 的特征图池化为 1×1（即得到每张图一个 2048 维向量）。
        self.classifier = nn.Linear(2048, num_classes)  #用于图像分类任务。

    def forward(self, x):
        # 获取 ResNet 主干
        vision_backbone = self.medclip_model.vision_model.model

        # 手动 forward 到 layer4，获取 feature map
        x = vision_backbone.conv1(x)    #卷积层
        x = vision_backbone.bn1(x)  #批归一化
        x = vision_backbone.relu(x)     #激活函数
        x = vision_backbone.maxpool(x)  # 降维
        x = vision_backbone.layer1(x)
        x = vision_backbone.layer2(x)
        x = vision_backbone.layer3(x)
        feats = vision_backbone.layer4(x)  # 输出: [B, 2048, 7, 7]    (B, 2048, H/32, W/32)
        feats = self.cot_layer(feats)  # [B, 2048, 7, 7] 加上下文建模

        #防止模型结构不符或特征维度错误，提前报错排查问题。
        if feats.dim() != 4 or feats.shape[1] != 2048:
            raise ValueError(f"Unexpected feature shape: {feats.shape}")

        #将 [B, 2048, 7, 7] 转换为 [B, 49, 2048]：把每个 7×7 的空间位置作为一个 patch，展平后成为序列。
        B, C, H, W = feats.shape
        patch_feats = feats.view(B, C, H * W).permute(0, 2, 1)  # [B, 49, 2048]
        #对每张图像的特征图进行全局平均池化，得到整图的 2048 维向量（用于分类）。
        avg_feats = self.avg_pool(feats).view(B, -1)  # [B, 2048]
        #用分类头得到图像的多标签预测结果
        labels = self.classifier(avg_feats)  # [B, num_classes]
        return patch_feats, avg_feats, labels



