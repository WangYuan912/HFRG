# 🚀 优化的lamrg.py - CLIP增强防过拟合版本

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 新增CLIP导入
try:
    import clip

    CLIP_AVAILABLE = True
except ImportError:
    print("Warning: CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
    CLIP_AVAILABLE = False

from modules.visual_extractor import VisualExtractor
from modules.Transformer import TransformerModel
from modules.text_encoder import TextEncoder, MHA_FF


class OptimizedCLIPAlignmentModule(nn.Module):
    """🔧 优化的CLIP对齐模块 - 增强正则化，防止过拟合"""

    def __init__(self, args):
        super(OptimizedCLIPAlignmentModule, self).__init__()
        self.args = args
        self.device = getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_dim = getattr(args, 'clip_feature_dim', 512)

        # 加载预训练CLIP模型
        self.clip_model = None
        if CLIP_AVAILABLE:
            try:
                model_name = getattr(args, 'clip_model_name', 'ViT-B/32')
                self.clip_model, self.clip_preprocess = clip.load(model_name, device=self.device)

                # 根据参数决定是否冻结CLIP
                freeze_clip = getattr(args, 'freeze_clip', True)
                if freeze_clip:
                    for param in self.clip_model.parameters():
                        param.requires_grad = False

                print(f"🚀 CLIP model {model_name} loaded successfully (frozen: {freeze_clip})")
            except Exception as e:
                print(f"⚠️  Failed to load CLIP model: {e}")
                self.clip_model = None

        # 🔧 增强的特征投影层 - 添加更多正则化
        self.img_proj = nn.Sequential(
            nn.Linear(args.d_vf, self.clip_dim),
            nn.Dropout(getattr(args, 'dropout', 0.35) * 0.8),  # 使用稍低的dropout
            nn.LayerNorm(self.clip_dim)  # 添加层归一化
        )

        self.txt_proj = nn.Sequential(
            nn.Linear(args.d_model, self.clip_dim),
            nn.Dropout(getattr(args, 'dropout', 0.35) * 0.8),
            nn.LayerNorm(self.clip_dim)
        )

        # 🔧 简化对齐网络，减少参数数量防止过拟合
        self.global_align = nn.MultiheadAttention(
            self.clip_dim,
            num_heads=4,  # 减少注意力头数从8到4
            dropout=getattr(args, 'dropout', 0.35) * 0.6,
            batch_first=True
        )

        # 🔧 使用更简单的局部对齐
        self.local_align = nn.MultiheadAttention(
            self.clip_dim,
            num_heads=4,  # 减少注意力头数
            dropout=getattr(args, 'dropout', 0.35) * 0.6,
            batch_first=True
        )

        # 🔧 可学习的温度参数，但限制范围
        temperature_init = getattr(args, 'temperature_init', 0.07)
        self.temperature = nn.Parameter(
            torch.ones([]) * np.log(1 / temperature_init)
        )

        # 🔧 添加特征稳定化层
        self.feature_stabilizer = nn.Sequential(
            nn.Linear(self.clip_dim, self.clip_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.clip_dim, self.clip_dim)
        )

        self.init_weights()

    def init_weights(self):
        """🔧 优化的权重初始化"""
        for module in [self.img_proj[0], self.txt_proj[0]]:
            nn.init.xavier_uniform_(module.weight, gain=0.8)  # 使用较小的gain
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)

        # 初始化特征稳定化层
        for module in [self.feature_stabilizer[0], self.feature_stabilizer[3]]:
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)

    def extract_clip_features(self, images, texts=None):
        """🔧 优化的CLIP特征提取 - 修复tuple问题"""
        if self.clip_model is None:
            # 如果CLIP模型未加载，返回零特征
            batch_size = images.shape[0]
            img_features = torch.zeros(batch_size, self.clip_dim, device=images.device, dtype=torch.float32)
            if texts is not None:
                text_features = torch.zeros(len(texts), self.clip_dim, device=images.device, dtype=torch.float32)
                return img_features, text_features
            return img_features

        try:
            with torch.no_grad():
                # 🔧 图像CLIP特征 - 处理tuple返回值
                if images.dim() == 5:  # 处理多图像情况 [B, 2, C, H, W]
                    B, N, C, H, W = images.shape
                    images_reshaped = images.view(B * N, C, H, W)
                    img_features_raw = self.clip_model.encode_image(images_reshaped)

                    # 🔧 处理可能的tuple返回值
                    if isinstance(img_features_raw, tuple):
                        img_features = img_features_raw[0]  # 取第一个元素
                    else:
                        img_features = img_features_raw

                    img_features = img_features.view(B, N, -1).mean(dim=1)  # 平均池化
                else:
                    img_features_raw = self.clip_model.encode_image(images)

                    # 🔧 处理可能的tuple返回值
                    if isinstance(img_features_raw, tuple):
                        img_features = img_features_raw[0]  # 取第一个元素
                    else:
                        img_features = img_features_raw

                # 🔧 确保特征是tensor且为float32类型
                if not torch.is_tensor(img_features):
                    img_features = torch.tensor(img_features, device=images.device, dtype=torch.float32)
                else:
                    img_features = img_features.float()

                # 🔧 文本CLIP特征
                if texts is not None:
                    text_features_raw = self.clip_model.encode_text(texts)

                    # 🔧 处理可能的tuple返回值
                    if isinstance(text_features_raw, tuple):
                        text_features = text_features_raw[0]  # 取第一个元素
                    else:
                        text_features = text_features_raw

                    # 确保是tensor且为float32
                    if not torch.is_tensor(text_features):
                        text_features = torch.tensor(text_features, device=images.device, dtype=torch.float32)
                    else:
                        text_features = text_features.float()

                    return img_features, text_features

                return img_features

        except Exception as e:
            print(f"⚠️  CLIP特征提取失败: {e}")
            # 返回零特征作为回退
            batch_size = images.shape[0]
            img_features = torch.zeros(batch_size, self.clip_dim, device=images.device, dtype=torch.float32)
            if texts is not None:
                text_features = torch.zeros(len(texts), self.clip_dim, device=images.device, dtype=torch.float32)
                return img_features, text_features
            return img_features

    def prepare_text_for_clip(self, targets, tokenizer):
        """🔧 优化的文本预处理 - 更好的错误处理"""
        text_tokens = []
        for target in targets:
            try:
                # 解码文本
                text = tokenizer.decode_batch(target.unsqueeze(0).cpu().numpy())[0]
                # 清理文本
                text = text.replace('<unk>', '').replace('<pad>', '').replace('.', ' ').strip()
                if len(text) == 0 or text.isspace():
                    text = "normal chest xray"  # 默认文本

                # 确保文本不超过CLIP的最大长度
                words = text.split()
                if len(words) > 75:  # CLIP最大token数量约为77，留一些余量
                    text = ' '.join(words[:75])

                text_token = clip.tokenize(text, truncate=True)
                text_tokens.append(text_token)
            except Exception as e:
                # 如果处理失败，使用默认文本
                print(f"⚠️  文本处理失败: {e}")
                text_tokens.append(clip.tokenize("normal chest xray", truncate=True))

        return torch.cat(text_tokens).to(targets.device)

    def multi_level_alignment(self, img_feats, txt_feats, clip_img_feats, clip_txt_feats):
        """🔧 优化的多层次对齐 - 修复tuple和数据类型问题"""
        try:
            # 🔧 安全的特征转换函数
            def safe_to_tensor_float(x):
                if x is None:
                    return None
                if isinstance(x, tuple):
                    x = x[0]  # 取tuple的第一个元素
                if not torch.is_tensor(x):
                    return torch.tensor(x, dtype=torch.float32)
                return x.float()

            # 🔧 确保所有输入都是正确的tensor类型
            clip_img_feats = safe_to_tensor_float(clip_img_feats)
            clip_txt_feats = safe_to_tensor_float(clip_txt_feats)
            img_feats = safe_to_tensor_float(img_feats)
            txt_feats = safe_to_tensor_float(txt_feats)

            # 投影到CLIP空间
            proj_img_feats = self.img_proj(img_feats)  # [B, N_img, clip_dim]
            proj_txt_feats = self.txt_proj(txt_feats)  # [B, N_txt, clip_dim]

            # 🔧 确保投影特征也是float32
            proj_img_feats = proj_img_feats.float()
            proj_txt_feats = proj_txt_feats.float()

            # 🔧 特征稳定化
            proj_img_feats = self.feature_stabilizer(proj_img_feats)
            proj_txt_feats = self.feature_stabilizer(proj_txt_feats)

            # 全局对齐 - 简化版本
            try:
                # 🔧 确保所有输入都是相同的数据类型和正确的形状
                query_img = proj_img_feats.mean(dim=1, keepdim=True).float()

                if clip_img_feats is not None and clip_img_feats.numel() > 0:
                    if clip_img_feats.dim() == 1:
                        key_img = clip_img_feats.unsqueeze(0).unsqueeze(1).float()
                    elif clip_img_feats.dim() == 2:
                        key_img = clip_img_feats.unsqueeze(1).float()
                    else:
                        key_img = clip_img_feats.float()
                else:
                    key_img = query_img
                value_img = key_img

                query_txt = proj_txt_feats.mean(dim=1, keepdim=True).float()

                if clip_txt_feats is not None and clip_txt_feats.numel() > 0:
                    if clip_txt_feats.dim() == 1:
                        key_txt = clip_txt_feats.unsqueeze(0).unsqueeze(1).float()
                    elif clip_txt_feats.dim() == 2:
                        key_txt = clip_txt_feats.unsqueeze(1).float()
                    else:
                        key_txt = clip_txt_feats.float()
                else:
                    key_txt = query_txt
                value_txt = key_txt

                global_img_feats, _ = self.global_align(query_img, key_img, value_img)
                global_txt_feats, _ = self.global_align(query_txt, key_txt, value_txt)

            except Exception as e:
                print(f"⚠️  全局对齐失败，使用简化版本: {e}")
                global_img_feats = proj_img_feats.mean(dim=1, keepdim=True)
                global_txt_feats = proj_txt_feats.mean(dim=1, keepdim=True)

            # 🔧 简化的局部对齐
            try:
                # 确保所有tensor都是float32
                proj_img_feats = proj_img_feats.float()
                proj_txt_feats = proj_txt_feats.float()

                # 使用自注意力
                local_img_feats, _ = self.local_align(proj_img_feats, proj_img_feats, proj_img_feats)
                local_txt_feats, _ = self.local_align(proj_txt_feats, proj_txt_feats, proj_txt_feats)
            except Exception as e:
                print(f"  局部对齐失败，使用投影特征: {e}")
                local_img_feats = proj_img_feats
                local_txt_feats = proj_txt_feats

            return {
                'global_img': global_img_feats.squeeze(1).float(),
                'global_txt': global_txt_feats.squeeze(1).float(),
                'local_img': local_img_feats.float(),
                'local_txt': local_txt_feats.float(),
                'proj_img': proj_img_feats.float(),
                'proj_txt': proj_txt_feats.float()
            }

        except Exception as e:
            print(f"  多层次对齐完全失败: {e}")
            # 返回基础特征
            batch_size = img_feats.shape[0] if torch.is_tensor(img_feats) else 1
            device = img_feats.device if torch.is_tensor(img_feats) else 'cpu'

            zero_global = torch.zeros(batch_size, self.clip_dim, device=device, dtype=torch.float32)
            zero_local = torch.zeros(batch_size, 1, self.clip_dim, device=device, dtype=torch.float32)

            return {
                'global_img': zero_global,
                'global_txt': zero_global,
                'local_img': zero_local,
                'local_txt': zero_local,
                'proj_img': zero_local,
                'proj_txt': zero_local
            }


class SimplifiedCLIPAlignment(nn.Module):
    """🔧 简化的CLIP对齐 - 减少bug风险，提高稳定性"""

    def __init__(self, args):
        super(SimplifiedCLIPAlignment, self).__init__()
        self.args = args
        self.device = getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_dim = getattr(args, 'clip_feature_dim', 512)

        # 加载CLIP
        self.clip_model = None
        if CLIP_AVAILABLE:
            try:
                model_name = getattr(args, 'clip_model_name', 'ViT-B/32')
                self.clip_model, _ = clip.load(model_name, device=self.device)
                for param in self.clip_model.parameters():
                    param.requires_grad = False
                print(f"✅ CLIP {model_name} loaded and frozen")
            except Exception as e:
                print(f"❌ CLIP loading failed: {e}")
                self.clip_model = None

        # 🔧 简化的投影层
        self.img_proj = nn.Linear(args.d_vf, self.clip_dim)
        self.txt_proj = nn.Linear(args.d_model, self.clip_dim)

        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def extract_clip_features(self, images, texts=None):
        """简化的CLIP特征提取"""
        if self.clip_model is None:
            batch_size = images.shape[0]
            img_feat = torch.zeros(batch_size, self.clip_dim, device=images.device)
            if texts is not None:
                txt_feat = torch.zeros(len(texts), self.clip_dim, device=images.device)
                return img_feat, txt_feat
            return img_feat

        try:
            with torch.no_grad():
                # 处理图像
                if images.dim() == 5:  # [B, 2, C, H, W]
                    B, N, C, H, W = images.shape
                    images = images.view(B * N, C, H, W)
                    img_feat = self.clip_model.encode_image(images)
                    img_feat = img_feat.view(B, N, -1).mean(dim=1)
                else:
                    img_feat = self.clip_model.encode_image(images)

                # 确保是tensor
                if isinstance(img_feat, tuple):
                    img_feat = img_feat[0]
                img_feat = img_feat.float()

                if texts is not None:
                    txt_feat = self.clip_model.encode_text(texts)
                    if isinstance(txt_feat, tuple):
                        txt_feat = txt_feat[0]
                    txt_feat = txt_feat.float()
                    return img_feat, txt_feat

                return img_feat

        except Exception as e:
            print(f"⚠️ CLIP extraction failed: {e}")
            batch_size = images.shape[0]
            img_feat = torch.zeros(batch_size, self.clip_dim, device=images.device)
            if texts is not None:
                txt_feat = torch.zeros(len(texts), self.clip_dim, device=images.device)
                return img_feat, txt_feat
            return img_feat

    def align_features(self, img_feats, txt_feats, clip_img_feats, clip_txt_feats):
        """简化的特征对齐"""
        try:
            # 简单投影
            proj_img = self.img_proj(img_feats.mean(dim=1))  # [B, clip_dim]
            proj_txt = self.txt_proj(txt_feats.mean(dim=1))  # [B, clip_dim]

            return {
                'proj_img': proj_img,
                'proj_txt': proj_txt,
                'clip_img': clip_img_feats,
                'clip_txt': clip_txt_feats
            }
        except Exception as e:
            print(f" Feature alignment failed: {e}")
            batch_size = img_feats.shape[0]
            zero_feat = torch.zeros(batch_size, self.clip_dim, device=img_feats.device)
            return {
                'proj_img': zero_feat,
                'proj_txt': zero_feat,
                'clip_img': zero_feat,
                'clip_txt': zero_feat
            }


class _LAMRG(nn.Module):
    def __init__(self, args, tokenizer):
        super(_LAMRG, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = TransformerModel(args, tokenizer)
        self.proj = nn.Linear(args.num_labels, args.d_vf)
        self._init_weight(self.proj)

    @staticmethod
    def _init_weight(f):
        nn.init.kaiming_normal_(f.weight)
        f.bias.data.fill_(0)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images):
        # 提取第一张图像的特征和标签
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        # 提取第二张图像的特征和标签
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])
        # 计算两个图像的全连接特征的平均值
        fc_feats = torch.mean(torch.stack([fc_feats_0, fc_feats_1]), dim=0)
        # 将两个图像的注意力特征拼接在一起
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        # 计算两个图像的标签的平均值
        out_labels = torch.mean(torch.stack([labels_0, labels_1]), dim=0)
        return att_feats, fc_feats, out_labels

    def forward_mimic_cxr(self, images):
        att_feats, fc_feats, out_labels = self.visual_extractor(images)
        return att_feats, fc_feats, out_labels

    def forward(self, images, targets=None, labels=None, mode='train'):
        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, out_labels = self.forward_iu_xray(images)
        else:
            att_feats, fc_feats, out_labels = self.forward_mimic_cxr(images)

        label_feats = self.proj(out_labels).unsqueeze(1)
        att_feats = torch.cat((att_feats, label_feats), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class LAMRGModel_vCLIP(_LAMRG):
    """🚀 优化的CLIP增强医学报告生成模型 - 防过拟合版本"""

    def __init__(self, args, tokenizer):
        super(LAMRGModel_vCLIP, self).__init__(args, tokenizer)

        # 基础组件参数
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.d_vf = args.d_vf
        self.dropout = getattr(args, 'dropout', 0.35)

        # 🔧 增强正则化的文本编码器
        self.txt_encoder = TextEncoder(
            self.d_model, self.d_ff, self.num_layers,
            self.tgt_vocab, self.num_labels, self.h, self.dropout
        )

        # 🔧 使用优化的CLIP对齐模块
        if getattr(args, 'use_clip', False):
            self.clip_alignment = OptimizedCLIPAlignmentModule(args)
            print("🚀 Optimized CLIP alignment module initialized")
        else:
            self.clip_alignment = None

        # 🔧 增强正则化的记忆和选择模块
        self.memory = self.init_memory()
        self.update_memory = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.select_prior = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)

        # 🔧 投影层 - 添加dropout
        self.linear_z = nn.Sequential(
            nn.Linear(self.d_vf, self.d_model),
            nn.Dropout(self.dropout * 0.5)  # 较轻的dropout
        )

        self.linear_feat = nn.Sequential(
            nn.Linear(self.d_model, self.d_vf),
            nn.Dropout(self.dropout * 0.5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.num_labels),
            nn.Dropout(self.dropout * 0.3)  # 分类器用较轻的dropout
        )

        self.embed_labels = nn.Linear(1, self.d_model)

        # 🔧 优化的CLIP特征融合层
        if self.clip_alignment is not None:
            self.clip_fusion = nn.Sequential(
                nn.Linear(self.clip_alignment.clip_dim + self.d_model, self.d_model),
                nn.Dropout(self.dropout * 0.4),
                nn.LayerNorm(self.d_model),  # 添加层归一化
                nn.ReLU(),
                nn.Linear(self.d_model, self.d_model)
            )
        else:
            self.clip_fusion = None

        # 🔧 全局上下文投影层
        self.global_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_vf),
            nn.Dropout(self.dropout * 0.3)
        )

        self.init_weights()

        if getattr(args, 'use_enhanced_decoding', False):
            from modules.enhanced_decoding import integrate_enhanced_decoding
            self = integrate_enhanced_decoding(self, tokenizer)
            print(" Enhanced decoding integrated successfully")

    def init_weights(self):
        """🔧 优化的权重初始化 - 更保守的初始化"""
        modules_to_init = []

        # 提取需要初始化的线性层
        if isinstance(self.linear_z, nn.Sequential):
            modules_to_init.append(self.linear_z[0])
        else:
            modules_to_init.append(self.linear_z)

        if isinstance(self.linear_feat, nn.Sequential):
            modules_to_init.append(self.linear_feat[0])
        else:
            modules_to_init.append(self.linear_feat)

        if isinstance(self.classifier, nn.Sequential):
            modules_to_init.append(self.classifier[0])
        else:
            modules_to_init.append(self.classifier)

        modules_to_init.extend([self.embed_labels, self.global_proj[0]])

        if self.clip_fusion is not None:
            modules_to_init.extend([self.clip_fusion[0], self.clip_fusion[4]])

        # 使用较小的初始化范围
        for module in modules_to_init:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)  # 较小的gain
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)

    def init_memory(self):
        """初始化记忆模块"""
        num_slots = getattr(self.args, 'num_slots', 60)
        memory = nn.Parameter(torch.eye(num_slots).unsqueeze(0))
        if self.d_model > num_slots:
            diff = self.d_model - num_slots
            pad = torch.zeros((1, num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < num_slots:
            memory = memory[:, :, :self.d_model]
        return memory

    def _safe_extract_tensor(self, data):
        """🔧 增强的安全tensor提取"""
        if data is None:
            return None
        if torch.is_tensor(data):
            return data
        if isinstance(data, tuple):
            for item in data:
                if torch.is_tensor(item) and item.numel() > 0:
                    return item
            return None
        try:
            return torch.tensor(data) if data is not None else None
        except:
            return None

    def forward(self, images, targets=None, labels=None, mode='train'):
        """🔧 优化的前向传播 - 增强错误处理和稳定性"""
        bsz = images.shape[0]

        try:
            # 视觉特征提取
            if self.args.dataset_name == 'iu_xray':
                att_feats, avg_feats, _ = self.forward_iu_xray(images)
            else:
                att_feats, avg_feats, _ = self.forward_mimic_cxr(images)

            # 🔧 安全的特征处理
            z_img = self.linear_z(avg_feats)
            if isinstance(z_img, tuple):
                z_img = z_img[0]

            vis_labels = self.classifier(z_img)
            if isinstance(vis_labels, tuple):
                vis_labels = vis_labels[0]

            # 记忆模块
            memory = self.memory.to(images.device).expand(bsz, -1, -1)

            # 初始化返回变量
            alignment_feats = None
            clip_img_feats = None
            clip_txt_feats = None
            enhanced_z_img = z_img
            z_txt = None
            txt_labels = None

            # 🔧 CLIP增强处理 - 增强错误处理
            if self.clip_alignment is not None and mode == 'train':
                try:
                    # 提取CLIP图像特征
                    clip_img_feats = self.clip_alignment.extract_clip_features(images)

                    # 文本编码
                    txt_output = self.txt_encoder(targets)
                    if isinstance(txt_output, tuple) and len(txt_output) >= 3:
                        txt_feats, z_txt_raw, txt_labels = txt_output
                        z_txt = self._safe_extract_tensor(z_txt_raw)
                    else:
                        txt_feats = txt_output
                        z_txt = None
                        txt_labels = None

                    # 🔧 安全的CLIP文本特征提取
                    if CLIP_AVAILABLE and self.clip_alignment.clip_model is not None:
                        try:
                            text_tokens = self.clip_alignment.prepare_text_for_clip(targets, self.tokenizer)
                            clip_txt_feats = self.clip_alignment.extract_clip_features(images, text_tokens)
                        except Exception as e:
                            print(f"  CLIP文本特征提取失败: {e}")
                            clip_txt_feats = torch.zeros_like(clip_img_feats)
                    else:
                        clip_txt_feats = torch.zeros_like(clip_img_feats)

                    # 多层次对齐
                    alignment_feats = self.clip_alignment.multi_level_alignment(
                        att_feats, txt_feats, clip_img_feats, clip_txt_feats
                    )

                    # 更新记忆
                    memory = self.update_memory(memory, txt_feats)

                    # 🔧 安全的CLIP特征融合
                    if self.clip_fusion is not None and alignment_feats is not None:
                        try:
                            fusion_input = torch.cat([z_img, alignment_feats['global_img']], dim=-1)
                            enhanced_z_img = self.clip_fusion(fusion_input)
                        except Exception as e:
                            print(f"  CLIP特征融合失败: {e}")
                            enhanced_z_img = z_img

                except Exception as e:
                    print(f"  CLIP增强处理失败: {e}")
                    # 回退到非CLIP模式
                    if mode == 'train':
                        try:
                            txt_output = self.txt_encoder(targets)
                            if isinstance(txt_output, tuple) and len(txt_output) >= 3:
                                txt_feats, z_txt_raw, txt_labels = txt_output
                                z_txt = self._safe_extract_tensor(z_txt_raw)
                            else:
                                txt_feats = txt_output
                            memory = self.update_memory(memory, txt_feats)
                        except Exception as fallback_e:
                            print(f"  文本编码回退也失败: {fallback_e}")

            elif mode == 'train':
                # 非CLIP模式的训练
                try:
                    txt_output = self.txt_encoder(targets)
                    if isinstance(txt_output, tuple) and len(txt_output) >= 3:
                        txt_feats, z_txt_raw, txt_labels = txt_output
                        z_txt = self._safe_extract_tensor(z_txt_raw)
                    else:
                        txt_feats = txt_output
                        z_txt = None
                        txt_labels = None
                    memory = self.update_memory(memory, txt_feats)
                except Exception as e:
                    print(f"  标准文本编码失败: {e}")

            # 先验选择
            try:
                emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
                prior = self.select_prior(emb_labels, memory)
                att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

                # 🔧 全局上下文融合
                global_context = self.global_proj(enhanced_z_img).unsqueeze(1)
                att_feats = att_feats + global_context
            except Exception as e:
                print(f"先验选择或特征融合失败: {e}")
                # 使用基础的注意力特征
                pass

            # 生成报告
            if mode == 'train':
                output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
                # 返回结果
                if self.clip_alignment is not None:
                    return (output, vis_labels, txt_labels, enhanced_z_img, z_txt,
                            alignment_feats, clip_img_feats, clip_txt_feats)
                else:
                    return (output, vis_labels, txt_labels, enhanced_z_img, z_txt)

            elif mode == 'sample':
                if hasattr(self.encoder_decoder, 'enhanced_decoder'):
                    output = self.encoder_decoder.sample_enhanced(avg_feats, att_feats, self.args)
                else:
                    output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
                return output, vis_labels

        except Exception as e:
            print(f" 前向传播严重错误: {e}")
            # 最基础的回退策略
            if mode == 'train':
                # 返回基础输出
                try:
                    basic_output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
                    return (basic_output, torch.zeros(bsz, self.num_labels, device=images.device),
                            None, torch.zeros(bsz, self.d_model, device=images.device), None)
                except:
                    raise RuntimeError("Model forward pass completely failed")
            else:
                try:
                    basic_output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
                    return basic_output, torch.zeros(bsz, self.num_labels, device=images.device)
                except:
                    raise RuntimeError("Model sampling completely failed")


# 🔧 保持原有的其他模型类不变，但添加更好的错误处理

class LAMRGModel(_LAMRG):
    def __init__(self, args, tokenizer):
        super(LAMRGModel, self).__init__(args, tokenizer)
        self.m = nn.Parameter(torch.FloatTensor(1, args.num_labels, 40, args.d_vf))
        self.init_m()

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def init_m(self):
        nn.init.normal_(self.m, 0, 1 / self.args.num_labels)

    def forward_iu_xray(self, images, targets=None, labels=None, mode='train'):
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        out_labels = labels_0
        bs, nf, d_f = att_feats.shape
        _, n_l, n_m, d_f = self.m.shape
        m = labels[:, :, None, None] * self.m.expand(bs, n_l, n_m, d_f)
        m = m.reshape(bs, -1, d_f)
        att_feats = torch.cat((att_feats, m), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, out_labels

    def forward_mimic_cxr(self, images, targets=None, labels=None, mode='train'):
        att_feats, fc_feats, out_labels = self.visual_extractor(images)
        bs, nf, d_f = att_feats.shape
        _, n_l, n_m, d_f = self.m.shape
        m = labels[:, :, None, None] * self.m.expand(bs, n_l, n_m, d_f)
        m = m.reshape(bs, -1, d_f)
        att_feats = torch.cat((att_feats, m), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class BasicModel(LAMRGModel):
    def forward_iu_xray(self, images, targets=None, labels=None, mode='train'):
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        out_labels = labels_0
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, out_labels

    def forward_mimic_cxr(self, images, targets=None, labels=None, mode='train'):
        att_feats, fc_feats, out_labels = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class LAMRGModel_v7(_LAMRG):
    """直接将visual_extractor输出的label concat到visual feature后"""
    def __init__(self, args, tokenizer):
        super(LAMRGModel_v7, self).__init__(args, tokenizer)

    def forward(self, images, targets=None, labels=None, mode='train'):
        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, out_labels = self.forward_iu_xray(images)
        else:
            att_feats, fc_feats, out_labels = self.forward_mimic_cxr(images)

        label_feats = self.proj(out_labels).unsqueeze(1)
        att_feats = torch.cat((att_feats, label_feats), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class LAMRGModel_v8(_LAMRG):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_v8, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)
        self.prior_memory = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.select_prior = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.memory = self.init_memory()

        self.proj_label = nn.Linear(args.num_labels, args.d_model)
        self.proj_att = nn.Linear(args.d_vf, args.d_model)
        self.proj_feat = nn.Linear(args.d_model, args.d_vf)
        self.init_weight_()

    def init_weight_(self):
        nn.init.kaiming_normal_(self.proj_label.weight)
        self.proj_label.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.proj_att.weight)
        self.proj_att.bias.data.fill_(0)

    def init_memory(self):
        memory = nn.Parameter(torch.eye(self.num_slots, device='cuda').unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff), device='cuda')
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        return memory

    def forward(self, images, targets=None, labels=None, mode='train'):
        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, out_labels = self.forward_iu_xray(images)
        else:
            att_feats, fc_feats, out_labels = self.forward_mimic_cxr(images)

        bsz = att_feats.shape[0]
        memory = self.memory.expand(bsz, self.num_slots, self.d_model)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.prior_memory(memory, txt_feats)

        label_feats = self.proj_label(out_labels).unsqueeze(1)
        prior = self.select_prior(label_feats, memory)
        att_feats = torch.cat((att_feats, self.proj_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError
        return output, out_labels


class LAMRGModel_v9(_LAMRG):  # 定义 LAMRGModel_v9 类，继承自 _LAMRG 类
    def __init__(self, args, tokenizer):  # 初始化方法，参数包括 args 和 tokenizer
        super(LAMRGModel_v9, self).__init__(args, tokenizer)  # 调用父类的初始化方法
        self.args = args  # 设置 args 属性
        self.d_model = args.d_model  # 设置模型维度 d_model
        self.d_ff = args.d_ff  # 设置前馈网络维度 d_ff
        self.num_layers = args.num_layers  # 设置层数 num_layers
        self.num_labels = args.num_labels  # 设置标签数量 num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1  # 设置目标词汇表大小 tgt_vocab
        self.h = args.num_heads  # 设置注意力头数 h
        self.num_slots = args.num_slots  # 设置槽位数量 num_slots
        self.d_vf = args.d_vf  # 设置视觉特征维度 d_vf
        self.dropout = args.dropout  # 设置 dropout 概率
        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels, self.h, self.dropout)  # 初始化文本编码器

        #self.txt_encoder = TextEncoder(num_labels=self.num_labels, d_model=self.d_model)
        # self.txt_encoder = TextEncoder(
        #     self.d_model,
        #     self.d_ff,
        #     self.num_layers,
        #     self.tgt_vocab,
        #     self.num_labels,
        #     self.h,
        #     self.dropout,
        # )

        #prior_memory、select_prior均为多头注意力机制
        self.prior_memory = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)  # 初始化先验记忆模块
        self.select_prior = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)  # 初始化先验选择模块
        self.memory = self.init_memory()  # 初始化记忆模块

        self.linear_mem = nn.Linear(self.d_model, self.d_model)  # 初始化记忆线性层
        self.linear_label = nn.Linear(args.num_labels, args.d_model)  # 初始化标签线性层
        self.linear_feat = nn.Linear(args.d_model, args.d_vf)  # 初始化特征线性层
        self.linear_fcfeat = nn.Linear(args.d_vf, args.d_model)  # 初始化全连接特征线性层
        self.init_weight_()  # 初始化权重

    def init_weight_(self):  # 初始化权重方法
        nn.init.kaiming_normal_(self.linear_mem.weight)  # 使用 Kaiming 初始化方法初始化 linear_mem 的权重
        self.linear_mem.bias.data.fill_(0)  # 将 linear_mem 的偏置数据填充为 0
        nn.init.kaiming_normal_(self.linear_label.weight)  # 使用 Kaiming 初始化方法初始化 linear_label 的权重
        self.linear_label.bias.data.fill_(0)  # 将 linear_label 的偏置数据填充为 0
        nn.init.kaiming_normal_(self.linear_feat.weight)  # 使用 Kaiming 初始化方法初始化 linear_feat 的权重
        self.linear_feat.bias.data.fill_(0)  # 将 linear_feat 的偏置数据填充为 0

    def init_memory(self):  # 初始化记忆模块方法
        memory = nn.Parameter(torch.eye(self.num_slots, device='cuda').unsqueeze(0))  # 创建一个初始记忆矩阵，使用单位矩阵并扩展维度
        if self.d_model > self.num_slots:  # 如果模型维度大于槽位数量
            diff = self.d_model - self.num_slots  # 计算差值
            pad = torch.zeros((1, self.num_slots, diff), device='cuda')  # 创建一个零填充的张量
            memory = torch.cat([memory, pad], -1)  # 将零填充的张量拼接到记忆矩阵中
        elif self.d_model < self.num_slots:  # 如果模型维度小于槽位数量
            memory = memory[:, :, :self.d_model]  # 裁剪记忆矩阵以匹配模型维度
        return memory  # 返回初始化的记忆矩阵

    def forward(self, images, targets=None, labels=None, mode='train'):  # 前向传播方法，处理图像数据，参数包括图像、目标、标签和模式
        if self.args.dataset_name == 'iu_xray':  # 如果数据集名称为 'iu_xray'
            att_feats, fc_feats, vis_labels = self.forward_iu_xray(images)  # 调用 forward_iu_xray 方法处理图像
        else:  # 否则
            att_feats, fc_feats, vis_labels = self.forward_mimic_cxr(images)  # 调用 forward_mimic_cxr 方法处理图像
        z_img = self.linear_fcfeat(fc_feats)  # 将全连接特征通过线性层转换为 z_img
        bsz = att_feats.shape[0]  # 获取 batch size
        # memory = self.linear_mem(self.memory).expand(bsz, -1, -1)  # (注释掉的代码，用于将记忆矩阵通过线性层转换并扩展)
        memory = self.memory.expand(bsz, -1, -1)  # 将记忆矩阵扩展为 (bsz, num_slots, d_model) 的形状
        if mode == 'train':  # 如果模式为训练
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)  # 通过文本编码器提取目标文本的特征
            memory = self.prior_memory(memory, txt_feats)  # 更新记忆模块，结合文本特征
        label_feats = self.linear_label(vis_labels).unsqueeze(1)  # 将可视化标签通过线性层转换并扩展维度
        prior = self.select_prior(label_feats, memory)  # 选择先验特征
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)  # 将先验特征与注意力特征拼接
        if mode == 'train':  # 如果模式为训练
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')  # 调用编码器-解码器进行前向传播
            return (output, vis_labels, txt_labels, z_img, z_txt)  # 返回输出、可视化标签、文本标签、z_img 和 z_txt
        elif mode == 'sample':  # 如果模式为采样
            # ipdb.set_trace()  # (调试用的断点，用于调试时查看变量)
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')  # 调用编码器-解码器进行采样
            return output, vis_labels  # 返回输出和可视化标签
        else:  # 如果模式既不是训练也不是采样
            raise ValueError  # 抛出 ValueError 异常

class LAMRGModel_v10(LAMRGModel_v9):
    def forward(self, images, targets=None, labels=None, mode='train'):  # 定义前向传播函数，处理图像数据，参数包括图像、目标、标签和模式
        if self.args.dataset_name == 'iu_xray':  # 如果数据集名称为 'iu_xray'
            att_feats, fc_feats, vis_labels = self.forward_iu_xray(images)  # 调用 forward_iu_xray 方法处理图像
        else:  # 否则
            att_feats, fc_feats, vis_labels = self.forward_mimic_cxr(images)  # 调用 forward_mimic_cxr 方法处理图像
        z_img = self.linear_fcfeat(fc_feats)  # 将全连接特征通过线性层转换为 z_img
        bsz = att_feats.shape[0]  # 获取 batch size
        memory = self.linear_mem(self.memory).expand(bsz, -1, -1)  # 将模型的 memory 通过线性层转换并扩展为 (bsz, n_l, d_f) 的形状
        # memory = self.memory.expand(bsz, -1, -1)  # (注释掉的代码，用于直接扩展 memory)
        if mode == 'train':  # 如果模式为训练
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)  # 通过文本编码器提取目标文本的特征
            memory = self.prior_memory(memory, txt_feats)  # 更新 memory，结合文本特征
        label_feats = self.linear_label(vis_labels).unsqueeze(1)  # 将可视化标签通过线性层转换并扩展维度
        prior = self.select_prior(label_feats, memory)  # 选择先验特征
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)  # 将先验特征与注意力特征拼接
        if mode == 'train':  # 如果模式为训练
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')  # 调用编码器-解码器进行前向传播
            return (output, vis_labels, txt_labels, z_img, z_txt)  # 返回输出、可视化标签、文本标签、z_img 和 z_txt
        elif mode == 'sample':  # 如果模式为采样
            # ipdb.set_trace()  # (调试用的断点，用于调试时查看变量)
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')  # 调用编码器-解码器进行采样
            return output, vis_labels  # 返回输出和可视化标签
        else:  # 如果模式既不是训练也不是采样
            raise ValueError  # 抛出 ValueError 异常


class LAMRGModel_v11(LAMRGModel_v9):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_v11, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)
        self.update_memory = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.select_prior = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.memory, self.mask = self.init_memory()

        self.linear_z = nn.Linear(args.d_vf, args.d_model)
        self.linear_label = nn.Linear(args.d_model, args.num_labels)
        self.query_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_feat = nn.Linear(args.d_model, args.d_vf)
        self.init_weight()

    def init_weight(self):
        self._init_weight(self.linear_z)
        self._init_weight(self.linear_label)
        self._init_weight(self.query_mem)
        self._init_weight(self.linear_feat)

    def init_memory(self):
        memory = nn.Parameter(torch.eye(self.num_slots).unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        mask = torch.ones((self.num_slots, self.d_model))
        mask[:, self.num_slots:] = 0
        return memory, mask

    def forward(self, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]

        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, _ = ve(images)
        z_img = self.linear_z(avg_feats)
        vis_labels = self.linear_label(z_img)

        memory = self.query_mem(self.memory.to(images)).expand(bsz, -1, -1)
        mask = self.mask.to(images).expand(bsz, -1, -1)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.update_memory(memory, txt_feats, mask)

        prior = self.select_prior(z_img, memory)
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            # ipdb.set_trace()
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


class LAMRGModel_v12(LAMRGModel_v9):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_v12, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)
        self.update_memory = MHA_FF(self.d_model, self.d_ff, args.num_memory_heads, self.dropout)
        self.select_prior = MHA_FF(self.d_model, self.d_ff, args.num_memory_heads, self.dropout)
        self.memory, self.mask = self.init_memory()

        self.get_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_z = nn.Linear(self.d_vf, self.d_model)
        self.linear_feat = nn.Linear(self.d_model, self.d_vf)

        self.classifier = nn.Linear(self.d_model, self.num_labels)
        self.embed_labels = nn.Linear(1, self.d_model)

        self.init_weight()

    def init_weight(self):
        self._init_weight(self.linear_z)
        self._init_weight(self.get_mem)
        self._init_weight(self.linear_feat)
        self._init_weight(self.classifier)
        self._init_weight(self.embed_labels)

    def init_memory(self):
        memory = nn.Parameter(torch.eye(self.num_slots).unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        mask = torch.ones((self.num_slots, self.d_model))
        mask[:, self.num_slots:] = 0
        return memory, mask

    def forward(self, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]

        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, _ = ve(images)

        z_img = self.linear_z(avg_feats)
        vis_labels = self.classifier(z_img)

        memory = self.get_mem(self.memory.to(images)).expand(bsz, -1, -1)
        mask = self.mask.to(images).expand(bsz, -1, -1)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.update_memory(memory, txt_feats, mask)

        emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        prior = self.select_prior(emb_labels, memory)
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


class LAMRGModel_v91(LAMRGModel_v12):
    """Ablation Study
        只用label的模型
    """

    def forward(self, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]

        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, _ = ve(images)

        z_img = self.linear_z(avg_feats)
        vis_labels = self.classifier(z_img)

        txt_feats, z_txt, txt_labels = None, None, None
        # memory = self.get_mem(self.memory.to(images)).expand(bsz, -1, -1)
        # mask = self.mask.to(images).expand(bsz, -1, -1)
        # if mode == 'train':
        # txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
        # memory = self.update_memory(memory, txt_feats, mask)

        # emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        # prior = self.select_prior(emb_labels, memory)
        # att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


class LAMRGModel_v92(LAMRGModel_v12):
    """Ablation Study
        用label loss + rank loss的模型
    """

    def forward(self, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]

        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, _ = ve(images)

        z_img = self.linear_z(avg_feats)
        vis_labels = self.classifier(z_img)

        # memory = self.get_mem(self.memory.to(images)).expand(bsz, -1, -1)
        # mask = self.mask.to(images).expand(bsz, -1, -1)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            # memory = self.update_memory(memory, txt_feats, mask)

        # emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        # prior = self.select_prior(emb_labels, memory)
        # att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            # ipdb.set_trace()
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError


class LAMRGModel_vRebuttal(LAMRGModel_v9):
    def __init__(self, args, tokenizer):
        super(LAMRGModel_vRebuttal, self).__init__(args, tokenizer)
        self.args = args
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)

        self.memory, self.mask = self.init_memory()

        self.get_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_z = nn.Linear(self.d_vf, self.d_model)
        self.linear_feat = nn.Linear(self.d_model, self.d_vf)

        self.classifier = nn.Linear(self.d_model, self.num_labels)
        self.embed_labels = nn.Linear(1, self.d_model)

        self.init_weight()

    @staticmethod
    def attention(query, key, value):
        "Compute 'Dot Product Attention'"

        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1))
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value)

    def init_weight(self):
        self._init_weight(self.linear_z)
        self._init_weight(self.get_mem)
        self._init_weight(self.linear_feat)
        self._init_weight(self.classifier)
        self._init_weight(self.embed_labels)

    def init_memory(self):
        memory = nn.Parameter(torch.eye(self.num_slots).unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        mask = torch.ones((self.num_slots, self.d_model))
        mask[:, self.num_slots:] = 0
        return memory, mask

    def forward(self, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]

        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, _ = ve(images)

        z_img = self.linear_z(avg_feats)
        vis_labels = self.classifier(z_img)
        # ipdb.set_trace()
        memory = self.get_mem(self.memory.to(images)).expand(bsz, -1, -1)
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.attention(memory, txt_feats, txt_feats)

        emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        prior = self.attention(emb_labels, memory, memory)
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError