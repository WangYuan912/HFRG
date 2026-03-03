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


class CLIPAlignmentModule(nn.Module):
    """CLIP对齐模块 - 核心创新组件"""

    def __init__(self, args):
        super(CLIPAlignmentModule, self).__init__()
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

                print(f"CLIP model {model_name} loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load CLIP model: {e}")
                self.clip_model = None

        # 特征投影层
        self.img_proj = nn.Linear(args.d_vf, self.clip_dim)
        self.txt_proj = nn.Linear(args.d_model, self.clip_dim)

        # 多层次对齐网络
        self.global_align = nn.MultiheadAttention(self.clip_dim, num_heads=8, batch_first=True)
        self.local_align = nn.MultiheadAttention(self.clip_dim, num_heads=8, batch_first=True)

        # 可学习的温度参数
        temperature_init = getattr(args, 'temperature_init', 0.07)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature_init))

        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.img_proj.weight)
        nn.init.xavier_uniform_(self.txt_proj.weight)
        if hasattr(self.img_proj, 'bias') and self.img_proj.bias is not None:
            self.img_proj.bias.data.fill_(0)
        if hasattr(self.txt_proj, 'bias') and self.txt_proj.bias is not None:
            self.txt_proj.bias.data.fill_(0)

    def extract_clip_features(self, images, texts=None):
        """提取CLIP特征"""
        if self.clip_model is None:
            # 如果CLIP模型未加载，返回零特征
            batch_size = images.shape[0]
            img_features = torch.zeros(batch_size, self.clip_dim, device=images.device)
            if texts is not None:
                text_features = torch.zeros(len(texts), self.clip_dim, device=images.device)
                return img_features, text_features
            return img_features

        with torch.no_grad():
            # 图像CLIP特征
            if images.dim() == 5:  # 处理多图像情况 [B, 2, C, H, W]
                B, N, C, H, W = images.shape
                images_reshaped = images.view(B * N, C, H, W)
                img_features = self.clip_model.encode_image(images_reshaped)
                img_features = img_features.view(B, N, -1).mean(dim=1)  # 平均池化
            else:
                img_features = self.clip_model.encode_image(images)

            # 文本CLIP特征
            if texts is not None:
                text_features = self.clip_model.encode_text(texts)
                return img_features, text_features

            return img_features

    def prepare_text_for_clip(self, targets, tokenizer):
        """将targets转换为CLIP可用的文本"""
        text_tokens = []
        for target in targets:
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

            try:
                text_token = clip.tokenize(text, truncate=True)
                text_tokens.append(text_token)
            except:
                # 如果tokenize失败，使用默认文本
                text_tokens.append(clip.tokenize("normal chest xray", truncate=True))

        return torch.cat(text_tokens).to(targets.device)

    def multi_level_alignment(self, img_feats, txt_feats, clip_img_feats, clip_txt_feats):
        """多层次对齐"""
        # 投影到CLIP空间
        proj_img_feats = self.img_proj(img_feats)  # [B, N_img, clip_dim]
        proj_txt_feats = self.txt_proj(txt_feats)  # [B, N_txt, clip_dim]

        # 全局对齐
        try:
            global_img_feats, _ = self.global_align(
                proj_img_feats.mean(dim=1, keepdim=True),   #Q
                clip_img_feats.unsqueeze(1),    #K
                clip_img_feats.unsqueeze(1)     #V
            )

            global_txt_feats, _ = self.global_align(
                proj_txt_feats.mean(dim=1, keepdim=True),   #Q
                clip_txt_feats.unsqueeze(1),    #K
                clip_txt_feats.unsqueeze(1)     #V
            )
        except:
            # 如果attention失败，使用简单的线性变换
            global_img_feats = proj_img_feats.mean(dim=1, keepdim=True)
            global_txt_feats = proj_txt_feats.mean(dim=1, keepdim=True)

        # 局部对齐
        try:
            local_img_feats, _ = self.local_align(proj_img_feats, proj_txt_feats, proj_txt_feats)
            local_txt_feats, _ = self.local_align(proj_txt_feats, proj_img_feats, proj_img_feats)
        except:
            # 如果attention失败，直接使用投影特征
            local_img_feats = proj_img_feats
            local_txt_feats = proj_txt_feats

        return {
            'global_img': global_img_feats.squeeze(1),
            'global_txt': global_txt_feats.squeeze(1),
            'local_img': local_img_feats,
            'local_txt': local_txt_feats,
            'proj_img': proj_img_feats,
            'proj_txt': proj_txt_feats
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

    # 定义前向传播函数，处理IU X-ray图像
    # 参数: images: 输入的图像张量，形状为 (batch_size, 2, channels, height, width)
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
        # 返回拼接后的注意力特征、平均后的全连接特征和平均后的标签
        return att_feats, fc_feats, out_labels


    def forward_mimic_cxr(self, images):
        att_feats, fc_feats, out_labels = self.visual_extractor(images)
        return att_feats, fc_feats, out_labels

    # 定义前向传播函数
    # 参数: images: 输入图像，targets: 目标序列，labels: 标签，mode: 模式（训练或采样）
    def forward(self, images, targets=None, labels=None, mode='train'):
        # 如果数据集名称为 'iu_xray'
        if self.args.dataset_name == 'iu_xray':
            # 调用 forward_iu_xray 方法处理图像
            att_feats, fc_feats, out_labels = self.forward_iu_xray(images)
        else:
            # 调用 forward_mimic_cxr 方法处理图像
            att_feats, fc_feats, out_labels = self.forward_mimic_cxr(images)
        # 将输出标签投影到特征空间并扩展维度
        label_feats = self.proj(out_labels).unsqueeze(1)
        # 将标签特征与注意力特征拼接
        att_feats = torch.cat((att_feats, label_feats), dim=1)
        # 如果模式为 'train'
        if mode == 'train':
            # 调用编码器-解码器模型进行前向传播
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        # 如果模式为 'sample'
        elif mode == 'sample':
            # 调用编码器-解码器模型进行采样
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            # 抛出 ValueError 异常
            raise ValueError
        # 返回输出和输出标签
        return output, out_labels


class LAMRGModel_vCLIP(_LAMRG):
    """CLIP增强的医学报告生成模型"""

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
        self.dropout = args.dropout

        # 文本编码器
        self.txt_encoder = TextEncoder(
            self.d_model, self.d_ff, self.num_layers,
            self.tgt_vocab, self.num_labels, self.h, self.dropout
        )

        # CLIP对齐模块（核心创新）
        if getattr(args, 'use_clip', False):
            self.clip_alignment = CLIPAlignmentModule(args)
        else:
            self.clip_alignment = None

        # 记忆和选择模块
        self.memory = self.init_memory()
        self.update_memory = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)
        self.select_prior = MHA_FF(self.d_model, self.d_ff, self.h, self.dropout)

        # 投影层
        self.linear_z = nn.Linear(self.d_vf, self.d_model)
        self.linear_feat = nn.Linear(self.d_model, self.d_vf)
        self.classifier = nn.Linear(self.d_model, self.num_labels)
        self.embed_labels = nn.Linear(1, self.d_model)

        # CLIP特征融合层
        if self.clip_alignment is not None:
            self.clip_fusion = nn.Linear(self.clip_alignment.clip_dim + self.d_model, self.d_model)
        else:
            self.clip_fusion = None

        self.init_weights()

    def init_weights(self):
        """权重初始化"""
        modules_to_init = [self.linear_z, self.linear_feat, self.classifier, self.embed_labels]
        if self.clip_fusion is not None:
            modules_to_init.append(self.clip_fusion)
        for module in modules_to_init:
            nn.init.xavier_uniform_(module.weight)
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
        """安全地从tuple或其他结构中提取tensor"""
        if data is None:
            return None
        if torch.is_tensor(data):
            return data
        if isinstance(data, tuple):
            # 从tuple中找到有效的tensor
            for item in data:
                if torch.is_tensor(item) and item.numel() > 0:
                    return item
            return None
        return data

    def forward(self, images, targets=None, labels=None, mode='train'):
        """主要前向传播函数"""
        bsz = images.shape[0]

        # 路径一，iu_xray、mimic_cxr视觉特征提取
        if self.args.dataset_name == 'iu_xray':
            att_feats, avg_feats, _ = self.forward_iu_xray(images)
        else:
            att_feats, avg_feats, _ = self.forward_mimic_cxr(images)

        # 图像特征处理
        z_img = self.linear_z(avg_feats)
        vis_labels = self.classifier(z_img)

        # 记忆模块
        memory = self.memory.to(images.device).expand(bsz, -1, -1)

        # CLIP增强处理
        alignment_feats = None
        clip_img_feats = None
        clip_txt_feats = None
        enhanced_z_img = z_img
        z_txt = None  # 初始化为None
        txt_labels = None  # 初始化为None

        if self.clip_alignment is not None and mode == 'train':
            # 路径二，提取CLIP图像特征
            clip_img_feats = self.clip_alignment.extract_clip_features(images)
            # 文本编码和CLIP特征提取
            txt_output = self.txt_encoder(targets)
            if isinstance(txt_output, tuple) and len(txt_output) >= 3:
                txt_feats, z_txt_raw, txt_labels = txt_output
                # 安全地提取z_txt
                z_txt = self._safe_extract_tensor(z_txt_raw)
            else:
                txt_feats = txt_output
                z_txt = None
                txt_labels = None

            # 提取CLIP文本特征
            if CLIP_AVAILABLE and self.clip_alignment.clip_model is not None:
                try:
                    text_tokens = self.clip_alignment.prepare_text_for_clip(targets, self.tokenizer)
                    clip_txt_feats = self.clip_alignment.extract_clip_features(images, text_tokens)
                except Exception as e:
                    print(f"Warning: CLIP text feature extraction failed: {e}")
                    clip_txt_feats = torch.zeros_like(clip_img_feats)
            else:
                clip_txt_feats = torch.zeros_like(clip_img_feats)

            # 多层次对齐
            alignment_feats = self.clip_alignment.multi_level_alignment(
                att_feats, txt_feats, clip_img_feats, clip_txt_feats
            )

            # 更新记忆
            memory = self.update_memory(memory, txt_feats)

            # CLIP增强的特征融合
            if self.clip_fusion is not None:
                enhanced_z_img = self.clip_fusion(torch.cat([z_img, alignment_feats['global_img']], dim=-1))

        elif mode == 'train':
            # 非CLIP模式的训练
            txt_output = self.txt_encoder(targets)
            if isinstance(txt_output, tuple) and len(txt_output) >= 3:
                txt_feats, z_txt_raw, txt_labels = txt_output
                # 安全地提取z_txt
                z_txt = self._safe_extract_tensor(z_txt_raw)
            else:
                txt_feats = txt_output
                z_txt = None
                txt_labels = None
            memory = self.update_memory(memory, txt_feats)

        # 先验选择
        emb_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        prior = self.select_prior(emb_labels, memory)
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        # 生成报告
        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')

            # 返回CLIP增强的输出
            if self.clip_alignment is not None:
                return (output, vis_labels, txt_labels, enhanced_z_img, z_txt,
                        alignment_feats, clip_img_feats, clip_txt_feats)
            else:
                return (output, vis_labels, txt_labels, enhanced_z_img, z_txt)

        elif mode == 'sample':
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError("Invalid mode")


# 保持原有的其他模型类不变
class LAMRGModel(_LAMRG):
    # 初始化模型
    def __init__(self, args, tokenizer):
        # 调用父类的初始化方法
        super(LAMRGModel, self).__init__(args, tokenizer)

        # 定义一个可学习的参数m，形状为(1, num_labels, 40, d_vf)
        self.m = nn.Parameter(torch.FloatTensor(1, args.num_labels, 40, args.d_vf))
        # 初始化参数m
        self.init_m()

        # 根据数据集名称选择前向传播方法
        if args.dataset_name == 'iu_xray':
            # 如果数据集是iu_xray，使用特定的前向传播方法
            self.forward = self.forward_iu_xray
        else:
            # 否则，使用默认的前向传播方法
            self.forward = self.forward_mimic_cxr
    # 初始化m方法
    def init_m(self):
        # 使用正态分布初始化m，均值为0，标准差为1除以标签数量
        nn.init.normal_(self.m, 0, 1 / self.args.num_labels)
    # 定义前向传播函数，处理IU X-ray数据
    # 参数: images: 图像数据，targets: 目标数据，labels: 标签数据，mode: 模式（'train'或'sample'）
    def forward_iu_xray(self, images, targets=None, labels=None, mode='train'):
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])       # 从第一张图像中提取特征
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])       # 从第二张图像中提取特征
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)       # 将两个图像的全连接特征拼接在一起
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)       # 将两个图像的注意力特征拼接在一起
        out_labels = labels_0       # 使用第一张图像的标签作为输出标签
        bs, nf, d_f = att_feats.shape          # 获取注意力特征的形状
        _, n_l, n_m, d_f = self.m.shape        # 获取模型m的形状
        m = labels[:, :, None, None] * self.m.expand(bs, n_l, n_m, d_f)   # 将标签与模型m的特征进行广播乘法
        m = m.reshape(bs, -1, d_f)        # 将结果重塑为新的形状
        att_feats = torch.cat((att_feats, m), dim=1)       # 将重塑后的特征与注意力特征拼接在一起
        if mode == 'train':         # 如果模式为训练
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')       # 调用编码器-解码器进行前向传播
        elif mode == 'sample':       # 如果模式为采样
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')       # 调用编码器-解码器进行采样
        else:       # 如果模式不为训练或采样
            raise ValueError            # 抛出值错误
        return output, out_labels        # 返回输出和输出标签

    def forward_mimic_cxr(self, images, targets=None, labels=None, mode='train'):  # 定义前向传播函数，用于处理MIMIC-CXR数据集，参数包括图像、目标、标签和模式
        att_feats, fc_feats, out_labels = self.visual_extractor(images)  # 通过视觉提取器提取图像的注意力特征、全连接特征和标签
        bs, nf, d_f = att_feats.shape  # 获取注意力特征的形状，bs为批次大小，nf为特征数量，d_f为特征维度
        _, n_l, n_m, d_f = self.m.shape  # 获取模型参数m的形状，n_l为标签数量，n_m为特征数量，d_f为特征维度
        m = labels[:, :, None, None] * self.m.expand(bs, n_l, n_m, d_f)  # 将标签与模型参数m进行广播乘法，生成标签特征
        m = m.reshape(bs, -1, d_f)  # 将生成的标签特征重塑为 (bs, n_l * n_m, d_f) 的形状
        att_feats = torch.cat((att_feats, m), dim=1)  # 将注意力特征与标签特征在维度1上拼接
        if mode == 'train':  # 如果模式为训练
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')  # 调用编码器-解码器进行前向传播
        elif mode == 'sample':  # 如果模式为采样
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')  # 调用编码器-解码器进行采样
        else:  # 如果模式既不是训练也不是采样
            raise ValueError  # 抛出 ValueError 异常

        return output, out_labels  # 返回输出和输出标签


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