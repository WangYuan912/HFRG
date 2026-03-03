import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


class LabelSmoothing(nn.Module):
    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None

    def forward(self, input, target, mask):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1)
        mask = mask.reshape(-1).to(input)

        self.size = input.size(1)
        true_dist = input.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()


class CLIPContrastiveLoss(nn.Module):
    """🔧 优化的CLIP对比学习损失 - 增强稳定性"""

    def __init__(self, temperature=0.07):
        super(CLIPContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        """
        计算优化的CLIP对比学习损失
        Args:
            image_features: [B, D] 图像特征
            text_features: [B, D] 文本特征
        Returns:
            对比学习损失
        """

        # 🔧 安全的特征处理
        def safe_process_features(feats):
            if feats is None:
                return None
            if isinstance(feats, tuple):
                feats = feats[0]  # 取tuple的第一个元素
            if not torch.is_tensor(feats):
                return torch.tensor(feats, dtype=torch.float32)
            return feats.float()

        # 处理输入特征
        image_features = safe_process_features(image_features)
        text_features = safe_process_features(text_features)

        # 检查输入有效性
        if image_features is None or text_features is None:
            device = 'cpu'
            if image_features is not None:
                device = image_features.device
            elif text_features is not None:
                device = text_features.device
            return torch.tensor(0.0, device=device, requires_grad=True)

        if image_features.numel() == 0 or text_features.numel() == 0:
            return torch.tensor(0.0, device=image_features.device, requires_grad=True)

        try:
            # L2归一化
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # 计算相似度矩阵 - 添加数值稳定性
            logits = torch.matmul(image_features, text_features.T) / self.temperature

            # 防止数值溢出
            logits = torch.clamp(logits, min=-50, max=50)

            batch_size = image_features.shape[0]
            labels = torch.arange(batch_size, device=image_features.device)

            # 双向对比损失
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)

            return (loss_i2t + loss_t2i) / 2

        except Exception as e:
            print(f"  CLIP对比损失内部错误: {e}")
            # 返回零损失而不是抛出异常
            device = image_features.device if torch.is_tensor(image_features) else 'cpu'
            return torch.tensor(0.0, device=device, requires_grad=True)


class GlobalAlignmentLoss(nn.Module):
    """🔧 优化的全局对齐损失 - 增强鲁棒性"""

    def __init__(self):
        super(GlobalAlignmentLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, global_img_feats, global_txt_feats):
        """
        计算全局特征对齐损失
        Args:
            global_img_feats: [B, D] 全局图像特征
            global_txt_feats: [B, D] 全局文本特征
        """
        if global_img_feats is None or global_txt_feats is None:
            return torch.tensor(0.0, requires_grad=True)

        try:
            # 确保维度一致
            if global_img_feats.shape != global_txt_feats.shape:
                min_dim = min(global_img_feats.shape[-1], global_txt_feats.shape[-1])
                global_img_feats = global_img_feats[..., :min_dim]
                global_txt_feats = global_txt_feats[..., :min_dim]

            return self.mse_loss(global_img_feats, global_txt_feats)
        except Exception:
            return torch.tensor(0.0, requires_grad=True)


class LocalAlignmentLoss(nn.Module):
    """🔧 优化的局部对齐损失 - 增强稳定性"""

    def __init__(self):
        super(LocalAlignmentLoss, self).__init__()

    def forward(self, local_img_feats, local_txt_feats):
        """
        计算局部特征对齐损失
        Args:
            local_img_feats: [B, N_img, D] 局部图像特征
            local_txt_feats: [B, N_txt, D] 局部文本特征
        """
        if local_img_feats is None or local_txt_feats is None:
            return torch.tensor(0.0, requires_grad=True)

        try:
            # 归一化
            local_img_feats = F.normalize(local_img_feats, dim=-1)
            local_txt_feats = F.normalize(local_txt_feats, dim=-1)

            # 计算跨模态注意力矩阵
            attention_matrix = torch.bmm(local_img_feats, local_txt_feats.transpose(1, 2))

            # 软对齐损失：鼓励每个图像区域与最相关的文本区域对齐
            max_similarities, _ = torch.max(attention_matrix, dim=2)
            loss = -torch.mean(max_similarities)

            return loss
        except Exception:
            return torch.tensor(0.0, requires_grad=True)


class OptimizedCLIPLoss(nn.Module):
    """🚀 优化的CLIP损失函数 - 动态权重衰减，防止过拟合"""

    def __init__(self, args):
        super(OptimizedCLIPLoss, self).__init__()
        self.args = args

        # 基础损失
        self.lm_criterion = LanguageModelCriterion()

        # CLIP相关损失
        self.clip_contrastive = CLIPContrastiveLoss(getattr(args, 'temperature_init', 0.07))
        self.global_align = GlobalAlignmentLoss()
        self.local_align = LocalAlignmentLoss()

        # 🔧 动态权重衰减策略 - 核心创新
        self.initial_clip_weight = getattr(args, 'clip_weight', 0.2)
        self.initial_global_weight = getattr(args, 'global_align_weight', 0.1)
        self.initial_local_weight = getattr(args, 'local_align_weight', 0.1)
        self.weight_decay_factor = 0.95  # 每个epoch权重衰减5%

        # 当前epoch记录
        self.current_epoch = 0

        # 性能历史记录
        self.performance_history = []
        self.best_performance = 0.0

    def update_epoch(self, epoch, current_performance=None):
        """更新当前epoch，调整权重"""
        self.current_epoch = epoch

        if current_performance is not None:
            self.performance_history.append(current_performance)
            if current_performance > self.best_performance:
                self.best_performance = current_performance

    def get_dynamic_weights(self):
        """🔧 获取动态调整的权重 - 防止过拟合的核心机制"""
        # 随着训练进行，逐渐降低CLIP权重，避免过拟合
        # 从第3个epoch开始衰减（因为原始模型在epoch3达到最优）
        decay_start_epoch = 2

        if self.current_epoch <= decay_start_epoch:
            # 前2个epoch保持原权重
            clip_weight = self.initial_clip_weight
            global_weight = self.initial_global_weight
            local_weight = self.initial_local_weight
        else:
            # 从第3个epoch开始衰减
            decay_epochs = max(0, self.current_epoch - decay_start_epoch)
            decay = self.weight_decay_factor ** decay_epochs

            clip_weight = self.initial_clip_weight * decay
            global_weight = self.initial_global_weight * decay
            local_weight = self.initial_local_weight * decay

        # 检测性能下降的激进衰减策略
        if len(self.performance_history) >= 3:
            recent_performances = self.performance_history[-3:]
            if all(recent_performances[i] > recent_performances[i + 1] for i in range(len(recent_performances) - 1)):
                # 连续下降，激进衰减CLIP权重
                clip_weight *= 0.5
                global_weight *= 0.5
                local_weight *= 0.5

        return clip_weight, global_weight, local_weight

    def forward(self, outputs, reports_ids, reports_masks, labels, args):
        """计算优化的总损失"""

        # 解析模型输出
        if isinstance(outputs, tuple) and len(outputs) >= 8:
            (output, vis_labels, txt_labels, z_img, z_txt,
             alignment_feats, clip_img_feats, clip_txt_feats) = outputs[:8]
            use_clip = True
        else:
            if isinstance(outputs, tuple):
                output = outputs[0]
                vis_labels = outputs[1] if len(outputs) > 1 else None
                z_img = outputs[2] if len(outputs) > 2 else None
                z_txt = outputs[3] if len(outputs) > 3 else None
            else:
                output = outputs
                vis_labels = None
                z_img = None
                z_txt = None

            alignment_feats = None
            clip_img_feats = None
            clip_txt_feats = None
            txt_labels = None
            use_clip = False

        # 基础语言模型损失
        lm_loss = self.lm_criterion(
            output[:, :-1],
            reports_ids[:, 1:],
            reports_masks[:, 1:]
        ).mean()

        total_loss = lm_loss
        loss_dict = {'lm_loss': lm_loss.item()}

        # CLIP增强损失（动态权重）
        if use_clip and getattr(args, 'use_clip', False):
            clip_weight, global_weight, local_weight = self.get_dynamic_weights()

            # 记录当前权重
            loss_dict['clip_weight'] = clip_weight
            loss_dict['global_weight'] = global_weight
            loss_dict['local_weight'] = local_weight

            # 只有在权重足够大时才计算CLIP损失
            if clip_weight > 0.01:
                # CLIP对比学习损失
                if clip_img_feats is not None and clip_txt_feats is not None:
                    try:
                        clip_loss = self.clip_contrastive(clip_img_feats, clip_txt_feats)
                        total_loss += clip_weight * clip_loss
                        loss_dict['clip_contrastive'] = clip_loss.item()
                    except Exception as e:
                        print(f"Warning: CLIP对比损失计算失败: {e}")

                # 多层次对齐损失
                if alignment_feats is not None:
                    # 全局对齐损失
                    if ('global_img' in alignment_feats and
                            'global_txt' in alignment_feats and
                            global_weight > 0.01):
                        try:
                            global_loss = self.global_align(
                                alignment_feats['global_img'],
                                alignment_feats['global_txt']
                            )
                            total_loss += global_weight * global_loss
                            loss_dict['global_align'] = global_loss.item()
                        except Exception as e:
                            print(f"Warning: 全局对齐损失计算失败: {e}")

                    # 局部对齐损失
                    if ('local_img' in alignment_feats and
                            'local_txt' in alignment_feats and
                            local_weight > 0.01):
                        try:
                            local_loss = self.local_align(
                                alignment_feats['local_img'],
                                alignment_feats['local_txt']
                            )
                            total_loss += local_weight * local_loss
                            loss_dict['local_align'] = local_loss.item()
                        except Exception as e:
                            print(f"Warning: 局部对齐损失计算失败: {e}")

        # 标签损失（较小权重）
        if getattr(args, 'label_loss', False) and vis_labels is not None:
            try:
                label_criterion = torch.nn.BCEWithLogitsLoss()
                label_loss = label_criterion(vis_labels, labels)
                total_loss += 0.05 * label_loss  # 减小权重
                loss_dict['label_loss'] = label_loss.item()
            except Exception as e:
                print(f"Warning: 标签损失计算失败: {e}")

        loss_dict['total_loss'] = total_loss.item()
        return total_loss


class CLIPEnhancedLoss(nn.Module):
    """🔧 改进的CLIP增强损失函数 - 兼容原有接口"""

    def __init__(self, args):
        super(CLIPEnhancedLoss, self).__init__()
        self.args = args

        # 各种损失函数
        self.lm_criterion = LanguageModelCriterion()
        self.clip_contrastive = CLIPContrastiveLoss(getattr(args, 'temperature_init', 0.07))
        self.global_align = GlobalAlignmentLoss()
        self.local_align = LocalAlignmentLoss()
        self.ranking_loss = RankingLoss()

        # 损失权重 - 优化后的权重
        self.clip_weight = getattr(args, 'clip_weight', 0.2)
        self.global_align_weight = getattr(args, 'global_align_weight', 0.1)
        self.local_align_weight = getattr(args, 'local_align_weight', 0.1)
        self.label_weight = 0.05  # 降低标签权重
        self.rank_weight = 0.05  # 降低排序权重

    def forward(self, outputs, reports_ids, reports_masks, labels, args):
        """
        计算CLIP增强的总损失
        Args:
            outputs: 模型输出，可能包含CLIP相关特征
            reports_ids: 报告ID
            reports_masks: 报告掩码
            labels: 标签
            args: 参数配置
        """
        # 解析模型输出
        if isinstance(outputs, tuple) and len(outputs) >= 8:
            # CLIP增强模型输出
            (output, vis_labels, txt_labels, z_img, z_txt,
             alignment_feats, clip_img_feats, clip_txt_feats) = outputs[:8]
            use_clip = True
        else:
            # 普通模型输出
            if isinstance(outputs, tuple):
                output = outputs[0]
                vis_labels = outputs[1] if len(outputs) > 1 else None
                z_img = outputs[2] if len(outputs) > 2 else None
                z_txt = outputs[3] if len(outputs) > 3 else None
            else:
                output = outputs
                vis_labels = None
                z_img = None
                z_txt = None

            alignment_feats = None
            clip_img_feats = None
            clip_txt_feats = None
            txt_labels = None
            use_clip = False

        # 基础语言模型损失
        lm_loss = self.lm_criterion(output[:, :-1], reports_ids[:, 1:], reports_masks[:, 1:]).mean()

        total_loss = lm_loss
        loss_dict = {'lm_loss': lm_loss.item()}

        # 标签损失
        if getattr(args, 'label_loss', False) and vis_labels is not None:
            try:
                label_criterion = torch.nn.BCEWithLogitsLoss()
                label_loss = label_criterion(vis_labels, labels)
                total_loss += self.label_weight * label_loss
                loss_dict['label_loss'] = label_loss.item()
            except Exception:
                pass

        # 排序损失
        if getattr(args, 'rank_loss', False) and z_img is not None and z_txt is not None:
            try:
                rank_loss = self.ranking_loss(z_img, z_txt, labels)
                total_loss += self.rank_weight * rank_loss
                loss_dict['rank_loss'] = rank_loss.item()
            except Exception:
                pass

        # CLIP增强损失
        if use_clip and getattr(args, 'use_clip', False):
            # CLIP对比学习损失
            if clip_img_feats is not None and clip_txt_feats is not None:
                try:
                    clip_loss = self.clip_contrastive(clip_img_feats, clip_txt_feats)
                    total_loss += self.clip_weight * clip_loss
                    loss_dict['clip_contrastive'] = clip_loss.item()
                except Exception:
                    pass

            # 多层次对齐损失
            if alignment_feats is not None:
                # 全局对齐损失
                if 'global_img' in alignment_feats and 'global_txt' in alignment_feats:
                    try:
                        global_loss = self.global_align(
                            alignment_feats['global_img'],
                            alignment_feats['global_txt']
                        )
                        total_loss += self.global_align_weight * global_loss
                        loss_dict['global_align'] = global_loss.item()
                    except Exception:
                        pass

                # 局部对齐损失
                if 'local_img' in alignment_feats and 'local_txt' in alignment_feats:
                    try:
                        local_loss = self.local_align(
                            alignment_feats['local_img'],
                            alignment_feats['local_txt']
                        )
                        total_loss += self.local_align_weight * local_loss
                        loss_dict['local_align'] = local_loss.item()
                    except Exception:
                        pass

        return total_loss


class RankingLoss(nn.Module):
    """🔧 优化的排序损失 - 增强稳定性"""

    def __init__(self):
        super(RankingLoss, self).__init__()

    def _ensure_tensor(self, x):
        """确保输入是tensor而不是tuple"""
        if x is None:
            return None

        if isinstance(x, tuple):
            # 如果是tuple，尝试从中提取tensor
            for item in x:
                if torch.is_tensor(item) and item.numel() > 0:
                    return item
            return None
        elif torch.is_tensor(x):
            return x
        else:
            try:
                return torch.tensor(x) if x is not None else None
            except:
                return None

    def _safe_norm(self, tensor):
        """安全地计算tensor的norm"""
        tensor = self._ensure_tensor(tensor)
        if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
            return torch.tensor(0.0, requires_grad=True)
        try:
            return torch.norm(tensor)
        except:
            return torch.tensor(0.0, requires_grad=True)

    def forward(self, z_image, z_text, labels, similarity_function='dot'):
        z_image = self._ensure_tensor(z_image)
        z_text = self._ensure_tensor(z_text)

        if z_image is None or z_text is None or z_image.numel() == 0 or z_text.numel() == 0:
            return torch.tensor(0.0, device=labels.device if labels is not None else 'cpu', requires_grad=True)

        try:
            return self.imposter_img_loss(z_image, z_text, labels, similarity_function) + \
                self.imposter_txt_loss(z_image, z_text, labels, similarity_function)
        except Exception:
            return torch.tensor(0.0, device=z_image.device, requires_grad=True)

    def imposter_img_loss(self, z_image, z_text, labels, similarity_function):
        try:
            z_image = self._ensure_tensor(z_image)
            z_text = self._ensure_tensor(z_text)
            loss = torch.zeros(1, device=z_image.device, requires_grad=True)
            batch_size = z_image.size(0)

            for i in range(batch_size):
                j = i + 1 if i < batch_size - 1 else 0
                margin = self._compute_margin(labels[i], labels[j])

                if similarity_function == 'dot':
                    paired_similarity = torch.dot(z_image[i], z_text[i])
                    imposter_similarity = torch.dot(z_image[j], z_text[i])
                elif similarity_function == 'cosine':
                    paired_similarity = torch.dot(z_image[i], z_text[i]) / (
                                self._safe_norm(z_image[i]) * self._safe_norm(z_text[i]) + 1e-8)
                    imposter_similarity = torch.dot(z_image[j], z_text[i]) / (
                                self._safe_norm(z_image[j]) * self._safe_norm(z_text[i]) + 1e-8)
                elif similarity_function == 'l2':
                    paired_similarity = -1 * self._safe_norm(z_image[i] - z_text[i])
                    imposter_similarity = -1 * self._safe_norm(z_image[j] - z_text[i])

                diff_similarity = imposter_similarity - paired_similarity + margin
                if diff_similarity > 0:
                    loss = loss + diff_similarity

            return loss / batch_size
        except Exception as e:
            print(f"Error in imposter_img_loss: {e}")
            return torch.tensor(0.0, device=z_image.device, requires_grad=True)

    def imposter_txt_loss(self, z_image, z_text, labels, similarity_function):
        try:
            z_image = self._ensure_tensor(z_image)
            z_text = self._ensure_tensor(z_text)
            loss = torch.zeros(1, device=z_image.device, requires_grad=True)
            batch_size = z_image.size(0)

            for i in range(batch_size):
                j = i + 1 if i < batch_size - 1 else 0
                margin = self._compute_margin(labels[i], labels[j])

                if similarity_function == 'dot':
                    paired_similarity = torch.dot(z_image[i], z_text[i])
                    imposter_similarity = torch.dot(z_text[j], z_image[i])
                elif similarity_function == 'cosine':
                    paired_similarity = torch.dot(z_image[i], z_text[i]) / (
                                self._safe_norm(z_image[i]) * self._safe_norm(z_text[i]) + 1e-8)
                    imposter_similarity = torch.dot(z_text[j], z_image[i]) / (
                                self._safe_norm(z_text[j]) * self._safe_norm(z_image[i]) + 1e-8)
                elif similarity_function == 'l2':
                    paired_similarity = -1 * self._safe_norm(z_image[i] - z_text[i])
                    imposter_similarity = -1 * self._safe_norm(z_text[j] - z_image[i])

                diff_similarity = imposter_similarity - paired_similarity + margin
                if diff_similarity > 0:
                    loss = loss + diff_similarity

            return loss / batch_size
        except Exception as e:
            print(f"Error in imposter_txt_loss: {e}")
            return torch.tensor(0.0, device=z_image.device, requires_grad=True)

    def _compute_margin(self, label_i, label_j):
        try:
            if torch.equal(label_i, label_j):
                return 0
            else:
                n = (label_i.int() | label_j.int()).sum().item()
                diff = (label_i.int() ^ label_j.int()).sum().item()
                return max(0.5, diff / n) if n > 0 else 0.5
        except:
            return 0.5


def compute_fallback_loss(output, reports_ids, reports_masks):
    """回退损失计算（最简单的语言模型损失）"""
    criterion = LanguageModelCriterion()

    if isinstance(output, tuple):
        main_output = output[0]
    else:
        main_output = output

    return criterion(main_output[:, :-1], reports_ids[:, 1:], reports_masks[:, 1:]).mean()


def compute_standard_loss(output, reports_ids, reports_masks, labels=None,
                          vis_label=None, txt_label=None, z_img=None, z_txt=None,
                          args={}, similarity_function='dot'):
    """标准损失计算（原来的逻辑）"""
    criterion = LanguageModelCriterion()

    if isinstance(output, tuple):
        main_output = output[0]
        vis_label = output[1] if len(output) > 1 and vis_label is None else vis_label
        z_img = output[2] if len(output) > 2 and z_img is None else z_img
        z_txt = output[3] if len(output) > 3 and z_txt is None else z_txt
    else:
        main_output = output

    loss = criterion(main_output[:, :-1], reports_ids[:, 1:], reports_masks[:, 1:]).mean()

    label_loss, match_loss = 0, 0
    if getattr(args, 'label_loss', False) and vis_label is not None and labels is not None:
        try:
            label_criterion = torch.nn.BCEWithLogitsLoss()
            label_loss = label_criterion(vis_label, labels)
        except:
            label_loss = 0

    if getattr(args, 'rank_loss', False) and z_img is not None and z_txt is not None and labels is not None:
        try:
            ranking_loss = RankingLoss()
            match_loss = ranking_loss(z_img, z_txt, labels, similarity_function)
        except:
            match_loss = 0

    return loss + 0.05 * label_loss + 0.05 * match_loss  # 降低额外损失权重


def compute_loss(*args, **kwargs):
    """
    🚀 优化的损失计算主函数 - 智能路由和错误处理
    支持两种调用方式：
    1. compute_loss(output, reports_ids, reports_masks, labels, args, ...)
    2. compute_loss(outputs_tuple, reports_ids, reports_masks, labels, args)
    """
    try:
        # 处理位置参数
        if len(args) >= 4:
            output = args[0]
            reports_ids = args[1]
            reports_masks = args[2]
            labels = args[3]

            # 尝试获取args参数
            if len(args) > 4:
                args_param = args[4] if hasattr(args[4], '__dict__') else {}
            else:
                args_param = kwargs.get('args', {})

            # 其他可选参数
            vis_label = args[5] if len(args) > 5 else kwargs.get('vis_label', None)
            txt_label = args[6] if len(args) > 6 else kwargs.get('txt_label', None)
            z_img = args[7] if len(args) > 7 else kwargs.get('z_img', None)
            z_txt = args[8] if len(args) > 8 else kwargs.get('z_txt', None)
            similarity_function = args[9] if len(args) > 9 else kwargs.get('similarity_function', 'dot')

        elif len(kwargs) >= 4:
            # 关键字参数方式
            output = kwargs.get('output')
            reports_ids = kwargs.get('reports_ids')
            reports_masks = kwargs.get('reports_masks')
            labels = kwargs.get('labels')
            args_param = kwargs.get('args', {})
            vis_label = kwargs.get('vis_label', None)
            txt_label = kwargs.get('txt_label', None)
            z_img = kwargs.get('z_img', None)
            z_txt = kwargs.get('z_txt', None)
            similarity_function = kwargs.get('similarity_function', 'dot')

        else:
            raise ValueError("Insufficient arguments provided to compute_loss")

        # 🔧 智能损失路由 - 根据模型类型选择最佳损失函数
        if getattr(args_param, 'use_clip', False) and isinstance(output, tuple) and len(output) >= 8:
            # 使用优化的CLIP损失
            optimized_loss_fn = OptimizedCLIPLoss(args_param)

            # 尝试获取当前epoch信息（用于动态权重调整）
            current_epoch = getattr(args_param, 'current_epoch', 0)
            if hasattr(args_param, 'current_performance'):
                optimized_loss_fn.update_epoch(current_epoch, args_param.current_performance)
            else:
                optimized_loss_fn.update_epoch(current_epoch)

            return optimized_loss_fn(output, reports_ids, reports_masks, labels, args_param)
        else:
            # 使用标准损失计算
            return compute_standard_loss(output, reports_ids, reports_masks, labels,
                                         vis_label, txt_label, z_img, z_txt,
                                         args_param, similarity_function)

    except Exception as e:
        print(f"Warning: 优化损失计算失败，使用回退策略: {e}")
        # 回退到最基础的损失计算
        if len(args) >= 3:
            return compute_fallback_loss(args[0], args[1], args[2])
        elif 'output' in kwargs and 'reports_ids' in kwargs and 'reports_masks' in kwargs:
            return compute_fallback_loss(kwargs['output'], kwargs['reports_ids'], kwargs['reports_masks'])
        else:
            raise ValueError("Cannot compute fallback loss with provided arguments")


def compute_simplified_clip_loss(outputs, reports_ids, reports_masks, labels, args):
    """简化的CLIP损失计算"""
    try:
        # 解析输出
        if isinstance(outputs, tuple) and len(outputs) >= 6:
            output, vis_labels, _, _, _, alignment_feats = outputs[:6]

            # 基础损失
            lm_criterion = LanguageModelCriterion()
            lm_loss = lm_criterion(output[:, :-1], reports_ids[:, 1:], reports_masks[:, 1:]).mean()

            total_loss = lm_loss

            # 简化的CLIP损失
            if alignment_feats is not None and getattr(args, 'use_clip', False):
                try:
                    proj_img = alignment_feats.get('proj_img')
                    proj_txt = alignment_feats.get('proj_txt')
                    clip_img = alignment_feats.get('clip_img')
                    clip_txt = alignment_feats.get('clip_txt')

                    if proj_img is not None and proj_txt is not None:
                        # 简单的对比损失
                        proj_img_norm = F.normalize(proj_img, dim=-1)
                        proj_txt_norm = F.normalize(proj_txt, dim=-1)

                        logits = torch.matmul(proj_img_norm, proj_txt_norm.T) / 0.07
                        batch_size = proj_img.shape[0]
                        labels_clip = torch.arange(batch_size, device=proj_img.device)

                        clip_loss = F.cross_entropy(logits, labels_clip)
                        total_loss += getattr(args, 'clip_weight', 0.25) * clip_loss

                except Exception as e:
                    print(f"⚠️ CLIP loss failed: {e}")

            return total_loss
        else:
            # 回退到基础损失
            if isinstance(outputs, tuple):
                output = outputs[0]
            else:
                output = outputs

            lm_criterion = LanguageModelCriterion()
            return lm_criterion(output[:, :-1], reports_ids[:, 1:], reports_masks[:, 1:]).mean()

    except Exception as e:
        print(f"❌ Loss calculation completely failed: {e}")
        # 最基础的回退
        lm_criterion = LanguageModelCriterion()
        main_output = outputs[0] if isinstance(outputs, tuple) else outputs
        return lm_criterion(main_output[:, :-1], reports_ids[:, 1:], reports_masks[:, 1:]).mean()


class PerformanceDrivenLoss(nn.Module):
    """🎯 性能导向的动态损失权重调整"""

    def __init__(self, args):
        super(PerformanceDrivenLoss, self).__init__()
        self.args = args
        self.lm_criterion = LanguageModelCriterion()
        self.clip_contrastive = CLIPContrastiveLoss(args.temperature_init)

        # 🎯 目标指标权重映射
        self.target_weights = {
            'BLEU_4': 0.3,  # 重点提升BLEU_4
            'METEOR': 0.25,  # 重点提升METEOR
            'CIDEr': 0.2,  # 保持CIDEr
            'BLEU_1': 0.15,
            'ROUGE_L': 0.1
        }

        # 动态权重
        self.current_performance = {}
        self.performance_history = []

    def update_performance(self, metrics):
        """更新当前性能指标"""
        self.current_performance = metrics
        self.performance_history.append(metrics)

    def get_adaptive_weights(self):
        """🔧 根据当前性能计算自适应权重"""
        if not self.current_performance:
            return {
                'clip_weight': getattr(self.args, 'clip_weight', 0.25),
                'meteor_boost': 1.0,
                'bleu4_boost': 1.0
            }

        # 计算距离目标的差距
        bleu4_gap = max(0, 0.20 - self.current_performance.get('BLEU_4', 0))
        meteor_gap = max(0, 0.22 - self.current_performance.get('METEOR', 0))
        cider_gap = max(0, 0.40 - self.current_performance.get('CIDEr', 0))

        # 动态调整权重
        meteor_boost = 1.0 + meteor_gap * 3.0  # METEOR差距越大，权重越高
        bleu4_boost = 1.0 + bleu4_gap * 4.0  # BLEU_4差距越大，权重越高

        # CLIP权重根据整体性能调整
        avg_performance = (
                self.current_performance.get('BLEU_4', 0) * 0.3 +
                self.current_performance.get('METEOR', 0) * 0.25 +
                self.current_performance.get('CIDEr', 0) * 0.2 +
                self.current_performance.get('BLEU_1', 0) * 0.15 +
                self.current_performance.get('ROUGE_L', 0) * 0.1
        )

        # 性能越低，CLIP权重越低，更专注基础语言建模
        clip_weight = getattr(self.args, 'clip_weight', 0.25) * (0.5 + avg_performance)

        return {
            'clip_weight': min(0.4, max(0.1, clip_weight)),
            'meteor_boost': min(2.0, meteor_boost),
            'bleu4_boost': min(2.5, bleu4_boost)
        }

    def forward(self, outputs, reports_ids, reports_masks, labels, args):
        """🎯 性能导向的损失计算"""

        # 基础语言模型损失
        if isinstance(outputs, tuple):
            output = outputs[0]
        else:
            output = outputs

        lm_loss = self.lm_criterion(
            output[:, :-1],
            reports_ids[:, 1:],
            reports_masks[:, 1:]
        ).mean()

        # 获取自适应权重
        weights = self.get_adaptive_weights()

        # 🔧 增强语言模型损失以提升METEOR和BLEU_4
        enhanced_lm_loss = lm_loss * (
                weights['meteor_boost'] * 0.4 +
                weights['bleu4_boost'] * 0.6
        )

        total_loss = enhanced_lm_loss

        # CLIP损失（如果可用）
        if (isinstance(outputs, tuple) and len(outputs) >= 8 and
                getattr(args, 'use_clip', False)):

            try:
                _, _, _, _, _, alignment_feats, clip_img_feats, clip_txt_feats = outputs[:8]

                if clip_img_feats is not None and clip_txt_feats is not None:
                    clip_loss = self.clip_contrastive(clip_img_feats, clip_txt_feats)
                    total_loss += weights['clip_weight'] * clip_loss

            except Exception as e:
                print(f" CLIP loss computation failed: {e}")

        return total_loss


def compute_performance_driven_loss(outputs, reports_ids, reports_masks, labels, args, current_metrics=None):
    """🎯 使用性能导向损失的主函数"""

    # 创建或获取全局损失计算器
    if not hasattr(compute_performance_driven_loss, 'loss_calculator'):
        compute_performance_driven_loss.loss_calculator = PerformanceDrivenLoss(args)

    # 更新性能指标
    if current_metrics:
        compute_performance_driven_loss.loss_calculator.update_performance(current_metrics)

    return compute_performance_driven_loss.loss_calculator(
        outputs, reports_ids, reports_masks, labels, args
    )