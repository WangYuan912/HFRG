# modules/enhanced_decoding.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedBeamSearch:
    """🚀 增强的束搜索 - 专门优化METEOR和BLEU_4"""

    def __init__(self, model, tokenizer, beam_size=5, max_length=60):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_length = max_length

        # 🎯 针对医学报告的奖励函数
        self.medical_keywords = {
            'anatomy': ['heart', 'lung', 'chest', 'rib', 'spine', 'mediastinum'],
            'findings': ['normal', 'abnormal', 'enlarged', 'opacity', 'clear'],
            'quality': ['stable', 'improved', 'unchanged', 'acute', 'chronic']
        }

    def calculate_medical_reward(self, sequence):
        """🔧 计算医学相关性奖励"""
        text = self.tokenizer.decode_batch([sequence])[0].lower()
        reward = 0.0

        for category, keywords in self.medical_keywords.items():
            category_hits = sum(1 for kw in keywords if kw in text)
            if category == 'anatomy':
                reward += category_hits * 0.3  # 解剖结构很重要
            elif category == 'findings':
                reward += category_hits * 0.5  # 发现描述最重要
            elif category == 'quality':
                reward += category_hits * 0.2  # 质量描述适中

        # 奖励完整的句子结构
        if '.' in text and len(text.split()) >= 5:
            reward += 0.3

        return reward

    def diversity_penalty(self, sequences):
        """🔧 多样性惩罚 - 避免重复生成"""
        if len(sequences) <= 1:
            return [0.0] * len(sequences)

        penalties = []
        for i, seq in enumerate(sequences):
            penalty = 0.0
            for j, other_seq in enumerate(sequences):
                if i != j:
                    # 计算序列相似度
                    overlap = len(set(seq) & set(other_seq)) / max(len(set(seq)), 1)
                    penalty += overlap * 0.1
            penalties.append(penalty)

        return penalties

    def beam_search_with_rewards(self, img_feats, att_feats):
        """🎯 带奖励机制的束搜索"""
        batch_size = img_feats.size(0)
        device = img_feats.device

        # 初始化束搜索状态
        beams = [[(torch.tensor([self.tokenizer.bos_idx], device=device), 0.0, 0.0)]
                 for _ in range(batch_size)]

        for step in range(self.max_length):
            new_beams = [[] for _ in range(batch_size)]

            for batch_idx in range(batch_size):
                candidates = []

                for seq, score, reward in beams[batch_idx]:
                    if seq[-1] == self.tokenizer.eos_idx:
                        candidates.append((seq, score, reward))
                        continue

                    # 获取下一个token的概率分布
                    with torch.no_grad():
                        outputs = self.model.encoder_decoder(
                            img_feats[batch_idx:batch_idx + 1],
                            att_feats[batch_idx:batch_idx + 1],
                            seq.unsqueeze(0),
                            mode='forward'
                        )
                        logits = outputs[:, -1, :]  # 最后一步的logits
                        probs = torch.softmax(logits, dim=-1)

                    # 选择top-k候选
                    top_probs, top_indices = torch.topk(probs, self.beam_size * 2)

                    for prob, idx in zip(top_probs[0], top_indices[0]):
                        new_seq = torch.cat([seq, idx.unsqueeze(0)])
                        new_score = score + torch.log(prob).item()

                        # 🎯 计算医学奖励
                        medical_reward = self.calculate_medical_reward(new_seq.cpu().numpy())
                        new_reward = reward + medical_reward

                        # 🔧 综合评分：语言模型分数 + 医学奖励
                        combined_score = new_score + new_reward * 0.1

                        candidates.append((new_seq, new_score, new_reward))

                # 🔧 应用多样性惩罚
                if len(candidates) > 1:
                    sequences = [cand[0] for cand in candidates]
                    penalties = self.diversity_penalty(sequences)
                    candidates = [(seq, score - penalty, reward)
                                  for (seq, score, reward), penalty in zip(candidates, penalties)]

                # 选择最佳候选
                candidates.sort(key=lambda x: x[1] + x[2] * 0.1, reverse=True)
                new_beams[batch_idx] = candidates[:self.beam_size]

            beams = new_beams

        # 返回最佳序列
        results = []
        for batch_beams in beams:
            if batch_beams:
                best_seq = max(batch_beams, key=lambda x: x[1] + x[2] * 0.1)[0]
                results.append(best_seq)
            else:
                # 回退方案
                results.append(torch.tensor([self.tokenizer.bos_idx, self.tokenizer.eos_idx],
                                            device=device))

        return torch.stack([torch.nn.functional.pad(seq, (0, self.max_length - len(seq)),
                                                    value=self.tokenizer.pad_idx)
                            for seq in results])


def integrate_enhanced_decoding(model, tokenizer):
    """🔧 集成增强解码到模型中"""

    enhanced_decoder = EnhancedBeamSearch(model, tokenizer, beam_size=5)

    # 重写模型的sample方法
    original_sample = model.encoder_decoder.sample

    def enhanced_sample(fc_feats, att_feats, opt=None, **kwargs):
        if getattr(opt, 'enhanced_decoding', False):
            return enhanced_decoder.beam_search_with_rewards(fc_feats, att_feats)
        else:
            return original_sample(fc_feats, att_feats, opt, **kwargs)

    model.encoder_decoder.sample = enhanced_sample
    return model