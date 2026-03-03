# 🚀 优化的trainer.py - 解决过拟合问题，增强训练稳定性
# 🚀 优化的trainer.py - 解决过拟合问题，增强训练稳定性

import os
import logging
from abc import abstractmethod
import json
import numpy as np
import time
import torch
import pandas as pd
from scipy import sparse
from numpy import inf

from tqdm import tqdm
from tensorboardX import SummaryWriter

METRICS = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'CIDEr', 'ROUGE_L', 'METEOR']


class EnhancedEarlyStopping:
    """🔧 增强的早停机制 - 专门针对过拟合优化"""

    def __init__(self, patience=3, min_delta=0.001, mode='max', restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False

        # 性能历史记录
        self.score_history = []
        self.improvement_history = []

    def __call__(self, val_score, epoch):
        """检查是否应该早停"""
        self.score_history.append(val_score)

        if self.best_score is None:
            self.best_score = val_score
            self.best_epoch = epoch
            self.improvement_history.append(True)
        elif self._is_better(val_score, self.best_score):
            improvement = val_score - self.best_score if self.mode == 'max' else self.best_score - val_score
            self.improvement_history.append(True)

            self.best_score = val_score
            self.best_epoch = epoch
            self.counter = 0

            logging.info(f" 新的最佳性能! Epoch {epoch}: {val_score:.4f} (提升: +{improvement:.4f})")
        else:
            self.improvement_history.append(False)
            self.counter += 1

            decline = self.best_score - val_score if self.mode == 'max' else val_score - self.best_score
            logging.info(
                f"  性能未提升 ({self.counter}/{self.patience}): Epoch {epoch}: {val_score:.4f} (相比最佳下降: -{decline:.4f})")

            # 检测连续下降趋势
            if len(self.score_history) >= 3:
                recent_scores = self.score_history[-3:]
                if self._is_declining_trend(recent_scores):
                    logging.warning(f" 检测到连续下降趋势: {recent_scores}")

        if self.counter >= self.patience:
            self.early_stop = True
            logging.info(f" 早停触发! 最佳性能在第 {self.best_epoch} 轮: {self.best_score:.4f}")

        return self.early_stop

    def _is_better(self, current, best):
        """判断当前性能是否更好"""
        if self.mode == 'max':
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta

    def _is_declining_trend(self, scores):
        """检测是否为连续下降趋势"""
        if len(scores) < 3:
            return False

        if self.mode == 'max':
            return all(scores[i] > scores[i + 1] for i in range(len(scores) - 1))
        else:
            return all(scores[i] < scores[i + 1] for i in range(len(scores) - 1))


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args
        # tensorboard 记录参数和结果
        self.writer = SummaryWriter(args.save_dir)
        self.print_args2tensorbord()

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf

        # 🔧 使用增强的早停机制
        self.enhanced_early_stopping = EnhancedEarlyStopping(
            patience=getattr(self.args, 'early_stop', 3),
            min_delta=0.001,  # CIDEr需要至少0.001的提升才算改进
            mode=self.mnt_mode,
            restore_best_weights=True
        )

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        # 确保record_dir存在
        if not hasattr(args, 'record_dir') or args.record_dir is None:
            self.args.record_dir = os.path.join(args.save_dir, 'records')

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        # 添加所有epoch结果的记录
        self.all_epochs_results = []
        self.epoch_summary = []

        # 🔧 性能跟踪
        self.performance_tracker = {
            'epoch_times': [],
            'loss_history': [],
            'val_scores': [],
            'test_scores': [],
            'lr_history': []
        }

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        """🚀 优化的训练主循环 - 防止过拟合"""
        logging.info(f"\n{'=' * 80}")
        logging.info(f" 开始训练 - 优化版本")
        logging.info(f" 总共 {self.epochs} 个 epoch")
        logging.info(f" 监控指标: {self.args.monitor_metric}")
        logging.info(f" 早停容忍度: {self.enhanced_early_stopping.patience}")
        logging.info(f" 模型保存目录: {self.checkpoint_dir}")
        logging.info(f" 结果记录目录: {self.args.record_dir}")
        logging.info(f"{'=' * 80}")

        for epoch in range(self.start_epoch, self.epochs + 1):
            try:
                epoch_start_time = time.time()

                # 🔧 记录学习率，支持CLIP模型的多参数组
                current_lrs = []
                if len(self.optimizer.param_groups) >= 2:
                    current_lrs = [self.optimizer.param_groups[0]["lr"], self.optimizer.param_groups[1]["lr"]]
                    logging.info(
                        f' Epoch {epoch}/{self.epochs} | '
                        f'VE lr: {current_lrs[0]:.7f}, '
                        f'Model lr: {current_lrs[1]:.7f}')
                else:
                    current_lrs = [self.optimizer.param_groups[0]["lr"]]
                    logging.info(f' Epoch {epoch}/{self.epochs} | lr: {current_lrs[0]:.7f}')

                self.performance_tracker['lr_history'].append(current_lrs)

                # 🔧 传递当前epoch信息给损失函数（用于动态权重调整）
                if hasattr(self.args, 'use_clip') and self.args.use_clip:
                    self.args.current_epoch = epoch

                result = self._train_epoch(epoch)
                log = {'epoch': epoch}
                log.update(result)

                # 记录训练时间
                epoch_time = time.time() - epoch_start_time
                self.performance_tracker['epoch_times'].append(epoch_time)
                self.performance_tracker['loss_history'].append(log.get('train_loss', 0))

                # 记录当前epoch结果
                self.all_epochs_results.append(log.copy())

                # 🔧 更新性能跟踪
                val_score = log.get(self.mnt_metric, 0)
                test_score = log.get(self.mnt_metric_test, 0)
                self.performance_tracker['val_scores'].append(val_score)
                self.performance_tracker['test_scores'].append(test_score)

                # 🔧 传递性能信息给损失函数
                if hasattr(self.args, 'use_clip') and self.args.use_clip:
                    self.args.current_performance = val_score

                self._record_best(log)
                self._print_epoch(log, epoch_time)

                # 创建epoch总结
                epoch_summary = {
                    'epoch': epoch,
                    'train_loss': log.get('train_loss', 0),
                    'val_BLEU_4': log.get('val_BLEU_4', 0),
                    'test_BLEU_4': log.get('test_BLEU_4', 0),
                    'val_CIDEr': log.get('val_CIDEr', 0),
                    'test_CIDEr': log.get('test_CIDEr', 0),
                    'epoch_time': epoch_time,
                    'learning_rate': current_lrs[0] if current_lrs else 0
                }
                self.epoch_summary.append(epoch_summary)

                # 🔧 使用增强早停机制
                current_val_score = log.get(self.mnt_metric, -inf if self.mnt_mode == 'max' else inf)

                # 检查早停
                if self.enhanced_early_stopping(current_val_score, epoch):
                    logging.info(f" 增强早停触发，停止训练")
                    self._save_checkpoint(epoch, save_best=True, early_stop=True)
                    break

                # 🔧 传统早停逻辑保留作为备份
                improved = False
                if self.mnt_mode != 'off':
                    try:
                        improved = (self.mnt_mode == 'min' and current_val_score <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and current_val_score >= self.mnt_best)
                    except KeyError:
                        logging.error(
                            f"Warning: Metric '{self.mnt_metric}' is not found. Model performance monitoring is disabled.")
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = current_val_score
                        logging.info(f" 传统早停: 新的最佳结果! {self.mnt_metric}: {self.mnt_best:.4f}")

                # 保存检查点
                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch, save_best=improved)

                # 每个epoch结束后显示历史最佳结果
                self._print_current_best()

                # 🔧 性能分析和警告
                self._analyze_performance(epoch)

                # 每5个epoch保存一次历史结果
                if epoch % 5 == 0:
                    self._save_epoch_history()

            except KeyboardInterrupt:
                logging.info(' 用户手动停止训练!')
                self._save_checkpoint(epoch, save_best=False, interrupt=True)
                logging.info(' 已保存中断检查点!')
                if epoch > 1:
                    self._print_best()
                    self._print_best_to_file()
                    self._save_epoch_history()
                    self._save_performance_analysis()
                return

        logging.info(f" 训练完成!")
        self._print_best()
        self._print_best_to_file()
        self._save_epoch_history()
        self._save_performance_analysis()

    def _analyze_performance(self, current_epoch):
        """🔧 性能分析和过拟合检测"""
        if len(self.performance_tracker['val_scores']) < 3:
            return

        recent_val_scores = self.performance_tracker['val_scores'][-3:]
        recent_train_losses = self.performance_tracker['loss_history'][-3:]

        # 检测过拟合信号
        if len(recent_val_scores) >= 3:
            val_declining = all(
                recent_val_scores[i] > recent_val_scores[i + 1] for i in range(len(recent_val_scores) - 1))
            loss_declining = all(
                recent_train_losses[i] > recent_train_losses[i + 1] for i in range(len(recent_train_losses) - 1))

            if val_declining and loss_declining:
                logging.warning(f"   •检测到过拟合信号:")
                logging.warning(f"   • 验证分数连续下降: {[f'{s:.4f}' for s in recent_val_scores]}")
                logging.warning(f"   • 训练损失仍在下降: {[f'{l:.4f}' for l in recent_train_losses]}")
                logging.warning(f"   • 建议考虑早停或降低学习率")

        # 学习率建议
        if current_epoch > 5:
            current_lr = self.performance_tracker['lr_history'][-1][0]
            if current_lr > 1e-5 and not self.enhanced_early_stopping.improvement_history[-3:].count(True):
                if len([h for h in self.enhanced_early_stopping.improvement_history[-5:] if h]) <= 1:
                    logging.info(f" 建议: 学习率可能过高 (当前: {current_lr:.2e})，考虑降低")

    def _print_current_best(self):
        """🔧 优化的最佳结果显示"""
        if 'epoch' in self.best_recorder['val']:
            val_best = self.best_recorder['val']
            test_best = self.best_recorder['test']

            val_epoch = val_best['epoch']
            test_epoch = test_best['epoch']
            val_cider = val_best.get('val_CIDEr', 0)
            test_cider = test_best.get('test_CIDEr', 0)
            val_bleu4 = val_best.get('val_BLEU_4', 0)
            test_bleu4 = test_best.get('test_BLEU_4', 0)

            logging.info(f"\n 当前最佳结果:")
            logging.info(f"    Val  Best (Epoch {val_epoch:2d}): CIDEr={val_cider:.4f}, BLEU_4={val_bleu4:.4f}")
            logging.info(f"    Test Best (Epoch {test_epoch:2d}): CIDEr={test_cider:.4f}, BLEU_4={test_bleu4:.4f}")

            # 显示性能趋势
            if len(self.performance_tracker['val_scores']) >= 3:
                recent_trend = self.performance_tracker['val_scores'][-3:]
                trend_symbol = "-" if recent_trend[-1] > recent_trend[0] else "-" if recent_trend[-1] < recent_trend[
                    0] else "-"
                logging.info(f"   {trend_symbol} 最近趋势: {' → '.join([f'{s:.3f}' for s in recent_trend])}")

    def _save_epoch_history(self):
        """保存所有epoch的历史结果"""
        if self.epoch_summary:
            history_df = pd.DataFrame(self.epoch_summary)
            history_path = os.path.join(self.args.record_dir, f'{self.args.dataset_name}_epoch_history.csv')
            history_df.to_csv(history_path, index=False)
            logging.info(f" 历史结果已保存到: {history_path}")

    def _save_performance_analysis(self):
        """🔧 保存性能分析结果"""
        analysis_path = os.path.join(self.args.record_dir, f'{self.args.dataset_name}_performance_analysis.json')

        analysis_data = {
            'total_epochs': len(self.performance_tracker['epoch_times']),
            'total_training_time': sum(self.performance_tracker['epoch_times']),
            'avg_epoch_time': np.mean(self.performance_tracker['epoch_times']),
            'best_val_score': max(self.performance_tracker['val_scores']) if self.performance_tracker[
                'val_scores'] else 0,
            'best_test_score': max(self.performance_tracker['test_scores']) if self.performance_tracker[
                'test_scores'] else 0,
            'final_lr': self.performance_tracker['lr_history'][-1] if self.performance_tracker['lr_history'] else [],
            'early_stop_triggered': self.enhanced_early_stopping.early_stop,
            'early_stop_epoch': self.enhanced_early_stopping.best_epoch if self.enhanced_early_stopping.early_stop else None,
            'performance_tracker': self.performance_tracker
        }

        with open(analysis_path, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)

        logging.info(f" 性能分析已保存到: {analysis_path}")

    def print_args2tensorbord(self):
        for k, v in vars(self.args).items():
            self.writer.add_text(k, str(v))

    def _print_best_to_file(self):
        """🔧 优化的最佳结果保存"""
        crt_time = time.asctime(time.localtime(time.time()))
        for split in ['val', 'test']:
            if 'epoch' not in self.best_recorder[split]:
                logging.warning(f"No best results recorded for {split} split")
                continue

            self.best_recorder[split]['version'] = f'V{self.args.version}'
            self.best_recorder[split]['visual_extractor'] = self.args.visual_extractor
            self.best_recorder[split]['time'] = crt_time
            self.best_recorder[split]['seed'] = self.args.seed
            self.best_recorder[split]['best_model_from'] = 'val'
            self.best_recorder[split]['lr'] = self.args.lr_ed
            self.best_recorder[split]['dataset'] = self.args.dataset_name

            # 🔧 添加优化相关信息
            self.best_recorder[split]['early_stop_patience'] = self.enhanced_early_stopping.patience
            self.best_recorder[split]['dropout'] = self.args.dropout
            self.best_recorder[split]['weight_decay'] = self.args.weight_decay
            self.best_recorder[split]['lr_scheduler'] = self.args.lr_scheduler

            # 添加CLIP相关信息
            if getattr(self.args, 'use_clip', False):
                self.best_recorder[split]['use_clip'] = True
                self.best_recorder[split]['clip_model'] = getattr(self.args, 'clip_model_name', 'ViT-B/32')
                self.best_recorder[split]['clip_weight'] = getattr(self.args, 'clip_weight', 0.2)
                self.best_recorder[split]['global_align_weight'] = getattr(self.args, 'global_align_weight', 0.1)
                self.best_recorder[split]['local_align_weight'] = getattr(self.args, 'local_align_weight', 0.1)
            else:
                self.best_recorder[split]['use_clip'] = False

        # 确保record_dir存在
        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)

        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.csv')

        try:
            if not os.path.exists(record_path):
                record_table = pd.DataFrame()
            else:
                record_table = pd.read_csv(record_path)

            # 只添加有有效结果的记录
            if 'epoch' in self.best_recorder['val']:
                record_table = pd.concat([record_table, pd.DataFrame([self.best_recorder['val']])], ignore_index=True)
            if 'epoch' in self.best_recorder['test']:
                record_table = pd.concat([record_table, pd.DataFrame([self.best_recorder['test']])], ignore_index=True)

            record_table.to_csv(record_path, index=False)
            logging.info(f" 最佳结果已保存到: {record_path}")

        except Exception as e:
            logging.error(f" 保存CSV文件时出错: {e}")

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            logging.info("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            logging.info(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False, interrupt=False, early_stop=False):
        """🔧 优化的检查点保存"""
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'best_recorder': self.best_recorder,
            'all_epochs_results': self.all_epochs_results,
            'performance_tracker': self.performance_tracker,
            'early_stopping_state': {
                'best_score': self.enhanced_early_stopping.best_score,
                'best_epoch': self.enhanced_early_stopping.best_epoch,
                'counter': self.enhanced_early_stopping.counter,
                'score_history': self.enhanced_early_stopping.score_history
            }
        }

        if interrupt:
            filename = os.path.join(self.checkpoint_dir, 'interrupt_checkpoint.pth')
            logging.info(" 保存中断检查点...")
        elif early_stop:
            filename = os.path.join(self.checkpoint_dir, 'early_stop_checkpoint.pth')
            logging.info(" 保存早停检查点...")
        else:
            filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')

        torch.save(state, filename)
        logging.debug(f" 保存检查点: {filename}")

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            logging.info(" 保存最佳模型: model_best.pth")

    def _resume_checkpoint(self, resume_path):
        """🔧 优化的检查点恢复"""
        resume_path = str(resume_path)
        logging.info(f" 加载检查点: {resume_path}")
        checkpoint = torch.load(resume_path, weights_only=False)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        # 恢复历史记录
        if 'best_recorder' in checkpoint:
            self.best_recorder = checkpoint['best_recorder']
        if 'all_epochs_results' in checkpoint:
            self.all_epochs_results = checkpoint['all_epochs_results']
        if 'performance_tracker' in checkpoint:
            self.performance_tracker = checkpoint['performance_tracker']

        # 恢复早停状态
        if 'early_stopping_state' in checkpoint:
            es_state = checkpoint['early_stopping_state']
            self.enhanced_early_stopping.best_score = es_state.get('best_score')
            self.enhanced_early_stopping.best_epoch = es_state.get('best_epoch', 0)
            self.enhanced_early_stopping.counter = es_state.get('counter', 0)
            self.enhanced_early_stopping.score_history = es_state.get('score_history', [])

        logging.info(f" 检查点加载完成，从第 {self.start_epoch} 轮继续训练")

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)
            self.writer.add_text(f'best_BLEU4_byVal', str(log["test_BLEU_4"]), log["epoch"])

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)
            self.writer.add_text(f'best_BLEU4_byTest', str(log["test_BLEU_4"]), log["epoch"])

    def _print_best(self):
        """🔧 优化的最佳结果打印"""
        logging.info('\n' + '-' * 20 + ' 最终最佳结果 ' + '-' * 20)
        logging.info(f' 最佳结果 (基于 {self.args.monitor_metric}) - 验证集:')
        self._prin_metrics(self.best_recorder['val'], summary=True)

        logging.info(f' 最佳结果 (基于 {self.args.monitor_metric}) - 测试集:')
        self._prin_metrics(self.best_recorder['test'], summary=True)

        # 🔧 添加训练总结
        if self.performance_tracker['epoch_times']:
            total_time = sum(self.performance_tracker['epoch_times'])
            avg_time = np.mean(self.performance_tracker['epoch_times'])
            logging.info(f'\n 训练总结:')
            logging.info(f'    总训练时间: {total_time / 3600:.2f} 小时')
            logging.info(f'    平均每轮时间: {avg_time:.1f} 秒')
            logging.info(f'    早停轮次: {self.enhanced_early_stopping.best_epoch}')
            if self.enhanced_early_stopping.early_stop:
                logging.info(f'    早停成功触发，避免过拟合')

        print(f'\n 模型保存目录: {self.checkpoint_dir}')
        vlog, tlog = self.best_recorder['val'], self.best_recorder['test']
        if 'epoch' in vlog and 'epoch' in tlog:
            print(f' Val  最佳: Epoch {vlog["epoch"]} | ' + 'loss: {:.4} | '.format(
                vlog.get("train_loss", 0)) + ' | '.join(
                ['{}: {:.4}'.format(m, vlog.get('val_' + m, 0)) for m in METRICS]))
            print(f' Test 最佳: Epoch {tlog["epoch"]} | ' + 'loss: {:.4} | '.format(
                tlog.get("train_loss", 0)) + ' | '.join(
                ['{}: {:.4}'.format(m, tlog.get('test_' + m, 0)) for m in METRICS]))
            print(' 最终指标: ' + ','.join(['{:.4}'.format(vlog.get('val_' + m, 0)) for m in METRICS]) +
                  f',E={vlog["epoch"]},TE={tlog["epoch"]},B4={tlog.get("test_BLEU_4", 0):.4}')

    def _prin_metrics(self, log, summary=False):
        if 'epoch' not in log:
            logging.info("  此次运行未产生最佳结果!")
            return
        logging.info(
            f' VAL  ||| Epoch: {log["epoch"]} ||| ' + 'train_loss: {:.4} ||| '.format(
                log.get("train_loss", 0)) + ' ||| '.join(
                ['{}: {:.4}'.format(m, log.get('val_' + m, 0)) for m in METRICS]))
        logging.info(
            f' TEST ||| Epoch: {log["epoch"]} ||| ' + 'train_loss: {:.4} ||| '.format(
                log.get("train_loss", 0)) + ' ||| '.join(
                ['{}: {:.4}'.format(m, log.get('test_' + m, 0)) for m in METRICS]))

        if not summary:
            if isinstance(log['epoch'], str):
                epoch_split = log['epoch'].split('-')
                e = int(epoch_split[0])
                if len(epoch_split) > 1:
                    it = int(epoch_split[1])
                    epoch = len(self.train_dataloader) * e + it
                else:
                    epoch = len(self.train_dataloader) * e
            else:
                epoch = int(log['epoch']) * len(self.train_dataloader)

            for m in METRICS:
                self.writer.add_scalar(f'val/{m}', log.get("val_" + m, 0), epoch)
                self.writer.add_scalar(f'test/{m}', log.get("test_" + m, 0), epoch)

    def _output_generation(self, predictions, gts, idxs, epoch, iters=0, split='val'):
        output = list()
        for idx, pre, gt in zip(idxs, predictions, gts):
            output.append({'filename': idx, 'prediction': pre, 'ground_truth': gt})

        json_file = f'Enc2Dec-{epoch}_{iters}_{split}_generated.json'
        output_filename = os.path.join(self.checkpoint_dir, json_file)
        with open(output_filename, 'w') as f:
            json.dump(output, f, ensure_ascii=False)

    def _print_epoch(self, log, epoch_time):
        """🔧 优化的轮次结果打印"""
        logging.info(f"\n{'-' * 80}")
        logging.info(f" Epoch [{log['epoch']}/{self.epochs}] 完成 (用时: {epoch_time:.1f}s)")
        logging.info(f" 保存目录: {self.checkpoint_dir}")
        logging.info(f"{'-' * 80}")
        self._prin_metrics(log)


class LanguageModelCriterion(torch.nn.Module):
    """用于处理错误情况的基础损失函数"""

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets, masks):
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        targets = targets.view(-1)
        masks = masks.view(-1)

        losses = self.loss_fn(logits, targets)
        masked_losses = losses * masks

        return masked_losses.view(batch_size, seq_len)


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.use_clip = getattr(args, 'use_clip', False)
        if self.use_clip:
            logging.info(" Trainer initialized with CLIP enhancement")

    def _train_epoch(self, epoch):
        """🔧 优化的训练轮次 - 增强错误处理和监控"""
        train_loss = 0
        self.model.train()

        print(f"\n{'-' * 60}")
        print(f"Epoch {epoch}/{self.epochs} - Training Phase")
        print(f"{'-' * 60}")

        t = tqdm(self.train_dataloader, ncols=120, leave=True,
                 desc=f"Epoch {epoch}")

        loss_details = {}
        batch_error_count = 0
        successful_batches = 0

        for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(t):
            try:
                images, reports_ids, reports_masks, labels = images.to(self.device), reports_ids.to(self.device), \
                    reports_masks.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(images, reports_ids, labels, mode='train')

                # 🔧 智能损失计算 - 多重错误处理
                loss = None
                try:
                    if self.use_clip and isinstance(outputs, tuple) and len(outputs) >= 8:
                        # CLIP增强模式
                        loss = self.criterion(outputs, reports_ids, reports_masks, labels, self.args)
                    else:
                        # 标准模式
                        if isinstance(outputs, tuple):
                            output = outputs[0]
                            vis_label = outputs[1] if len(outputs) > 1 else None
                            z_img = outputs[2] if len(outputs) > 2 else None
                            z_txt = outputs[3] if len(outputs) > 3 else None

                            loss = self.criterion(
                                output=output,
                                reports_ids=reports_ids,
                                reports_masks=reports_masks,
                                labels=labels,
                                vis_label=vis_label,
                                z_img=z_img,
                                z_txt=z_txt,
                                args=self.args
                            )
                        else:
                            loss = self.criterion(
                                output=outputs,
                                reports_ids=reports_ids,
                                reports_masks=reports_masks,
                                labels=labels,
                                args=self.args
                            )

                    # 处理返回的损失字典
                    if isinstance(loss, dict):
                        total_loss = loss.get('total_loss', loss.get('loss', 0))
                        for key, value in loss.items():
                            if key != 'total_loss' and isinstance(value, (int, float)):
                                if key not in loss_details:
                                    loss_details[key] = []
                                loss_details[key].append(value)
                        loss = total_loss

                except Exception as loss_error:
                    # 损失计算失败，使用基础损失
                    batch_error_count += 1
                    if isinstance(outputs, tuple):
                        main_output = outputs[0]
                    else:
                        main_output = outputs

                    criterion = LanguageModelCriterion()
                    loss = criterion(main_output[:, :-1], reports_ids[:, 1:], reports_masks[:, 1:]).mean()

                    if batch_error_count <= 3:  # 只打印前3个错误
                        logging.warning(f"  Batch {batch_idx} 损失计算失败，使用基础损失: {loss_error}")

                # 检查损失有效性
                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"  Batch {batch_idx} 损失无效，跳过")
                    continue

                successful_batches += 1
                train_loss += loss.item()

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()

                # 🔧 优化的进度条显示
                avg_loss = train_loss / successful_batches if successful_batches > 0 else 0
                progress_info = {
                    'loss': f'{loss.item():.4f}',
                    'avg': f'{avg_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.1e}'
                }

                # 添加CLIP损失详情
                if loss_details:
                    for key in ['clip_contrastive', 'global_align', 'local_align']:
                        if key in loss_details and loss_details[key]:
                            progress_info[key[:4]] = f'{np.mean(loss_details[key]):.3f}'

                t.set_postfix(progress_info)

            except Exception as e:
                batch_error_count += 1
                logging.error(f" Batch {batch_idx} 处理失败: {e}")
                continue

        # 🔧 训练总结
        avg_train_loss = train_loss / successful_batches if successful_batches > 0 else 0
        log = {'train_loss': avg_train_loss}

        if loss_details:
            for key, values in loss_details.items():
                if values:
                    log[f'train_{key}'] = np.mean(values)

        if batch_error_count > 0:
            logging.warning(f"  [{epoch}/{self.epochs}] {batch_error_count} 个批次出现错误 "
                            f"({successful_batches} 个成功)")

        # 验证和测试
        print(f"\n{'-' * 60}")
        print(f"Epoch {epoch}/{self.epochs} - Validation Phase")
        print(f"{'-' * 60}")
        ilog = self._test_step(epoch, 0, 'val')
        log.update(**ilog)

        print(f"\n{'-' * 60}")
        print(f"Epoch {epoch}/{self.epochs} - Testing Phase")
        print(f"{'-' * 60}")
        ilog = self._test_step(epoch, 0, 'test')
        log.update(**ilog)

        # 🔧 学习率调度 - 支持ReduceLROnPlateau
        if hasattr(self.lr_scheduler, 'step'):
            if hasattr(self.lr_scheduler, 'mode'):  # ReduceLROnPlateau
                current_val_score = log.get('val_' + self.args.monitor_metric, 0)
                self.lr_scheduler.step(current_val_score)
                if hasattr(self.lr_scheduler, '_last_lr'):
                    new_lr = self.lr_scheduler._last_lr[0] if self.lr_scheduler._last_lr else \
                    self.optimizer.param_groups[0]["lr"]
                    if new_lr != self.optimizer.param_groups[0]["lr"]:
                        logging.info(f" 学习率调整: {self.optimizer.param_groups[0]['lr']:.2e} → {new_lr:.2e}")
            else:  # 其他调度器
                self.lr_scheduler.step()

        print(f"\n Epoch {epoch} 所有阶段完成!")
        return log

    def _test_step(self, epoch, iters=0, mode='test'):
        """🔧 优化的测试步骤 - 增强稳定性"""
        ilog = {}
        self.model.eval()
        data_loader = self.val_dataloader if mode == 'val' else self.test_dataloader

        with torch.no_grad():
            val_gts, val_res, val_idxs = [], [], []
            failed_batches = 0

            # 单一测试进度条
            t = tqdm(data_loader,
                     desc=f'{mode.upper()}',
                     ncols=100,
                     leave=True)

            for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(t):
                try:
                    images, reports_ids, reports_masks, labels = images.to(self.device), reports_ids.to(self.device), \
                        reports_masks.to(self.device), labels.to(self.device)

                    # 推理模式
                    outputs = self.model(images, mode='sample')
                    if isinstance(outputs, tuple):
                        generated_reports = outputs[0]
                    else:
                        generated_reports = outputs

                    # 解码生成的报告和真实报告
                    reports = self.model.tokenizer.decode_batch(generated_reports.cpu().numpy())
                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                    val_res.extend(reports)
                    val_gts.extend(ground_truths)
                    val_idxs.extend(images_id)

                except Exception as e:
                    failed_batches += 1
                    if failed_batches <= 3:  # 只记录前3个错误
                        logging.warning(f"  {mode} batch {batch_idx} 失败: {e}")
                    continue

                # 更新进度条
                t.set_postfix({
                    'processed': f'{batch_idx + 1}/{len(data_loader)}',
                    'failed': failed_batches
                })

            # 🔧 计算评估指标
            if val_gts and val_res and len(val_gts) == len(val_res):
                try:
                    val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                               {i: [re] for i, re in enumerate(val_res)})
                    ilog.update(**{f'{mode}_' + k: v for k, v in val_met.items()})
                    self._output_generation(val_res, val_gts, val_idxs, epoch, iters, mode)

                    # 记录评估质量
                    logging.info(f" {mode.upper()} 评估完成: {len(val_gts)} 样本, {failed_batches} 失败")

                except Exception as e:
                    logging.error(f" {mode} 指标计算失败: {e}")
                    # 返回默认的零指标
                    for metric in METRICS:
                        ilog[f'{mode}_{metric}'] = 0.0
            else:
                logging.warning(f"  {mode} 评估数据不足或不匹配: gts={len(val_gts)}, res={len(val_res)}")
                # 返回默认的零指标
                for metric in METRICS:
                    ilog[f'{mode}_{metric}'] = 0.0

        return ilog

    def test_step(self, epoch, iters):
        """独立测试步骤"""
        ilog = {'epoch': f'{epoch}-{iters}', 'train_loss': 0.0}

        log = self._test_step(epoch, iters, 'val')
        ilog.update(**(log))

        log = self._test_step(epoch, iters, 'test')
        ilog.update(**(log))

        self._prin_metrics(ilog)