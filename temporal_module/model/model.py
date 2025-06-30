import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange

import sys
sys.path.append('/root/autodl-tmp/VQualA/')
sys.path.append('/root/autodl-tmp/VQualA/temporal_module/model')

from utils import save_metrics_to_file, setup_metrics_logging, Regress
from utils import calculate_plcc, calculate_srocc, correlation_loss

from slowfast import Slowfast
from datetime import datetime
import numpy as np
import os
import json
import scipy.stats
    

class Temporalmodule(pl.LightningModule):
    def __init__(self,
                 logs_path="/root/autodl-tmp/VQualA/temporal_module/logs",
                 checkpoints_path="/root/autodl-tmp/VQualA/temporal_module/checkpoints",
                 hidden_ratio=4,
                 lr=1e-4,
                 mse_weight=0.1,
                 feature_extractor='True'):
        super().__init__()

        self.lr = lr
        self.mse_weight = mse_weight

        self.model = Slowfast()
        self.regress = Regress(in_features=2304, hidden_ratio=hidden_ratio)

        if feature_extractor == 'True':
            for _, param in self.model.named_parameters():
                param.requires_grad = False
        else:
            for _, param in self.model.named_parameters():
                param.requires_grad = True

        #初始化指标记录
        self.log_file_path, self.metrics_history = setup_metrics_logging(logs_path, checkpoints_path)
        
        self.validation_predictions = []
        self.validation_targets = []

    def get_input(self, batch):
        video = batch["video"]
        score = batch["Temporal_MOS"]

        data = (video, score)
        return data
    
    def pack_pathway_output(self, video):
        """
        Args:
            frames (tensor): frames of images sampled from the video. [B, 8, N, C, H, W]
        Returns:
            frames_list (list): list of tensors
        """
        video = rearrange(video, 'b m n c h w -> b c (m n) h w')
        fast_pathway = video

        total_frames = fast_pathway.shape[2]
        slow_frames = total_frames // 4
    
        indices = torch.linspace(0, total_frames - 1, slow_frames).long().to(video.device)
    
        slow_pathway = torch.index_select(fast_pathway, 2, indices)
        frame_list = [slow_pathway, fast_pathway]

        return frame_list
    
    def forward(self, video):
        """
        Args:
            video: [B, 8, N, C, H, W]
        Returns:
            pred_scores: [B]
        """
        feature_list = self.pack_pathway_output(video)
        slow_feature, fast_feature = self.model(feature_list)
        #print(slow_feature.shape)
        #print(fast_feature.shape)
        feature = torch.cat((slow_feature, fast_feature), dim=-1) #[B, 2304]

        predicted_score = self.regress(feature).squeeze(-1) #[B]
        #print(predicted_score)

        return predicted_score
    
    def training_step(self, batch, batch_idx):
        video, target_score = self.get_input(batch)

        predicted_score = self.forward(video)

        corr_loss, plcc, srocc = correlation_loss(predicted_score, target_score)

        #辅助损失函数
        mse_loss = F.mse_loss(predicted_score, target_score.float())

        #总损失：相关性损失 + 小权重的MSE损失
        total_loss = corr_loss + self.mse_weight * mse_loss

        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_plcc', plcc, prog_bar=True)
        self.log('train_srocc', srocc, prog_bar=True)
        self.log('train_corr_avg', (plcc + srocc) / 2.0, prog_bar=True)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        video, target_score = self.get_input(batch)

        predicted_score = self.forward(video)

        corr_loss, plcc, srocc = correlation_loss(predicted_score, target_score)

        #辅助损失函数
        mse_loss = F.mse_loss(predicted_score, target_score.float())

        #总损失
        total_loss = corr_loss + self.mse_weight * mse_loss

        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_plcc', plcc, prog_bar=True)
        self.log('val_srocc', srocc, prog_bar=True)
        self.log('val_corr_avg', (plcc + srocc) / 2.0, prog_bar=True)

        self.validation_predictions.extend(predicted_score.detach().cpu().numpy())
        self.validation_targets.extend(target_score.detach().cpu().numpy())

        return {
            'val_loss': total_loss,
            'val_plcc': plcc,
            'val_srocc': srocc,
            'val_corr_avg': (plcc + srocc) / 2.0,
            'predicted': predicted_score,
            'target': target_score
        }
    
    
    def on_validation_epoch_start(self):
        """validation_epoch开始前清空上一个epoch的结果"""
        self.validation_predictions = []
        self.validation_targets = []

    def on_validation_epoch_end(self):
        """每次validation epoch结束后评估整个验证集的结果"""
        all_predictions = np.array(self.validation_predictions)
        all_targets = np.array(self.validation_targets)

        #相关性指标
        final_plcc = calculate_plcc(torch.tensor(all_predictions), torch.tensor(all_targets))
        final_srocc = calculate_srocc(torch.tensor(all_predictions), torch.tensor(all_targets))
        final_corr_avg = (final_plcc + final_srocc) / 2.0

        results = {
            'epoch': self.current_epoch,
            'num_samples': len(all_predictions),
            'plcc': final_plcc,
            'srocc': final_srocc,
            'corr_avg': final_corr_avg
        }

        self.metrics_history['val'].append(results)

        #保存结果
        save_metrics_to_file(self.metrics_history, self.log_file_path)

        #打印结果
        print(f"\n=== 验证集第{self.current_epoch + 1}轮评估结果 ===")
        print(f"当前epoch: {self.current_epoch + 1}")
        print(f"PLCC: {final_plcc:.4f}")
        print(f"SROCC: {final_srocc:.4f}")
        print(f"Corr_AVG: {final_corr_avg:.4f}")

        return results
    
    def test_step(self, batch, batch_idx):
        video, target_score = self.get_input(batch)

        predicted_score = self.forward(video)

        _, plcc, srocc = correlation_loss(predicted_score, target_score)

        self.log('test_plcc', plcc)
        self.log('test_srocc', srocc)
        self.log('test_corr_avg', (plcc + srocc) / 2.0)

        return {
            'test_plcc': plcc,
            'test_srocc': srocc,
            'test_corr_avg': (plcc + srocc) / 2.0
        }
    
    def evaluate_dataset(self,
                         dataloader,
                         results_file="final_evaluation.json",
                         results_path="/root/autodl-tmp/VQualA/temporal_module/logs"):
        """在训练结束后用于评估数据集"""
        print("\n=== 开始对数据集进行评估 ===")

        self.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                #移动数据到设备
                video = batch["video"].to(self.device)
                target_score = batch["Temporal_MOS"].to(self.device)

                predicted_score = self.forward(video)

                all_predictions.extend(predicted_score.cpu().numpy())
                all_targets.extend(target_score.cpu().numpy())

                if (batch_idx + 1) % 50 == 0:
                    print(f"已处理 {batch_idx + 1} 个batch")

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        #相关性指标
        final_plcc = calculate_plcc(torch.tensor(all_predictions), torch.tensor(all_targets))
        final_srocc = calculate_srocc(torch.tensor(all_predictions), torch.tensor(all_targets))
        final_corr_avg = (final_plcc + final_srocc) / 2.0

        results = {
            'num_samples': len(all_predictions),
            'plcc': final_plcc,
            'srocc': final_srocc,
            'corr_avg': final_corr_avg
        }

        #保存结果
        results_path = os.path.join(results_path, results_file)

        save_metrics_to_file(results, results_path)

        print(f"\n=== 数据集评估结果 ===")
        print(f"总样本数： {len(all_predictions)}")
        print(f"PLCC: {final_plcc:.4f}")
        print(f"SROCC: {final_srocc:.4f}")
        print(f"Corr_AVG: {final_corr_avg:.4f}")

        return results
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.01
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_corr_avg',
                'mode': 'max',
                'interval': 'epoch'
            }
        }