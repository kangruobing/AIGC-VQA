import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pytorch_lightning as pl
from einops import rearrange

import sys
sys.path.append('/root/autodl-tmp/VQualA')
sys.path.append('/root/autodl-tmp/VQualA/aesthetic_module/model')

from utils import save_metrics_to_file, setup_metrics_logging, Regress
from utils import calculate_plcc, calculate_srocc, combined_loss

from conv_backbone import convnext_3d_tiny, convnext_3d_small
from datetime import datetime
import numpy as np
import os
import json
import scipy.stats
    

class Aestheticmodule(pl.LightningModule):
    def __init__(self,
                 logs_path="/root/autodl-tmp/VQualA/aesthetic_module/logs",
                 checkpoints_path="/root/autodl-tmp/VQualA/aesthetic_module/checkpoints",
                 model_size="tiny",
                 lr=1e-4,
                 alpha=1.0,
                 beta=0.3,
                 weight_decay=0.01):
        super().__init__()

        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.weight_decay = weight_decay

        if model_size == "tiny":
            self.model = convnext_3d_tiny(pretrained=True, in_22k=True)
        else:
            self.model = convnext_3d_small(pretrained=True, in_22k=True)

        self.regress = Regress(768)

        #初始化指标记录
        self.log_file_path, self.metrics_history = setup_metrics_logging(logs_path, checkpoints_path)
               
        self.validation_predictions = []
        self.validation_targets = []
   
    def get_input(self, batch):
        video = batch["video"] #[B, M, N, C, H, W]
        score = batch["Aesthetic_MOS"]

        data = (video, score)
        return data
    
    def forward(self, video):
        """
        Args:
            video: [B, M, N, C, H, W] - batch of video frames
        Returns:
            pred_scores: [B] - predicted aesthetic scores
        """
        B, M, N, C, H, W = video.shape
        
        video = rearrange(video, 'b m n c h w -> (b m) c n h w') #[B*M, C, N, H, W]
        
        video_features = self.model.forward_features(video, return_spatial=False) #[B*M, 768]
        
        video_features = rearrange(video_features, '(b m) f -> b m f', b=B, m=M) #[B, M, 768]
        video_features = torch.mean(video_features, dim=1)
        
        predicted_scores = self.regress(video_features).squeeze(-1) #[B]
        #print(predicted_scores)

        return predicted_scores

    def training_step(self, batch, batch_idx):
        video, target_score = self.get_input(batch)

        predicted_score = self.forward(video)

        plcc = calculate_plcc(predicted_score, target_score)
        srocc = calculate_srocc(predicted_score, target_score)
        loss = combined_loss(predicted_score, target_score, self.alpha, self.beta)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_plcc', plcc, prog_bar=True)
        self.log('train_srocc', srocc, prog_bar=True)
        self.log('train_corr_avg', (plcc + srocc) / 2.0, prog_bar=True)
    
        return loss
    
    def validation_step(self, batch, batch_idx):
        video, target_score = self.get_input(batch)

        predicted_score = self.forward(video)

        plcc = calculate_plcc(predicted_score, target_score)
        srocc = calculate_srocc(predicted_score, target_score)
        loss = combined_loss(predicted_score, target_score, self.alpha, self.beta)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_plcc', plcc, prog_bar=True)
        self.log('val_srocc', srocc, prog_bar=True)
        self.log('val_corr_avg', (plcc + srocc) / 2.0, prog_bar=True)

        self.validation_predictions.extend(predicted_score.detach().cpu().numpy())
        self.validation_targets.extend(target_score.detach().cpu().numpy())

        return {
            'val_loss': loss,
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
            'epoch': self.current_epoch + 1,
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

        plcc = calculate_plcc(predicted_score, target_score)
        srocc = calculate_srocc(predicted_score, target_score)
        loss = combined_loss(predicted_score, target_score, self.alpha, self.beta)

        self.log('test_loss', loss)
        self.log('test_plcc', plcc)
        self.log('test_srocc', srocc)
        self.log('test_corr_avg', (plcc + srocc) / 2.0)

        return {
            'test_loss': loss,
            'test_plcc': plcc,
            'test_srocc': srocc,
            'test_corr_avg': (plcc + srocc) / 2.0
        }
    
    def evaluate_dataset(self,
                         dataloader,
                         results_file="final_evaluation.json",
                         results_path="/root/autodl-tmp/VQualA/aesthetic_module/logs"):
        """在训练结束后用于评估数据集"""
        print("\n=== 开始对数据集进行评估 ===")

        self.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                #移动数据到设备
                video = batch["video"].to(self.device)
                target_score = batch["Aesthetic_MOS"].to(self.device)

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
            weight_decay=self.weight_decay
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