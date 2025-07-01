import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from einops import rearrange
from collections import OrderedDict

import sys
sys.path.append('/root/autodl-tmp/VQualA')

from utils import save_metrics_to_file, setup_metrics_logging, VQARegress
from utils import calculate_plcc, calculate_srocc, combined_loss, performance_fit

import scipy.stats
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

from alignment_module.model.model import BLIP, BLIPPooler
from traditional_module.model.model import Identity
from aesthetic_module.model.conv_backbone import convnext_3d_tiny, convnext_3d_small
from temporal_module.model.slowfast import Slowfast


class AIGCVQA(pl.LightningModule):
    def __init__(self,
                 split_id,
                 output_dir="/root/autodl-tmp/VQualA",
                 bert_config="/root/autodl-tmp/VQualA/alignment_module/med_config.json",
                 imagereward_path="/root/autodl-tmp/VQualA/alignment_module/imagereward.pth",
                 align_frozen_ratio=0.7,
                 traditional_path="/root/autodl-tmp/VQualA/traditional_module/model/Swin_b_384_in22k_SlowFast_Fast_LSVQ.pth",
                 traditional_freeze_strategy="partial",
                 traditional_freeze_ratio=0.3,
                 aesmodule_size="tiny",
                 freeze_temporalmodule="False",
                 lr=1e-4,
                 alpha=1.0,
                 beta=0.3,
                 weight_decay=0.05,
                 dropout=0.2):
        super().__init__()

        self.lr = lr
        self.plcc_weight = alpha
        self.rank_weight = beta
        self.weight_decay = weight_decay

        #初始化一致性评价模块
        self.align_frozen_ratio = align_frozen_ratio
        self.alignmentmodule = BLIP(bert_config)
        
        self.load_imagereward_weight(imagereward_path)

        for name, param in self.alignmentmodule.named_parameters():
            if ("text_encoder" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.freeze_parameters()

        alignment_size = self.alignmentmodule.text_encoder.config.hidden_size
        self.blippooler = BLIPPooler(alignment_size)

        #初始化传统评价模块
        self.traditionalmodule = timm.create_model('swin_base_patch4_window12_384_in22k', pretrained=False)
        self.load_rqvqa_weights(traditional_path)
        self.freeze_swin_parameters(traditional_freeze_strategy, traditional_freeze_ratio)
        self.traditionalmodule.head = Identity()

        traditional_size = 1024

        #初始化美学评价模块
        if aesmodule_size == "tiny":
            self.aestheticmodule = convnext_3d_tiny(pretrained=True, in_22k=True)
        else:
            self.aestheticmodule = convnext_3d_small(pretrained=True, in_22k=True)
        
        aesthetic_size = 768

        #初始化时序评价模块
        self.temporalmodule = Slowfast()
        temporal_size = 2304

        if freeze_temporalmodule == "True":
            for _, param in self.temporalmodule.named_parameters():
                param.requires_grad = False
        else:
            for _, param in self.temporalmodule.named_parameters():
                param.requires_grad = True
        
        #初始化回归头
        self.regress = VQARegress(in_features=(alignment_size+traditional_size+aesthetic_size+temporal_size), dropout=dropout)
        
        #初始化指标记录
        logs_path = os.path.join(output_dir, f"split_{split_id+1}")
        checkpoints_path = os.path.join(output_dir, f"split_{split_id+1}")

        self.log_file_path, self.metrics_history = setup_metrics_logging(logs_path, checkpoints_path)

        self.validation_predictions = []
        self.validation_targets = []

    def load_imagereward_weight(self, weight_path):
        """加载imagereward权重"""
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            filtered_dict = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith('blip.'):
                    new_key = key.replace('blip.', '')
                    filtered_dict[new_key] = value
                    
            missing, unexpected = self.alignmentmodule.load_state_dict(filtered_dict, strict=False)

            print(f"\n=== alignment_module权重加载结果 ===")
            print(f"成功加载: {len(filtered_dict) - len(missing)} 个权重")
            print(f"缺失权重: {len(missing)} 个")
            print(f"意外权重: {len(unexpected)} 个")

        except Exception as e:
            print(f"权重加载失败: {e}")

    def freeze_parameters(self):
        """冻结alignmodule.text_encoder固定层"""
        text_fix_num = "layer.{}".format(int(12 * self.align_frozen_ratio))

        for name, param in self.alignmentmodule.text_encoder.named_parameters():
            param.requires_grad = False
            if text_fix_num in name:
                break

    def load_rqvqa_weights(self, weight_path):
        """加载RQVQA权重"""
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            filtered_dict = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith('feature_extraction.'):
                    new_key = key.replace('feature_extraction.', '')
                    filtered_dict[new_key] = value

            missing, unexpected = self.traditionalmodule.load_state_dict(filtered_dict, strict=False)

            print(f"\n=== traditional_module权重加载结果 ===")
            print(f"成功加载: {len(filtered_dict) - len(missing)} 个权重")
            print(f"缺失权重: {len(missing)} 个")
            print(f"意外权重: {len(unexpected)} 个")

        except Exception as e:
            print(f"权重加载失败: {e}")

    def freeze_swin_parameters(self, freeze_strategy="partial", freeze_ratio=0.5):
        """
        冻结SwinTransformer的不同层

        Args:
            freeze_strategy: str, 冻结策略
                - "none": 不冻结任何层
                - "embed": 只冻结embedding层
                - "partial": 按比例冻结
                - "backbone": 冻结所有层
            freeze_ratio: float, 当strategy为"partial"时使用, 冻结的层数比例
        """

        total_params = sum(p.numel() for p in self.traditionalmodule.parameters())
        frozen_params = 0
        trainable_params = 0

        if freeze_strategy == "none":
            for param in self.traditionalmodule.parameters():
                param.requires_grad = True
                trainable_params += param.numel()
            print("所有Swin参数可训练")

        elif freeze_strategy == "embed":
            for name, param in self.traditionalmodule.named_parameters():
                #只冻结embedding层
                if any(keyword in name for keyword in ['patch_embed', 'pos_embed', 'pos_drop']):
                    param.requires_grad = False
                    frozen_params += param.numel()
                else:
                    param.requires_grad = True
                    trainable_params += param.numel()
            print("冻结embedding层")

        elif freeze_strategy == "partial":
            #按比例冻结层数
            total_layers = 4
            freeze_layers = int(total_layers * freeze_ratio)

            frozen_layer_names = []

            for name, param in self.traditionalmodule.named_parameters():
                should_freeze = False

                if any(keyword in name for keyword in ['patch_embed', 'pos_embed', 'pos_drop']):
                    should_freeze = True

                for i in range(freeze_layers):
                    if f'layers.{i}.' in name:
                        should_freeze = True
                        if f'layers.{i}' not in frozen_layer_names:
                            frozen_layer_names.append(f'layers.{i}')
                        break
                
                if should_freeze:
                    param.requires_grad = False
                    frozen_params += param.numel()
                else:
                    param.requires_grad = True
                    trainable_params += param.numel()

            print(f"冻结策略: 比例冻结 - 冻结比例: {freeze_ratio:.1%}")
            print(f"冻结层: embedding + {frozen_layer_names}")

        elif freeze_strategy == 'backbone':
            #冻结整个SwinTransformer
            for param in self.traditionalmodule.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            print("整个SwinTransformer被冻结")

        else:
            raise ValueError(f"不支持的冻结策略: {freeze_strategy}")
        
        if freeze_strategy != "none":
            freeze_percentage = (frozen_params / total_params) * 100
            trainable_percentage = (trainable_params / total_params) * 100

            print(f"参数统计")
            print(f"总参数: {total_params:,} ({total_params/1e6:.1f}M)")
            print(f"冻结参数: {frozen_params:,} ({frozen_params/1e6:.1f}M, {freeze_percentage:.1f}%)")
            print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.1f}M, {trainable_percentage:.1f}%)")
        else:
            print(f"参数统计: 总参数 {total_params:,} ({total_params/1e6:.1f}M) 全部可训练")
    
    def get_input(self, batch):
        image = batch["image"]
        image_traditional = batch["image_traditional"]
        video = batch["video"]
        prompt = batch["prompt"]
        score = batch["Overall_MOS"]

        return (image, image_traditional, video, prompt, score)
    
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
    
    def forward(self, image, image_traditional, video, prompt):
        #对齐特征向量
        B, N, C, H, W = image.shape

        align_image = rearrange(image, 'b n c h w -> (b n) c h w')

        expanded_prompts = []
        for i, p in enumerate(prompt):
            expanded_prompts.extend([p] * N)

        feature = self.alignmentmodule(align_image, expanded_prompts)
        pooled_feature = self.blippooler(feature) #[B*N, hidden_size]
        pooled_feature = rearrange(pooled_feature, '(b n) h -> b n h', b=B, n=N)
        ali_feature = torch.mean(pooled_feature, dim=1) #[B, h]

        #美学特征向量
        B, M, N, C, H, W = video.shape
        
        aes_video = rearrange(video, 'b m n c h w -> (b m) c n h w') #[B*M, C, N, H, W]
        
        aes_feature = self.aestheticmodule.forward_features(aes_video, return_spatial=False) #[B*M, 768]
        
        aes_feature = rearrange(aes_feature, '(b m) f -> b m f', b=B, m=M) #[B, M, 768]
        aes_feature = torch.mean(aes_feature, dim=1)

        #传统特征向量
        B, N, C, H, W = image_traditional.shape

        tra_image = rearrange(image_traditional, 'b n c h w -> (b n) c h w')
        tra_feature = self.traditionalmodule(tra_image) #[B*N, 1024]
        tra_feature = rearrange(tra_feature, '(b n) f -> b n f', b=B, n=N)
        tra_feature = torch.mean(tra_feature, dim=1) #[B, 1024]

        #时序特征向量
        feature_list = self.pack_pathway_output(video)
        slow_feature, fast_feature = self.temporalmodule(feature_list)
        tem_feature = torch.cat((slow_feature, fast_feature), dim=1) #[B, 2304]

        #score
        feature = torch.cat((ali_feature, aes_feature, tra_feature, tem_feature), dim=1)
        predicted_score = self.regress(feature).squeeze(-1)

        return predicted_score
    
    def training_step(self, batch, batch_idx):
        image, image_traditional, video, prompt, target_score = self.get_input(batch)

        predicted_score = self.forward(image, image_traditional, video, prompt)

        plcc = calculate_plcc(predicted_score, target_score)
        srocc = calculate_srocc(predicted_score, target_score)
        loss = combined_loss(predicted_score, target_score, self.plcc_weight, self.rank_weight)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_plcc', plcc, prog_bar=True)
        self.log('train_srocc', srocc, prog_bar=True)
        self.log('train_corr_avg', (plcc + srocc) / 2.0, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        image, image_traditional, video, prompt, target_score = self.get_input(batch)

        predicted_score = self.forward(image, image_traditional, video, prompt)

        plcc, srocc = performance_fit(target_score, predicted_score)
        loss = combined_loss(predicted_score, target_score, self.plcc_weight, self.rank_weight)

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
        final_plcc, final_srocc = performance_fit(torch.tensor(all_targets), torch.tensor(all_predictions))
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
        image, image_traditional, video, prompt, target_score = self.get_input(batch)

        predicted_score = self.forward(image, image_traditional, video, prompt)

        plcc = calculate_plcc(predicted_score, target_score)
        srocc = calculate_srocc(predicted_score, target_score)
        loss = combined_loss(predicted_score, target_score, self.plcc_weight, self.rank_weight)

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
                         model_weights_path=None,
                         results_file="final_evaluation.csv",
                         results_path="/root/autodl-tmp/VQualA/logs"):
        """在训练结束后对验证/测试集进行评估"""
        print("\n=== 开始对数据集进行评估 ===")

        if model_weights_path is not None:
            if os.path.exists(model_weights_path):
                print(f"正在加载模型权重: {model_weights_path}")
                checkpoint = torch.load(model_weights_path, map_location='cpu')

                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                self.load_state_dict(state_dict, strict=False)
                print("模型权重加载完成")
            else:
                print(f"模型权重不存在: {model_weights_path}")
                print("使用当前模型权重进行评估")

        self.eval()
        all_predictions = []
        all_targets = []
        all_video_names = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                #移动数据到设备
                image = batch["image"].to(self.device)
                image_traditional = batch["image_traditional"].to(self.device)
                video = batch["video"].to(self.device)
                prompt = batch["prompt"]
                target_score = batch["Overall_MOS"].to(self.device)
                video_names = batch["video_name"]

                predicted_score = self.forward(image, image_traditional, video, prompt)

                all_predictions.extend(predicted_score.cpu().numpy())
                all_targets.extend(target_score.cpu().numpy())
                all_video_names.extend(video_names)

                if (batch_idx + 1) % 50 == 0:
                    print(f"已处理 {batch_idx + 1} 个batch")

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        #相关性指标
        final_plcc = calculate_plcc(torch.tensor(all_predictions), torch.tensor(all_targets))
        final_srocc = calculate_srocc(torch.tensor(all_predictions), torch.tensor(all_targets))
        final_corr_avg = (final_plcc + final_srocc) / 2.0

        detailed_results = pd.DataFrame({
            'video_name': all_video_names,
            'Overall_MOS': all_predictions
        })

        os.makedirs(results_path, exist_ok=True)

        detailed_csv_path = os.path.join(results_path, results_file)
        detailed_results.to_csv(detailed_csv_path, index=False)

        summary_file = results_file.replace('.csv', '_summary.json')
        summary_path = os.path.join(results_path, summary_file)

        summary_results = {
            'evaluation_info': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%H:%S'),
                'model_weights_path': model_weights_path,
                'num_samples': len(all_predictions)
            },
            'metrics': {
                'plcc': final_plcc,
                'srocc': final_srocc,
                'corr_avg': final_corr_avg
            }
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)

        print(f"\n=== 数据集评估结果 ===")
        print(f"总样本数： {len(all_predictions)}")
        print(f"PLCC: {final_plcc:.4f}")
        print(f"SROCC: {final_srocc:.4f}")
        print(f"Corr_AVG: {final_corr_avg:.4f}")

        return summary_results, detailed_results
    
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