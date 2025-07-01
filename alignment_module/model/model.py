import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('/root/autodl-tmp/VQualA')
sys.path.append('/root/autodl-tmp/VQualA/alignment_module/model')

from utils import save_metrics_to_file, setup_metrics_logging, Regress
from utils import calculate_plcc, calculate_srocc, combined_loss, performance_fit

from vit import VisionTransformer
from med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict

import scipy.stats
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/VQualA/alignment_module/bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width


class BLIP(nn.Module):
    def __init__(self,
                 config='/root/autodl-tmp/VQualA/alignment_module/med_config.json',
                 image_size=224,
                 vit='large',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0):
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        Bert_config = BertConfig.from_json_file(config)
        Bert_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=Bert_config, add_pooling_layer=False)

    def forward(self, image, prompt):
        text = self.tokenizer(prompt, 
                              return_tensors="pt",
                              padding=True,
                              truncation=True,
                              max_length=77).to(image.device)

        image = self.visual_encoder(image) #[batch_size, seq_length, vision_width]
        image_mask = torch.ones(image.size()[:-1], dtype=torch.long).to(image.device)

        text.input_ids[:,0] = self.tokenizer.enc_token_id

        output = self.text_encoder(text.input_ids,
                                   attention_mask = text.attention_mask,
                                   encoder_hidden_states = image,
                                   encoder_attention_mask = image_mask,
                                   return_dict = True)
        
        return output.last_hidden_state


class BLIPPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.act = nn.Tanh()

    def forward(self, last_hidden_state):
        first_token_tensor = last_hidden_state[:, 0, :]
        pooled_output = self.fc(first_token_tensor)
        pooled_output = self.act(pooled_output)

        return pooled_output


class Alignmodule(pl.LightningModule):
    def __init__(self,
                 bert_config, 
                 image_size,
                 vit_weights,
                 bert_weights,
                 split_id,
                 output_dir="/root/autodl-tmp/VQualA/alignment_module",
                 weight_type="imagereward",
                 imagereward_path="/root/autodl-tmp/VQualA/alignment_module/imagereward.pth",
                 frozen_ratio=0.7,
                 lr=3e-5,
                 alpha=1.0,
                 beta=0.3,
                 weight_decay=0.05):
        super().__init__()

        self.frozen_ratio=frozen_ratio
        self.lr = lr
        self.plcc_weight = alpha
        self.rank_weight = beta
        self.weight_decay = weight_decay

        self.blip = BLIP(bert_config, image_size, vit='large')

        if weight_type not in ["imagereward", "standard"]:
            raise ValueError(f"权重加载方式不存在!")
        
        if weight_type == "standard":
            state_dict_vision = torch.load(vit_weights, map_location='cpu')
            state_dict_text = torch.load(bert_weights, map_location='cpu')
            self.blip.visual_encoder.load_state_dict(state_dict_vision["model"], strict=False)
            self.blip.text_encoder.load_state_dict(state_dict_text, strict=False)
        else:
            self.load_imagereward_weight(imagereward_path)

        for name, param in self.blip.named_parameters():
            if ("text_encoder" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.freeze_parameters()

        hidden_size = self.blip.text_encoder.config.hidden_size

        self.pooler = BLIPPooler(hidden_size)

        self.regress = Regress(in_features=hidden_size)

        #初始化指标记录
        logs_path = os.path.join(output_dir, f"split_{split_id+1}")
        checkpoints_path = os.path.join(output_dir, f"split_{split_id+1}")

        self.log_file_path, self.metrics_history = setup_metrics_logging(logs_path, checkpoints_path)

        #存储预测结果
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

            #过滤权重
            filtered_dict = OrderedDict()
            processed_keys = []
            skipped_keys = []

            for key, value in state_dict.items():
                if key.startswith('blip.'):
                    new_key = key.replace('blip.', '')
                    filtered_dict[new_key] = value
                    processed_keys.append(f"{key} -> {new_key}")
                else:
                    skipped_keys.append(key)

            print(f"处理了 {len(filtered_dict)} 个BLIP权重")
            print(f"跳过了 {len(skipped_keys)} 个非BLIP权重")

            our_model_keys = set(self.blip.state_dict().keys())
            filtered_keys = set(filtered_dict.keys())

            print(f"我们的模型有 {len(our_model_keys)} 个权重")
            print(f"ImageReward提供了 {len(filtered_keys)} 个匹配权重")

            # 找出缺失和多余的权重
            missing_in_imagereward = our_model_keys - filtered_keys
            extra_in_imagereward = filtered_keys - our_model_keys
        
            print(f"\n=== 权重匹配分析 ===")
            print(f"我们模型中缺失的权重 ({len(missing_in_imagereward)} 个):")
            for i, key in enumerate(sorted(missing_in_imagereward)):
                print(f"  {i+1:2d}. {key}")
        
            if extra_in_imagereward:
                print(f"\nImageReward中多余的权重 ({len(extra_in_imagereward)} 个):")
                for i, key in enumerate(sorted(extra_in_imagereward)):
                    print(f"  {i+1:2d}. {key}")

            # 加载权重
            missing, unexpected = self.blip.load_state_dict(filtered_dict, strict=False)
        
            print(f"\n=== 权重加载结果 ===")
            print(f"成功加载: {len(filtered_dict) - len(missing)} 个权重")
            print(f"缺失权重: {len(missing)} 个")
            print(f"意外权重: {len(unexpected)} 个")
        
            if missing:
                print(f"\n详细缺失权重列表:")
                for i, key in enumerate(missing):
                    print(f"  {i+1:2d}. {key}")

        except Exception as e:
            print(f"权重加载失败: {e}")

    def freeze_parameters(self):
        """冻结固定层"""
        text_fix_num = "layer.{}".format(int(12 * self.frozen_ratio))

        for name, param in self.blip.text_encoder.named_parameters():
            param.requires_grad = False
            if text_fix_num in name:
                break
         
    def get_input(self, batch):
        image = batch["image"] #tensor: B, N, C, H, W
        prompt = batch["prompt"] #list
        score = batch["Alignment_MOS"] #tensor: B

        data = (image, prompt, score)

        return data
    
    def forward(self, image, prompt):
        """
        Args:
            image: [B, N, C, H, W] - batch of video frames
            prompt: List[str] - batch of text prompts
        Returns:
            pred_scores: [B] - predicted alignment scores
        """
        B, N, C, H, W = image.shape
        image = image.view(B * N, C, H, W)

        #为每一帧重复对应的prompt
        expanded_prompts = []
        for i, p in enumerate(prompt):
            expanded_prompts.extend([p] * N)

        features = self.blip(image, expanded_prompts) #[B*N, seq_length, hidden_size]
        pooled_features = self.pooler(features) #[B*N, hidden_size]
        pooled_features = pooled_features.view(B, N, -1) #[B, N, hidden_size]
        video_features = pooled_features.mean(dim=1) #video_features: [B, hidden_size]

        #预测一致性分数
        predicted_scores = self.regress(video_features).squeeze(-1)

        return predicted_scores
    
    def training_step(self, batch, batch_idx):
        image, prompt, target_score = self.get_input(batch)

        predicted_score = self.forward(image, prompt)

        plcc = calculate_plcc(predicted_score, target_score)
        srocc = calculate_srocc(predicted_score, target_score)
        loss = combined_loss(predicted_score, target_score, self.plcc_weight, self.rank_weight)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_plcc', plcc, prog_bar=True)
        self.log('train_srocc', srocc, prog_bar=True)
        self.log('train_corr_avg', (plcc + srocc) / 2.0, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        image, prompt, target_score = self.get_input(batch)

        predicted_score = self.forward(image, prompt)

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
        """validation epoch开始前清空上一个epoch的结果"""
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

        save_metrics_to_file(self.metrics_history, self.log_file_path)

        print(f"\n=== 验证集第{self.current_epoch + 1}轮评估结果 ===")
        print(f"当前epoch: {self.current_epoch + 1}")
        print(f"PLCC: {final_plcc:.4f}")
        print(f"SROCC: {final_srocc:.4f}")
        print(f"Corr_AVG: {final_corr_avg:.4f}")

        return results
   
    def test_step(self, batch, batch_idx):
        image, prompt, target_score = self.get_input(batch)

        predicted_score = self.forward(image, prompt)

        plcc, srocc = performance_fit(target_score, predicted_score)
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
                         results_path="/root/autodl-tmp/VQualA/alignment_module/logs"):
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
        all_prompts = []
        all_video_names = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                #移动数据到设备
                image = batch["image"].to(self.device)
                prompt = batch["prompt"]
                target_score = batch["Alignment_MOS"].to(self.device)
                video_names = batch["video_name"]

                predicted_score = self.forward(image, prompt)

                all_predictions.extend(predicted_score.cpu().numpy())
                all_targets.extend(target_score.cpu().numpy())
                all_prompts.extend(prompt)
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
            'Alignment_MOS': all_predictions
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
                'mode':'max',
                'interval': 'epoch'
            }
        }