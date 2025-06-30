import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import os
import json
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from datetime import datetime


def setup_metrics_logging(logs_path: str, checkpoints_path: str):
    """
    设置指标记录

    Args:
        logs_path: 日志路径
        checkpoints_path: 权重路径
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(logs_path, f"training_metrics_{timestamp}.json")
    metrics_history = {
        'val': []
    }

    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    return log_file_path, metrics_history

def save_metrics_to_file(metrics_history: dict, log_file_path: str):
    """
    保存指标到JSON文件

    Args:
        metrics_history: 指标历史字典
        log_file_path: 完整的日志文件路径
    """
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_history, f, indent=2, ensure_ascii=False)
        print(f"指标已保存到: {log_file_path}")
    except Exception as e:
        print(f"保存指标文件时出错: {e}")

def calculate_plcc(pred: Tensor, target: Tensor):
    """计算Pearson线性相关系数"""
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    if len(pred) < 2:
        return 0.0
        
    plcc = stats.pearsonr(pred, target)[0]
    return plcc if not np.isnan(plcc) else 0.0

def calculate_srocc(pred: Tensor, target: Tensor):
    """计算Spearman等级相关系数"""
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    if len(pred) < 2:
        return 0.0
        
    srocc = stats.spearmanr(pred, target)[0]
    return srocc if not np.isnan(srocc) else 0.0

def plcc_loss(y_pred: Tensor, y: Tensor) -> Tensor:
    """
    PLCC损失函数

    Args:
        y_pred: [B] or [B, 1]
        y: [B] or [B, 1]

    Returns:
        loss
    """
    if y_pred.dim() > 1:
        y_pred = y_pred.squeeze(-1)
    if y.dim() > 1:
        y = y.squeeze(-1)

    #检查batch_size
    if y_pred.shape[0] < 2:
        return F.mse_loss(y_pred, y.float())
    
    #标准化预测值
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    if sigma_hat < 1e-8:
        y_pred_norm = y_pred
    else:
        y_pred_norm = (y_pred - m_hat) / (sigma_hat + 1e-8)

    #标准化真实值
    sigma, m = torch.std_mean(y.float(), unbiased=False)
    if sigma < 1e-8:
        y_norm = y.float()
    else:
        y_norm = (y.float() - m) / (sigma + 1e-8)

    loss0 = F.mse_loss(y_pred_norm, y_norm) / 4

    #计算相关系数
    rho = torch.mean(y_pred_norm * y_norm)
    rho = torch.clamp(rho, -0.99, 0.99)

    loss1 = F.mse_loss(rho * y_pred_norm, y_norm) / 4

    total_loss = (loss0 + loss1) / 2

    return total_loss

def rank_loss(y_pred: Tensor, y: Tensor) -> Tensor:
    """
    排序损失函数

    Args:
        y_pred: [B] or [B, 1]
        y: [B] or [B, 1]

    Returns:
        loss
    """
    if y_pred.dim() > 1:
        y_pred = y_pred.squeeze(-1)
    if y.dim() > 1:
        y = y.squeeze()
    
    #检查batch_size
    if y_pred.shape[0] < 2:
        return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
    
    # 计算排序损失
    # y_pred: [B] -> [B, 1] -> [B, B]
    # y_pred.t(): [1, B] -> [B, B]
    pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)  # [B, B]
    target_diff = y.float().unsqueeze(0) - y.float().unsqueeze(1)  # [B, B]
    target_sign = torch.sign(target_diff)
    ranking_loss = F.relu(pred_diff * target_sign)

    #归一化
    scale = 1 + torch.max(ranking_loss)
    if scale < 1e-8:
        return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
    
    #平均损失
    total_loss = torch.sum(ranking_loss) / (y_pred.shape[0] * (y_pred.shape[0] - 1)) / scale

    return total_loss.float()

def combined_loss(y_pred: Tensor, y: Tensor, alpha: float = 1.0, beta: float = 0.3) -> Tensor:
    """组合损失函数"""

    p_loss = plcc_loss(y_pred, y)
    r_loss = rank_loss(y_pred, y)

    total_loss = alpha * p_loss + beta * r_loss

    return total_loss

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)

    return y_output_logistic

def performance_fit(y_label: Tensor, y_output: Tensor):
    y_label = y_label.detach().cpu().numpy()
    y_output = y_output.detach().cpu().numpy()

    y_output_logistic = fit_function(y_label, y_output)

    plcc = stats.pearsonr(y_output_logistic, y_label)[0]
    srocc = stats.spearmanr(y_output, y_label)[0]

    return plcc, srocc


class Regress(nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.2):
        super().__init__()

        self.in_features = in_features        

        self.layers = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: Tensor):
        output = self.layers(x)

        #score = 5.0 * torch.sigmoid(output)

        return output
    

class VQARegress(nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.2):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(512, 128),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: Tensor):
        output = self.layers(x)

        #score = 5.0 * torch.sigmoid(output)

        return output
