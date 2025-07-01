import os
import argparse
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model.model import Alignmodule

import sys
sys.path.append('/root/autodl-tmp/VQualA')

import torch
from utils import set_random_seed
from torch.utils.data import DataLoader
from dataset import DatasetImage
import json


def train_single_split(args, split_id):
    """训练单个数据划分"""
    print(f"\n{'='*50}")
    print(f"开始训练第 {split_id+1} 个划分")
    print(f"{'='*50}")

    current_seed = args.seed + split_id
    set_random_seed(current_seed)

    #输出目录
    split_dir = os.path.join(args.output_dir, f"split_{split_id+1}")
    checkpoint_dir = os.path.join(split_dir, "checkpoints")
    log_dir = os.path.join(split_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_corr_avg",
        dirpath=checkpoint_dir,
        filename=f"best_model_split_{split_id+1}",
        mode="max"
    )

    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="alignment_module"
    )

    train_dataset = DatasetImage('train', split_id)
    val_dataset = DatasetImage('val', split_id)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    model = Alignmodule(
        bert_config=args.bert_config,
        image_size=224,
        vit_weights=args.vit_weights,
        bert_weights=args.bert_weights,
        split_id=split_id,
        output_dir=args.output_dir,
        weight_type=args.weight_type,
        imagereward_path=args.imagereward_path,
        frozen_ratio=args.frozen_ratio,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        weight_decay=args.weight_decay
    )

    trainer = pl.Trainer(
        precision=32,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, ],
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=1.0,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=0
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    print(f"第 {split_id+1} 个划分训练完成")


def main():
    parser = argparse.ArgumentParser(description="alignment_module")

    #dataset
    parser.add_argument('--num_splits', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)

    #model
    parser.add_argument('--bert_config', type=str,
                        default="/root/autodl-tmp/VQualA/alignment_module/med_config.json")
    parser.add_argument('--vit_weights', type=str,
                        default="/root/autodl-tmp/VQualA/alignment_module/model_large.pth")
    parser.add_argument('--bert_weights', type=str,
                        default="/root/autodl-tmp/VQualA/alignment_module/bert-base-uncased/pytorch_model.bin")
    parser.add_argument('--weight_type', type=str, default='imagereward')
    parser.add_argument('--imagereward_path', type=str,
                        default="/root/autodl-tmp/VQualA/alignment_module/imagereward.pth")
    
    #train
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--frozen_ratio', type=float, default=0.7)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.3)

    #trainer
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--log_every_n_steps', type=int, default=5)

    #other
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--output_dir', type=str,
                        default="/root/autodl-tmp/VQualA/alignment_module")
    
    args = parser.parse_args()

    for split_id in range(args.num_splits):
        try:
            train_single_split(args, split_id)
        except Exception as e:
            print(f"第 {split_id+1} 个划分训练失败: {e}")
            continue


if __name__ == "__main__":
    main()
    

