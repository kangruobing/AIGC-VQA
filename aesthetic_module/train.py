import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model.model import Aestheticmodule

import sys
sys.path.append('/root/autodl-tmp/VQualA')

import torch
from torch.utils.data import DataLoader
from dataset import DatasetVideo

def train():
    checkpoint_callback = ModelCheckpoint(
        monitor="val_corr_avg",
        dirpath="/root/autodl-tmp/VQualA/aesthetic_module/checkpoints",
        filename="best_model",
        mode="max"
    )

    logger = TensorBoardLogger(
        save_dir="/root/autodl-tmp/VQualA/aesthetic_module/logs/tensorboard",
        name="aesthetic_module"
    )

    train_dataset = DatasetVideo('train')
    val_dataset = DatasetVideo('val')
    train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, num_workers=8, shuffle=True)

    model = Aestheticmodule(
        model_size="tiny",
        lr=1e-4,
        alpha=1.0,
        beta=0.3,
        weight_decay=0.01
    )

    trainer = pl.Trainer(
        precision=16,
        max_epochs=50,
        callbacks=[checkpoint_callback, ],
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
        num_sanity_val_steps=0
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    print(f"=== 训练完成 ===")

    #加载最佳模型对训练集进行评估
    if checkpoint_callback.best_model_path:
        print("\n加载最佳模型进行最终评估...")
        best_model = Aestheticmodule.load_from_checkpoint(checkpoint_callback.best_model_path)
        best_model.to(model.device)

        final_results = best_model.evaluate_dataset(train_dataloader)

        print(f"最终评估完成! 平均相关系数: {final_results['corr_avg']:.4f}")
    else:
        print("未找到最佳模型, 使用当前模型进行评估...")

        final_results = model.evaluate_dataset(train_dataloader)

        print(f"最终评估完成! 平均相关系数: {final_results['corr_avg']:.4f}")

if __name__ == "__main__":
    train()