import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torch.utils.data import DataLoader
from model import AIGCVQA
from dataset import DatasetTrack1

def train():
    checkpoint_callback = ModelCheckpoint(
        monitor="val_corr_avg",
        dirpath="/root/autodl-tmp/VQualA/checkpoints",
        filename="best_model",
        mode="max"
    )

    logger = TensorBoardLogger(
        save_dir="/root/autodl-tmp/VQualA/logs/tensorboard",
        name="AIGCVQA"
    )

    train_dataset = DatasetTrack1('train', video_clip_length=16)
    val_dataset = DatasetTrack1('val', video_clip_length=16)
    train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, num_workers=1, shuffle=True)

    model = AIGCVQA(
        align_frozen_ratio=0.7,
        traditional_freeze_strategy='partial',
        traditional_freeze_ratio=0.3,
        aesmodule_size='tiny',
        freeze_temporalmodule='False',
        lr=1e-5,
        alpha=1.0,
        beta=0.3,
        weight_decay=0.05,
        dropout=0.2
    )

    trainer = pl.Trainer(
        precision=16,
        max_epochs=20,
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
        best_model = AIGCVQA.load_from_checkpoint(checkpoint_callback.best_model_path)
        best_model.to(model.device)

        final_results = best_model.evaluate_dataset(train_dataloader)

        print(f"最终评估完成! 平均相关系数: {final_results['corr_avg']:.4f}")
    else:
        print("未找到最佳模型, 使用当前模型进行评估...")

        final_results = model.evaluate_dataset(train_dataloader)

        print(f"最终评估完成! 平均相关系数: {final_results['corr_avg']:.4f}")

if __name__ == "__main__":
    train()