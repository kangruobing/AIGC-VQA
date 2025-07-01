import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("/root/autodl-tmp/VQualA")
from dataset import DatasetImage

from model.model import Traditionalmodule


def load_model(model_path, device):
    """加载模型权重"""
    print(f"正在加载模型: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if all(k.startswith('model.') for k in state_dict.keys()):
        print("检测到Pytorch Lightning前缀, 正在移除...")
        state_dict = {k[6:]: v for k, v in state_dict.items()}

    model = Traditionalmodule(split_id=0)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if len(missing) > 0:
        print(f"警告: 缺少{len(missing)}个参数")
    
    if len(unexpected) > 0:
        print(f"警告: 有{len(unexpected)}个意外参数")

    model.to(device)
    model.eval()

    return model


def predict_single_model(model, dataloader, device):
    """使用单个模型进行预测"""
    print("使用单个模型进行预测...")
    all_predictions = []
    all_video_names = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预测中..."):
            image = batch["image"].to(device)
            video_names = batch["video_name"]

            predicted_score = model.forward(image)

            all_predictions.extend(predicted_score.cpu().numpy())
            all_video_names.extend(video_names)

    return all_video_names, all_predictions


def predict_ensemble_models(models, dataloader, device):
    """使用多个模型集成进行预测"""
    print(f"使用 {len(models)} 个模型进行预测...")
    all_video_names = []
    all_ensemble_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预测中..."):
            image = batch["image"].to(device)
            video_name = batch["video_name"]

            batch_predictions = []
            for i, model in enumerate(models):
                model.eval()
                predicted_score = model.forward(image)
                batch_predictions.append(predicted_score.cpu().numpy())

            ensemble_predictions = np.mean(batch_predictions, axis=0)

            all_ensemble_predictions.extend(ensemble_predictions)
            all_video_names.extend(video_name)

    return all_video_names, all_ensemble_predictions


def main():
    parser = argparse.ArgumentParser(description="test")

    parser.add_argument('--model_paths', nargs='+', type=str, required=True)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/VQualA/traditional_module/results')
    parser.add_argument('--output_file', type=str, default='traditional_predictions.csv')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = DatasetImage('test', 0, video_size=384)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    print(f"测试集大小: {len(test_dataset)}")

    is_ensemble = len(args.model_paths) > 1

    if is_ensemble:
        models = [load_model(path, device) for path in args.model_paths]

        video_names, predictions = predict_ensemble_models(models, test_dataloader, device)

        output_file = os.path.join(args.output_dir, f"ensemble_{args.output_file}")
    else:
        model = load_model(args.model_paths[0], device)

        video_names, predictions = predict_single_model(model, test_dataloader, device)

        model_name = os.path.basename(args.model_paths[0]).split('.')[0]
        output_file = os.path.join(args.output_dir, f"{model_name}_{args.output_file}")

    results_df = pd.DataFrame({
        'video_name': video_names,
        'Traditional_MOS': predictions
    })

    results_df.to_csv(output_file, index=False)
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()