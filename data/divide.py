import os
import shutil
import pandas as pd
from pathlib import Path


"""
划分csv文件
df = pd.read_csv('/root/autodl-tmp/VQualA/data/train.csv')

train_df = df.sample(n=3200, random_state=1)
val_df = df.drop(train_df.index)

train_df.to_csv('/root/autodl-tmp/VQualA/data/train.csv', index=False)
val_df.to_csv('/root/autodl-tmp/VQualA/data/val.csv', index=False)
"""

def organizevideos(source_dir, train_csv_path, val_csv_path, output_path):
    """
    按照train.csv和val.csv文件将视频分别移动到train和val目录

    Args:
        source_dir: 原视频路径
        train_csv_path: train.csv文件路径
        val_csv_path: val.csv文件路径
        output_dir: 输出目录
    """
    train_dir = os.path.join(output_path, 'train')
    val_dir = os.path.join(output_path, 'val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    try:
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return
    
    train_videos = train_df['video_name'].tolist()
    val_videos = val_df['video_name'].tolist()

    train_moved = 0
    val_moved = 0
    not_found = []

    source_path = Path(source_dir)

    for video_file in source_path.glob('*.mp4'):
        video_name = video_file.name

        if video_name in train_videos:
            dest_path = os.path.join(train_dir, video_name)
            try:
                shutil.move(str(video_file), dest_path)
                train_moved += 1
                print(f"已移动到train: {video_name}")
            except Exception as e:
                print(f"移动文件 {video_name} 到train目录时出错: {e}")
        elif video_name in val_videos:
            dest_path = os.path.join(val_dir, video_name)
            try:
                shutil.move(str(video_file), dest_path)
                val_moved += 1
                print(f"以移动到val: {video_name}")
            except Exception as e:
                print(f"移动文件 {video_name} 到val目录时出错: {e}")
        else:
            not_found.append(video_name)

    #检查是否有csv中的视频在源目录中找不到
    missing_train = []
    missing_val = []

    for video_name in train_videos:
        if not os.path.exists(os.path.join(train_dir, video_name)):
            missing_train.append(video_name)

    for video_name in val_videos:
        if not os.path.exists(os.path.join(val_dir, video_name)):
            missing_val.append(video_name)

    #打印统计信息
    print("\n" + "="*50)
    print("移动完成! 统计信息:")
    print(f"成功移动到train目录: {train_moved} 个视频")
    print(f"成功移动到val目录: {val_moved} 个视频")
    print(f"总共移动: {train_moved + val_moved} 个视频")

    if not_found:
        print(f"\n在源目录中找到但不在CSV文件中的视频: {len(not_found)} 个")
        print("这些视频没有被移动:")
        for video in not_found[:10]:  # 只显示前10个
            print(f"  - {video}")
        if len(not_found) > 10:
            print(f"  ... 还有 {len(not_found) - 10} 个")
    
    if missing_train:
        print(f"\ntrain.csv中列出但源目录中找不到的视频: {len(missing_train)} 个")
        for video in missing_train[:10]:
            print(f"  - {video}")
        if len(missing_train) > 10:
            print(f"  ... 还有 {len(missing_train) - 10} 个")
    
    if missing_val:
        print(f"\nval.csv中列出但源目录中找不到的视频: {len(missing_val)} 个")
        for video in missing_val[:10]:
            print(f"  - {video}")
        if len(missing_val) > 10:
            print(f"  ... 还有 {len(missing_val) - 10} 个")

if __name__ == "__main__":
    source_dir = "/root/autodl-tmp/VQualA/data/train"
    train_csv_path = "/root/autodl-tmp/VQualA/data/train.csv"
    val_csv_path = "/root/autodl-tmp/VQualA/data/val.csv"
    output_path = "/root/autodl-tmp/VQualA/data"

    organizevideos(source_dir, train_csv_path, val_csv_path, output_path)