import os

train_folder = "/root/autodl-tmp/VQualA/data/train"

if os.path.exists(train_folder):
    files = os.listdir(train_folder)
    
    print(f"总文件数: {len(files)}")
else:
    print(f"文件夹不存在: {train_folder}")