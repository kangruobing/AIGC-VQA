import pandas as pd

df = pd.read_csv("/root/autodl-tmp/VQualA/data/train.csv")

num_splits = 5
train_size = 3200

for i in range(num_splits):
    train_df = df.sample(n=train_size, random_state=i+3)
    val_df = df.drop(train_df.index)

    train_df.to_csv(f"/root/autodl-tmp/VQualA/data/train_{i+1}.csv", index=False)
    val_df.to_csv(f"/root/autodl-tmp/VQualA/data/val_{i+1}.csv", index=False)

    print(f"第 {i+1} 次划分: 训练集 {len(train_df)} 个样本, 验证集 {len(val_df)} 个样本")

print(f"生成 {num_splits} 组训练/验证集")


