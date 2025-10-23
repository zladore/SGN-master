#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from loader.data_loader import ParticleDataLoader
from models.model_builder import build_model
from training import train_epoch
from validation import val_epoch


# ===============================================================
# �� 可视化函数：绘制预测 vs 真值 对比图
# ===============================================================
@torch.no_grad()
def plot_predictions(model, data_loader, device, epoch, save_dir="results"):
    model.eval()
    all_preds, all_labels = [], []

    # 取一个 batch 来可视化
    for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)

        preds = outputs.detach().cpu().numpy().flatten()
        trues = labels.detach().cpu().numpy().flatten()

        all_preds.extend(preds)
        all_labels.extend(trues)
        break  # 只取第一个 batch 就够画图了

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    mse = np.mean((all_preds - all_labels) ** 2)

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(all_preds, label="prediction", color='royalblue', linewidth=1)
    plt.plot(all_labels, label="truth", color='orange', linewidth=1, alpha=0.8)
    plt.title(f"Epoch {epoch} | MSE={mse:.6f}")
    plt.xlabel("Sample index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"pred_vs_truth_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"�� 已保存预测对比图: {save_path}")


# ===============================================================
# �� 主函数
# ===============================================================
def main():
    # 1️⃣ 加载配置文件
    config_path = "data/particle_config/particle_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 2️⃣ 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 3️⃣ 构建数据加载器
    data_module = ParticleDataLoader(config)
    train_loader, val_loader, _ = data_module.get_loaders()

    # 4️⃣ 构建模型
    model_config = config.get("model", {
        "name": "3DResNeXt101",
        "in_channels": 4,
        "out_features": 250
    })
    model = build_model(model_config).to(device)
    print("✅ 模型已构建完成")

    # 5️⃣ 定义损失函数与优化器
    criterion = torch.nn.SmoothL1Loss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("optimizer", {}).get("lr", 1e-4),
        weight_decay=1e-5
    )

    # 6️⃣ 训练循环
    num_epochs = config.get("training", {}).get("n_epochs", 50)
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{num_epochs} ==========")

        train_loss, _ = train_epoch(epoch, train_loader, model, criterion, optimizer, device)
        val_loss, _ = val_epoch(epoch, val_loader, model, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"�� Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        # ✅ 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"checkpoints/best_model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"�� 最优模型已保存到: {save_path} (val_loss={val_loss:.6f})")

        # ✅ 每 10 个 epoch 绘制一次预测对比图
        if epoch % 10 == 0 or epoch == num_epochs:
            plot_predictions(model, val_loader, device, epoch, save_dir="results")

    print("✅ 训练完成！")

    # 7️⃣ 保存 Loss 数据
    loss_data = pd.DataFrame({
        "epoch": list(range(1, num_epochs + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    csv_path = "results/loss_history.csv"
    loss_data.to_csv(csv_path, index=False)
    print(f"�� Loss 数据已保存到 {csv_path}")

    # 8️⃣ 绘制 Loss 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(loss_data["epoch"], loss_data["train_loss"], label='Train Loss', marker='o')
    plt.plot(loss_data["epoch"], loss_data["val_loss"], label='Validation Loss', marker='s')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fig_path = "results/loss_curve.png"
    plt.savefig(fig_path)
    plt.show()
    print(f"�� 训练/验证 Loss 曲线已保存到 {fig_path}")


if __name__ == "__main__":
    main()
