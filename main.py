#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import datetime
import shutil

from loader.data_loader import ParticleDataLoader
from models.model_builder import build_model
from training import train_epoch
from validation import val_epoch
from torch.optim.lr_scheduler import LambdaLR


# ===============================================================
# 🔹 可视化函数：绘制预测 vs 真值 对比图（带反归一化）
# ===============================================================
@torch.no_grad()
def plot_predictions(model, data_loader, device, epoch, label_mean=None, label_std=None, save_dir="results"):
    model.eval()
    all_preds, all_labels = [], []

    for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)

        preds = outputs.detach().cpu().numpy().flatten()
        trues = labels.detach().cpu().numpy().flatten()

        all_preds.extend(preds)
        all_labels.extend(trues)
        break  # 只取一个 batch 可视化

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ✅ 反归一化（如果提供了均值和方差）
    if label_mean is not None and label_std is not None:
        label_mean = np.array(label_mean)
        label_std = np.array(label_std)
        all_preds = all_preds * label_std.mean() + label_mean.mean()
        all_labels = all_labels * label_std.mean() + label_mean.mean()

    mse = np.mean((all_preds - all_labels) ** 2)
    mae = np.mean(np.abs(all_preds - all_labels))

    plt.figure(figsize=(10, 5))
    plt.plot(all_preds, label="prediction", color='royalblue', linewidth=1)
    plt.plot(all_labels, label="truth", color='orange', linewidth=1, alpha=0.8)
    plt.title(f"Epoch {epoch} | MSE={mse:.4f} | MAE={mae:.4f}")
    plt.xlabel("Sample index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"pred_vs_truth_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 已保存预测对比图: {save_path}")


# ===============================================================
# 🔹 构建学习率调度器
# ===============================================================
def build_scheduler(optimizer, config):
    sched_cfg = config.get("scheduler", {})
    name = sched_cfg.get("name", None)

    if name == "CosineAnnealingWarmup":
        warmup_epochs = sched_cfg.get("warmup_epochs", 10)
        max_epochs = sched_cfg.get("max_epochs", 200)
        min_lr = sched_cfg.get("min_lr", 1e-6)
        base_lr = optimizer.param_groups[0]["lr"]

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                cosine_decay = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) /
                                                   (max_epochs - warmup_epochs)))
                return cosine_decay * (1 - min_lr / base_lr) + (min_lr / base_lr)

        scheduler = LambdaLR(optimizer, lr_lambda)
        print(f"✅ 使用 CosineAnnealingWarmup 调度器: warmup={warmup_epochs}, total={max_epochs}")
        return scheduler

    print("⚠️ 未定义或使用默认学习率（无调度）")
    return None


# ===============================================================
# 🔹 主函数
# ===============================================================
def main():
    # 1️⃣ 加载配置文件
    config_path = "data/particle_config/particle_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 从配置中提取归一化参数（用于可视化）
    input_mean = np.array(config.get("input_mean", [0]))
    input_std = np.array(config.get("input_std", [1]))
    label_mean = np.array(config.get("label_mean", [0]))
    label_std = np.array(config.get("label_std", [1]))

    # 2️⃣ 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用设备: {device}")

    # 3️⃣ 创建数据加载器
    data_module = ParticleDataLoader(config)
    train_loader, val_loader, _ = data_module.get_loaders()

    # 4️⃣ 构建模型
    model = build_model(config.get("model", {})).to(device)
    print("✅ 模型已构建完成")

    # 5️⃣ 定义损失与优化器
    criterion = torch.nn.SmoothL1Loss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("optimizer", {}).get("lr", 1e-4),
        weight_decay=1e-5
    )
    scheduler = build_scheduler(optimizer, config)

    # 6️⃣ 实验文件夹（带时间戳）
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = f"results/exp_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # 保存配置副本与模型结构
    shutil.copy(config_path, os.path.join(exp_dir, "config_used.json"))
    with open(os.path.join(exp_dir, "model_summary.txt"), "w") as f:
        f.write(str(model))

    # 7️⃣ 训练主循环
    num_epochs = config.get("training", {}).get("n_epochs", 50)
    best_val_loss = float('inf')
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": [], "val_mse_real": [], "val_mae_real": []}

    for epoch in range(1, num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{num_epochs} ==========")

        # 训练（train_epoch 返回 (epoch_loss, optional_metrics)）
        train_loss, _ = train_epoch(epoch, train_loader, model, criterion, optimizer, device)

        # 验证（val_epoch 返回 (epoch_loss, (mse_real, mae_real))）
        val_loss, (val_mse_real, val_mae_real) = val_epoch(epoch, val_loader, model, criterion, device)

        # 更新学习率（如果使用 LambdaLR 等按 epoch 调度）
        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"📉 当前学习率: {lr:.8f}")

        # 打印本 epoch 指标（train_loss 与 val_loss 使用 SmoothL1Loss，mse/mae 是反归一化后的评估指标）
        print(f"📏 Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}, MSE_real={val_mse_real:.2f}, MAE_real={val_mae_real:.2f}")

        # 记录日志
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)
        history["val_mse_real"].append(val_mse_real)
        history["val_mae_real"].append(val_mae_real)

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f"checkpoints/best_model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 最优模型已保存: {best_model_path}")

        # 可视化（每 10 个 epoch 或最后一个 epoch）
        if epoch % 10 == 0 or epoch == num_epochs:
            plot_predictions(model, val_loader, device, epoch, label_mean, label_std, save_dir=exp_dir)

    print("✅ 训练完成！")

    # 8️⃣ 保存日志与曲线
    df = pd.DataFrame(history)
    csv_path = os.path.join(exp_dir, "training_log.csv")
    df.to_csv(csv_path, index=False)
    print(f"🧾 训练日志已保存到: {csv_path}")

    # Loss 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], df["train_loss"], label='Train Loss', marker='o')
    plt.plot(df["epoch"], df["val_loss"], label='Val Loss', marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "loss_curve.png"))
    plt.close()

    # 学习率曲线
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["lr"], label='Learning Rate', color='purple')
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "lr_curve.png"))
    plt.close()

    print(f"📊 所有曲线与日志均已保存至: {exp_dir}")



if __name__ == "__main__":
    main()
