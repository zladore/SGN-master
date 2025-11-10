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
from loader.ParticleDataset import ParticleDataset
from models.model_builder import build_model
from training import train_epoch
from torch.optim.lr_scheduler import LambdaLR


# ===============================================================
# 🔹 预测可视化（反归一化版本）
# ===============================================================
@torch.no_grad()
def plot_predictions(model, data_loader, device, epoch, label_mean, label_std, save_dir="results"):
    model.eval()
    all_preds, all_labels = [], []

    for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)  # 已是标准化后的标签

        outputs = model(images)  # 输出 shape: (B,250)

        preds = outputs.detach().cpu().numpy()
        trues = labels.detach().cpu().numpy()

        # 仅取一个 batch
        break

    # ✅ 反归一化（逐点）
    # preds: (B,250)  label_mean: (250,) label_std: (250,)
    preds_denorm = preds * label_std + label_mean
    trues_denorm = trues * label_std + label_mean

    # flatten 用于绘图（仅展示趋势）
    preds_flat = preds_denorm.flatten()
    trues_flat = trues_denorm.flatten()

    mse = np.mean((preds_flat - trues_flat) ** 2)
    mae = np.mean(np.abs(preds_flat - trues_flat))

    plt.figure(figsize=(10, 5))
    plt.plot(preds_flat, label="Prediction", linewidth=1)
    plt.plot(trues_flat, label="Ground Truth", linewidth=1, alpha=0.8)
    plt.title(f"Epoch {epoch} | MSE={mse:.4f} | MAE={mae:.4f}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"pred_vs_truth_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"✅ 已保存预测对比图: {save_path}")


# ===============================================================
# 🔹 Cosine Warmup 调度器
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

        print(f"✅ 使用 CosineAnnealingWarmup 调度器: warmup={warmup_epochs}, total={max_epochs}")
        return LambdaLR(optimizer, lr_lambda)

    print("⚠️ 未定义调度器，将使用默认学习率")
    return None


# ===============================================================
# 🔹 主程序
# ===============================================================
def main():
    # ========= 1. 加载配置 =========
    config_path = "data/particle_config/particle_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # ========= 2. 加载 Dataset 计算的归一化参数（正确方式）=========
    norm_path = "data/norm_params/normalization_params.json"
    norm_params = ParticleDataset.load_normalization_params(norm_path)
    label_mean = np.array(norm_params["label_mean"])   # shape: (250,)
    label_std = np.array(norm_params["label_std"])     # shape: (250,)

    # ========= 3. 设备 =========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用设备: {device}")

    # ========= 4. DataLoader =========
    data_module = ParticleDataLoader(config)
    train_loader, _, _ = data_module.get_loaders()

    # ========= 5. 构建模型 =========
    model = build_model(config.get("model", {})).to(device)
    print("✅ 模型已构建完成")

    # ========= 6. 损失与优化器 =========
    criterion = torch.nn.SmoothL1Loss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("optimizer", {}).get("lr", 1e-4),
        weight_decay=1e-5
    )
    scheduler = build_scheduler(optimizer, config)

    # ========= 7. 输出目录 =========
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = f"results/exp_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    shutil.copy(config_path, os.path.join(exp_dir, "config_used.json"))
    with open(os.path.join(exp_dir, "model_summary.txt"), "w") as f:
        f.write(str(model))

    # ========= 8. 训练循环 =========
    num_epochs = config.get("training", {}).get("n_epochs", 50)
    history = {"epoch": [], "train_loss": [], "lr": []}

    for epoch in range(1, num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{num_epochs} ==========")

        train_loss, _ = train_epoch(epoch, train_loader, model, criterion, optimizer, device)

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"📉 当前学习率: {lr:.8f}")
        print(f"📏 Epoch {epoch}: Train Loss={train_loss:.6f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["lr"].append(lr)

        # 每 10 epoch 保存模型 & 绘图
        if epoch % 10 == 0 or epoch == num_epochs:
            ckpt_path = f"checkpoints/model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 模型已保存: {ckpt_path}")

            plot_predictions(model, train_loader, device, epoch, label_mean, label_std, save_dir=exp_dir)

    print("✅ 训练完成！")

    # ========= 9. 保存日志 & 曲线 =========
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(exp_dir, "training_log.csv"), index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], df["train_loss"], marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "train_loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["lr"], color='purple')
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "lr_curve.png"))
    plt.close()

    print(f"📊 所有结果已保存至: {exp_dir}")


if __name__ == "__main__":
    main()
