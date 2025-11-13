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

    for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        preds = outputs.detach().cpu().numpy()
        trues = labels.detach().cpu().numpy()
        break

    # 反归一化
    preds_denorm = preds * label_std + label_mean
    trues_denorm = trues * label_std + label_mean

    preds_flat = preds_denorm.flatten()
    trues_flat = trues_denorm.flatten()

    mse = np.mean((preds_flat - trues_flat) ** 2)
    mae = np.mean(np.abs(preds_flat - trues_flat))

    plt.figure(figsize=(10, 5))
    plt.plot(preds_flat, label="Prediction")
    plt.plot(trues_flat, label="Ground Truth")
    plt.title(f"Epoch {epoch} | MSE={mse:.4f} | MAE={mae:.4f}")
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
                cosine_decay = 0.5 * (1 + math.cos(
                    math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
                ))
                return cosine_decay * (1 - min_lr / base_lr) + (min_lr / base_lr)

        print(f"✅ 使用 CosineAnnealingWarmup 调度器")
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

    # ========= 2. 设备 =========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用设备: {device}")

    # ========= 3. DataLoader（自动计算归一化）=========
    data_module = ParticleDataLoader(config)
    train_loader, _, _ = data_module.get_loaders()
    train_dataset, _, _ = data_module.get_datasets()

    # ========= 4. 最新归一化参数 =========
    # 直接从训练集动态获取归一化参数。
    # norm_params = train_dataset.get_normalization_params()
    # label_mean = norm_params["label_mean"]
    # label_std = norm_params["label_std"]
    # 从 文件加载归一化参数
    norm_path = "data/norm_params/normalization_params.json"
    norm_params = ParticleDataset.load_normalization_params(norm_path)
    label_mean = np.array(norm_params["label_mean"])
    label_std = np.array(norm_params["label_std"])

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

    # ✅ 训练日志
    history = {"epoch": [], "train_loss": [], "mse_real": [], "mae_real": [], "lr": []}

    # ✅【你要求的：最优模型保存机制】
    BEST_METRIC = "loss"  # 可改为 "loss" / "mae" / "mse"
    best_value = float("inf")
    best_ckpt_path = "checkpoints/best_model.pth"
    print(f"⭐ 使用最优指标保存策略：{BEST_METRIC}")

    for epoch in range(1, num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{num_epochs} ==========")

        # 训练
        train_loss, (mse_real, mae_real) = train_epoch(
            epoch, train_loader, model, criterion, optimizer, device
        )

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        print(f"📉 当前学习率: {lr:.8f}")
        print(f"📊 Loss={train_loss:.6f},  MSE={mse_real:.6f},  MAE={mae_real:.6f}")

        # 记录日志
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["mse_real"].append(mse_real)
        history["mae_real"].append(mae_real)
        history["lr"].append(lr)

        # ========= ✅最优模型判定逻辑 =========
        if BEST_METRIC == "loss":
            current = train_loss
        elif BEST_METRIC == "mae":
            current = mae_real
        elif BEST_METRIC == "mse":
            current = mse_real
        else:
            raise ValueError("BEST_METRIC 必须是 loss / mae / mse")

        # ✅ 当前指标更优 → 保存
        if current < best_value:
            print(f"💾 最优模型更新 {best_value:.6f} → {current:.6f} (Epoch {epoch})")
            best_value = current
            best_ckpt_path = f"checkpoints/best_model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), best_ckpt_path)

        # ========= 可视化（每10轮） =========
        if epoch % 10 == 0 or epoch == num_epochs:
            plot_predictions(model, train_loader, device, epoch, label_mean, label_std, exp_dir)

    print(f"\n✅ 训练结束，最佳 {BEST_METRIC} = {best_value:.6f}")
    print(f"✅ 最优模型保存在：{best_ckpt_path}")

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
