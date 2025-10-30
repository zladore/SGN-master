#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime

from models.model_builder import build_model
from loader.ParticleDataset import ParticleDataset


# ===============================================================
# 🔧 基本配置
# ===============================================================
CONFIG_PATH = "data/particle_config/particle_config.json"
CHECKPOINT_DIR = "checkpoints"
RESULTS_BASE = "results"
os.makedirs(RESULTS_BASE, exist_ok=True)

# 实时输出
print = lambda *args, **kwargs: (__import__("builtins").print(*args, **kwargs), sys.stdout.flush())


# ===============================================================
# 🧠 自动加载最新 checkpoint
# ===============================================================
def load_latest_checkpoint(checkpoint_dir):
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not ckpts:
        raise FileNotFoundError(f"未在 {checkpoint_dir} 中找到模型权重文件 (.pth)")
    ckpts = sorted(ckpts, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    latest = os.path.join(checkpoint_dir, ckpts[-1])
    print(f"✅ 已加载最新权重: {latest}")
    return latest


# ===============================================================
# 🧮 主推理函数
# ===============================================================
def main():
    # 1️⃣ 读取配置
    print("加载配置中...")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # 2️⃣ 提取归一化参数（从 normalization_params.json）
    norm_path = "data/norm_params/normalization_params.json"
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"❌ 未找到归一化参数文件: {norm_path}")

    with open(norm_path, "r") as f:
        norm_params = json.load(f)

    # 转成 numpy 数组
    input_mean = np.array(norm_params.get("input_mean", [0]))
    input_std = np.array(norm_params.get("input_std", [1]))
    label_mean = np.array(norm_params.get("label_mean", [0]))
    label_std = np.array(norm_params.get("label_std", [1]))

    # ✅ 检查形状（input 4通道）
    if input_mean.size != 4 or input_std.size != 4:
        raise ValueError(f"❌ input_mean/std 形状错误: mean={input_mean.shape}, std={input_std.shape}，应为长度4")

    print(f"✅ 已加载归一化参数 from {norm_path}")
    print(f"   input_mean: {input_mean}")
    print(f"   input_std : {input_std}")
    print(f"   label_mean shape: {label_mean.shape}, label_std shape: {label_std.shape}")

    # 3️⃣ 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用设备: {device}")

    # 4️⃣ 加载测试数据
    dataset_cfg = config.get("dataset", {})
    test_files_cfg = config.get("test_filenames", [])
    if len(test_files_cfg) == 0:
        raise ValueError("⚠️ 配置文件中 test_filenames 为空")

    test_dataset = ParticleDataset(
        filenames=test_files_cfg,
        transform=None,
        normalize_input=True,
        normalize_label=True,
        input_mean=input_mean,
        input_std=input_std,
        label_mean=label_mean,
        label_std=label_std
    )

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"✅ 测试样本数: {len(test_dataset)}")

    # 5️⃣ 构建模型并加载权重
    model = build_model(config["model"]).to(device)
    ckpt_path = load_latest_checkpoint(CHECKPOINT_DIR)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    criterion = nn.SmoothL1Loss()
    total_loss = 0.0
    preds_list, labels_list, filenames = [], [], []

    # 6️⃣ 输出路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join(RESULTS_BASE, f"exp_{timestamp}_test")
    os.makedirs(exp_dir, exist_ok=True)

    # 7️⃣ 开始推理
    print("🚀 开始推理...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing", dynamic_ncols=True)):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            fname = batch["filename"][0]

            preds = model(x)
            preds = preds.view(preds.size(0), -1)
            y = y.view(y.size(0), -1)

            loss = criterion(preds, y)
            total_loss += loss.item()

            preds_np = preds.cpu().numpy().squeeze()
            labels_np = y.cpu().numpy().squeeze()

            # 反归一化
            preds_np = preds_np * label_std + label_mean
            labels_np = labels_np * label_std + label_mean

            preds_list.append(preds_np)
            labels_list.append(labels_np)
            filenames.append(fname)

            # 前几个样本打印
            if batch_idx < 3:
                nprint = min(10, len(preds_np))
                print(f"\n样本 {fname} (前{nprint}个点, 已反归一化):")
                for i in range(nprint):
                    print(f"  pred[{i:03d}]={preds_np[i]:.4f} | label[{i:03d}]={labels_np[i]:.4f}")

    # 8️⃣ 结果统计
    preds_all = np.stack(preds_list, axis=0)
    labels_all = np.stack(labels_list, axis=0)
    avg_loss = total_loss / len(test_loader)

    # 计算真实误差指标
    mse_real = np.mean((preds_all - labels_all) ** 2)
    mae_real = np.mean(np.abs(preds_all - labels_all))
    ss_res = np.sum((labels_all - preds_all) ** 2)
    ss_tot = np.sum((labels_all - np.mean(labels_all)) ** 2)
    r2_score = 1 - ss_res / ss_tot

    print(f"\n✅ 推理完成！")
    print(f"📊 平均 SmoothL1Loss(归一化域) = {avg_loss:.6f}")
    print(f"📏 真实误差: MSE={mse_real:.3f} | MAE={mae_real:.3f} | R²={r2_score:.4f}")

    # 9️⃣ 保存结果
    np.save(os.path.join(exp_dir, "preds_all.npy"), preds_all)
    np.save(os.path.join(exp_dir, "labels_all.npy"), labels_all)

    # 9️⃣ 保存结果
    preds_all = np.stack(preds_list, axis=0)
    labels_all = np.stack(labels_list, axis=0)
    avg_loss = total_loss / len(test_loader)

    # 计算真实误差指标
    mse_real = np.mean((preds_all - labels_all) ** 2)
    mae_real = np.mean(np.abs(preds_all - labels_all))
    ss_res = np.sum((labels_all - preds_all) ** 2)
    ss_tot = np.sum((labels_all - np.mean(labels_all)) ** 2)
    r2_score = 1 - ss_res / ss_tot

    print(f"\n✅ 推理完成！")
    print(f"📊 平均 SmoothL1Loss(归一化域) = {avg_loss:.6f}")
    print(f"📏 真实误差: MSE={mse_real:.3f} | MAE={mae_real:.3f} | R²={r2_score:.4f}")

    # =========================================================
    # ✅ 生成交错列格式的 CSV: filename, pred_0, label_0, pred_1, label_1, ...
    # =========================================================
    N, D = preds_all.shape
    cols = ["filename"]
    for i in range(D):
        cols += [f"pred_{i}", f"label_{i}"]

    rows = []
    for i in range(N):
        row = [filenames[i]]
        for j in range(D):
            row += [preds_all[i, j], labels_all[i, j]]
        rows.append(row)

    df = pd.DataFrame(rows, columns=cols)
    csv_path = os.path.join(exp_dir, "test_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ 已保存交错格式 CSV -> {csv_path}")


    # 误差摘要
    metrics_path = os.path.join(exp_dir, "metrics_summary.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Average SmoothL1Loss (normalized): {avg_loss:.6f}\n")
        f.write(f"MSE (real): {mse_real:.6f}\n")
        f.write(f"MAE (real): {mae_real:.6f}\n")
        f.write(f"R² (real): {r2_score:.6f}\n")
    print(f"✅ 误差摘要保存至: {metrics_path}")

    # 10️⃣ 绘图保存
    plt.figure(figsize=(14, 5))
    plt.plot(preds_all.flatten(), label="Prediction", linewidth=0.8)
    plt.plot(labels_all.flatten(), label="Ground Truth", linewidth=0.8)
    plt.title(f"Prediction vs Ground Truth (MSE={mse_real:.3f}, MAE={mae_real:.3f}, R²={r2_score:.3f})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "test_predictions_curve.png"), dpi=300)
    plt.close()

    # 散点对比图
    plt.figure(figsize=(6, 6))
    plt.scatter(labels_all.flatten(), preds_all.flatten(), s=5, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Predicted vs True (after denormalization)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "scatter_pred_vs_true.png"), dpi=300)
    plt.close()

    print(f"🎨 所有图像与结果已保存到: {exp_dir}")


if __name__ == "__main__":
    main()
