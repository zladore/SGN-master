#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_infer.py
适配于 main.py 自动推理调用。
支持：
 - 自动选择最新或指定 checkpoint
 - 使用固定归一化参数路径
 - 输出完整指标与可视化
"""

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
# 🧭 通用配置
# ===============================================================
CONFIG_PATH = "data/particle_config/particle_config.json"
NORM_PARAM_PATH = "data/norm_params/normalization_params.json"
CHECKPOINT_DIR = "checkpoints"
RESULTS_BASE = "results"
os.makedirs(RESULTS_BASE, exist_ok=True)


# ===============================================================
# 🔍 自动加载 checkpoint
# ===============================================================
def get_checkpoint_path(checkpoint_dir, specified_ckpt=None):
    """
    若指定路径存在，则直接使用；
    若不存在，则回退到目录中最新的 checkpoint。
    """
    if specified_ckpt and os.path.exists(specified_ckpt):
        print(f"✅ 使用指定模型权重: {specified_ckpt}")
        return specified_ckpt

    ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not ckpts:
        raise FileNotFoundError(f"❌ 未在 {checkpoint_dir} 中找到任何 .pth 文件")

    ckpts = sorted(ckpts, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    latest = os.path.join(checkpoint_dir, ckpts[-1])
    print(f"⚠️ 指定的权重未找到，已回退到最新权重: {latest}")
    return latest


# ===============================================================
# 🧮 推理函数（可从 main.py 调用）
# ===============================================================
def run_inference(checkpoint_path=None, tag="auto_test"):
    """
    可被 main.py 调用的推理函数。
    Args:
        checkpoint_path (str, optional): 指定 checkpoint 路径。
        tag (str): 输出文件夹标识名。
    """
    # ---------------------------------------------------------------
    # 1️⃣ 加载配置
    # ---------------------------------------------------------------
    print("📘 加载配置与归一化参数...")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # 归一化参数
    with open(NORM_PARAM_PATH, "r") as f:
        norm_params = json.load(f)

    input_mean = np.array(norm_params.get("input_mean", [0]))
    input_std = np.array(norm_params.get("input_std", [1]))
    label_mean = np.array(norm_params.get("label_mean", [0]))
    label_std = np.array(norm_params.get("label_std", [1]))

    if input_mean.size != 4 or input_std.size != 4:
        raise ValueError(f"❌ input_mean/std 形状错误，应为长度3")

    # ---------------------------------------------------------------
    # 2️⃣ 设置设备 & 加载模型
    # ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"]).to(device)

    ckpt_path = get_checkpoint_path(CHECKPOINT_DIR, checkpoint_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"✅ 模型加载完成: {ckpt_path}")

    # ---------------------------------------------------------------
    # 3️⃣ 数据集准备
    # ---------------------------------------------------------------
    test_files_cfg = config.get("test_filenames", [])
    if not test_files_cfg:
        raise ValueError("⚠️ 配置文件中 test_filenames 为空！")

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

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"✅ 测试样本数: {len(test_dataset)}")

    # ---------------------------------------------------------------
    # 4️⃣ 推理循环
    # ---------------------------------------------------------------
    criterion = nn.SmoothL1Loss()
    total_loss = 0.0
    preds_list, labels_list, filenames = [], [], []

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join(RESULTS_BASE, f"{tag}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    print("🚀 开始推理...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", dynamic_ncols=True):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            fname = batch["filename"][0]

            preds = model(x).view(x.size(0), -1)
            y = y.view(y.size(0), -1)

            loss = criterion(preds, y)
            total_loss += loss.item()

            preds_np = preds.cpu().numpy().squeeze()
            labels_np = y.cpu().numpy().squeeze()

            preds_np = preds_np * label_std + label_mean
            labels_np = labels_np * label_std + label_mean

            preds_list.append(preds_np)
            labels_list.append(labels_np)
            filenames.append(fname)

    # ---------------------------------------------------------------
    # 5️⃣ 结果计算与保存
    # ---------------------------------------------------------------
    preds_all = np.stack(preds_list)
    labels_all = np.stack(labels_list)

    avg_loss = total_loss / len(test_loader)
    mse_real = np.mean((preds_all - labels_all) ** 2)
    mae_real = np.mean(np.abs(preds_all - labels_all))
    ss_res = np.sum((labels_all - preds_all) ** 2)
    ss_tot = np.sum((labels_all - np.mean(labels_all)) ** 2)
    r2_score = 1 - ss_res / ss_tot

    print(f"\n✅ 推理完成！")
    print(f"📊 SmoothL1Loss(归一化域)={avg_loss:.6f}")
    print(f"📏 MSE={mse_real:.3f} | MAE={mae_real:.3f} | R²={r2_score:.4f}")

    # 保存 CSV（交错格式）
    cols = ["filename"] + [f"pred_{i},label_{i}" for i in range(preds_all.shape[1])]
    df_rows = []
    for i, fname in enumerate(filenames):
        row = [fname]
        for j in range(preds_all.shape[1]):
            row += [preds_all[i, j], labels_all[i, j]]
        df_rows.append(row)

    df = pd.DataFrame(df_rows, columns=["filename"] + sum([[f"pred_{i}", f"label_{i}"] for i in range(preds_all.shape[1])], []))
    csv_path = os.path.join(exp_dir, "test_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV 已保存 -> {csv_path}")

    # 保存误差摘要
    with open(os.path.join(exp_dir, "metrics_summary.txt"), "w") as f:
        f.write(f"Average SmoothL1Loss: {avg_loss:.6f}\n")
        f.write(f"MSE: {mse_real:.6f}\n")
        f.write(f"MAE: {mae_real:.6f}\n")
        f.write(f"R²: {r2_score:.6f}\n")

    # 绘图
    plt.figure(figsize=(14, 5))
    plt.plot(preds_all.flatten(), label="Pred")
    plt.plot(labels_all.flatten(), label="True")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
    plt.title(f"MSE={mse_real:.3f}, MAE={mae_real:.3f}, R²={r2_score:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(labels_all.flatten(), preds_all.flatten(), s=5, alpha=0.5)
    plt.xlabel("True"); plt.ylabel("Pred"); plt.grid(True, linestyle="--", alpha=0.4)
    plt.title("Pred vs True")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "scatter.png"), dpi=300)
    plt.close()

    print(f"🎨 所有图像与结果已保存至 {exp_dir}")

    return {
        "exp_dir": exp_dir,
        "ckpt": ckpt_path,
        "avg_loss": avg_loss,
        "mse": mse_real,
        "mae": mae_real,
        "r2": r2_score
    }


# ===============================================================
# CLI 启动
# ===============================================================
if __name__ == "__main__":
    run_inference()
