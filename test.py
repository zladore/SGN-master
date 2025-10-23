#!/usr/bin/env python3
import os
import sys
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

from models.model_builder import build_model
from loader.ParticleDataset import ParticleDataset

# -----------------------
CONFIG_PATH = "data/particle_config/particle_config.json"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 实时打印
print = lambda *args, **kwargs: (__import__("builtins").print(*args, **kwargs), sys.stdout.flush())


def resolve_paths(base_input_dir, base_output_dir, file_list):
    resolved = []
    for item in file_list:
        img = item["image"]
        lab = item["label"]
        if not os.path.isabs(img):
            img = os.path.join(base_input_dir, os.path.basename(img))
        if not os.path.isabs(lab):
            lab = os.path.join(base_output_dir, os.path.basename(lab))
        resolved.append({"image": img, "label": lab})
    return resolved


def load_latest_checkpoint(checkpoint_dir):
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not ckpts:
        raise FileNotFoundError(f"未在 {checkpoint_dir} 中找到权重文件")
    ckpts = sorted(ckpts, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    return os.path.join(checkpoint_dir, ckpts[-1])


def main():
    print("加载配置...")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # load normalization params for input
    norm_path = os.path.join(config["dataset"]["input_dir"], "normalization_params.json")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"找不到归一化参数文件: {norm_path}")
    norm_params = ParticleDataset.load_normalization_params(norm_path)
    print(f"已加载归一化参数: {norm_path}")

    # prepare test files
    test_files_cfg = config.get("test_filenames", [])
    test_files = resolve_paths(config["dataset"]["input_dir"], config["dataset"]["output_dir"], test_files_cfg)
    if len(test_files) == 0:
        raise ValueError("配置文件中 test_filenames 为空")

    test_dataset = ParticleDataset(
        filenames=test_files,
        transform=None,
        normalize=True,
        normalize_label=False,
        input_mean=norm_params.get("input_mean"),
        input_std=norm_params.get("input_std"),
    )

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"测试集样本数: {len(test_dataset)}")

    # model
    model_cfg = config["model"]
    model = build_model(model_cfg).to(device)
    ckpt_path = load_latest_checkpoint(CHECKPOINT_DIR)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"已加载权重: {ckpt_path}")

    criterion = nn.MSELoss()
    total_loss = 0.0

    preds_list = []
    labels_list = []
    filenames = []

    print("开始测试...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="测试中", dynamic_ncols=True, leave=True)):
            # keys are: "image", "label", "filename"
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            fname = batch["filename"][0]

            preds = model(x)
            # flatten to (B, out_dim)
            if preds.dim() > 2:
                preds = preds.view(preds.size(0), -1)
            if y.dim() > 2:
                y = y.view(y.size(0), -1)
            if preds.dim() == 1:
                preds = preds.unsqueeze(0)
            if y.dim() == 1:
                y = y.unsqueeze(0)

            # compute loss per batch
            loss = criterion(preds, y)
            total_loss += loss.item()

            preds_np = preds.detach().cpu().numpy().squeeze()
            labels_np = y.detach().cpu().numpy().squeeze()
            preds_np = np.asarray(preds_np).reshape(-1)
            labels_np = np.asarray(labels_np).reshape(-1)

            preds_list.append(preds_np.copy())
            labels_list.append(labels_np.copy())
            filenames.append(fname)

            # debug print for first 3 samples
            if batch_idx < 3:
                nprint = min(10, preds_np.size, labels_np.size)
                print(f"\n样本 {fname} 前{nprint}点 (pred | label):")
                for i in range(nprint):
                    print(f"  pred[{i:03d}]={preds_np[i]:.6f} | label[{i:03d}]={labels_np[i]:.6f}")

    # stack into arrays (N, D)
    preds_all = np.stack(preds_list, axis=0)
    labels_all = np.stack(labels_list, axis=0)

    print(f"\nDEBUG: preds_all.shape = {preds_all.shape}, labels_all.shape = {labels_all.shape}, n_files = {len(filenames)}")
    print("DEBUG: first sample pred (first 10):", preds_all[0][:10].tolist())
    print("DEBUG: first sample label (first 10):", labels_all[0][:10].tolist())

    avg_loss = total_loss / len(test_loader)
    print(f"\n测试完成 | 平均Loss={avg_loss:.6f} | 样本数={len(test_dataset)}")

    # save numpy for later inspection
    np.save(os.path.join(RESULTS_DIR, "preds_all.npy"), preds_all)
    np.save(os.path.join(RESULTS_DIR, "labels_all.npy"), labels_all)
    print("已保存 numpy arrays: preds_all.npy, labels_all.npy")

    # --------------------------
    # Save CSV with interleaved columns:
    # filename, pred_0, label_0, pred_1, label_1, ...
    # --------------------------
    out_csv = os.path.join(RESULTS_DIR, "test_results_full.csv")

    N, D = preds_all.shape
    # build columns header
    cols = ["filename"]
    for j in range(D):
        cols.append(f"pred_{j}")
        cols.append(f"label_{j}")

    # create rows
    rows = []
    for i in range(N):
        row = [filenames[i]]
        for j in range(D):
            # ensure float scalar values
            row.append(float(preds_all[i, j]))
            row.append(float(labels_all[i, j]))
        rows.append(row)

    # create DataFrame with specified columns order
    df = pd.DataFrame(rows, columns=cols)

    # preview first row
    if not df.empty:
        preview = df.iloc[0:1]
        print("\nCSV first row preview (interleaved):")
        # print up to first 20 columns for readability
        print(preview.iloc[0, :min(1 + 2*10, len(cols))].to_dict())

    # save to csv
    df.to_csv(out_csv, index=False)
    print(f"CSV 已保存 -> {out_csv}")

    # plot flattened comparison
    plt.figure(figsize=(14, 5))
    plt.plot(preds_all.flatten(), label="Prediction", linewidth=0.8)
    plt.plot(labels_all.flatten(), label="Ground Truth", linewidth=0.8)
    plt.title(f"Prediction vs Ground Truth | MSE={avg_loss:.6f}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    out_fig = os.path.join(RESULTS_DIR, "test_predictions_flatten.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"图像已保存 -> {out_fig}")


if __name__ == "__main__":
    main()
