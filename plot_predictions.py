import torch
import matplotlib.pyplot as plt
import numpy as np
import os


@torch.no_grad()
def plot_predictions(model, data_loader, device, epoch, save_dir="results"):
    """
    绘制预测结果 vs 真实值 对比图，并保存为 PNG 文件。

    Args:
        model: 已训练的 PyTorch 模型
        data_loader: 验证集或测试集的 DataLoader
        device: torch.device("cuda" or "cpu")
        epoch: 当前 epoch，用于命名保存文件
        save_dir: 保存图像的目录
    """
    model.eval()
    all_preds, all_labels = [], []

    # �� 仅取第一个 batch 进行可视化
    for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)

        preds = outputs.detach().cpu().numpy().flatten()
        trues = labels.detach().cpu().numpy().flatten()

        all_preds.extend(preds)
        all_labels.extend(trues)
        break  # 只取一个 batch 用于画图

    # �� 转为 numpy 并计算 MSE
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    mse = np.mean((all_preds - all_labels) ** 2)

    # �� 绘制预测 vs 真值曲线
    plt.figure(figsize=(10, 5))
    plt.plot(all_preds, label="prediction", color='royalblue', linewidth=1)
    plt.plot(all_labels, label="truth", color='orange', linewidth=1, alpha=0.8)
    plt.title(f"Epoch {epoch} | MSE={mse:.6f}")
    plt.xlabel("Sample index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    # �� 保存图像
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"pred_vs_truth_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"�� 已保存预测对比图: {save_path}")
