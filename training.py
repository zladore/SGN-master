import torch
from tqdm import tqdm
import numpy as np

def train_epoch(epoch, data_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    all_preds, all_labels = [], []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                desc=f"训练 Epoch {epoch}", dynamic_ncols=True, leave=True)

    for batch_idx, batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # ✅ 第一个 batch 打印范围
        if epoch == 1 and batch_idx == 0:
            print(f"[Debug] output range: {outputs.min().item():.2f} ~ {outputs.max().item():.2f}")
            print(f"[Debug] label range : {labels.min().item():.2f} ~ {labels.max().item():.2f}")

        # === 计算损失 ===
        loss = criterion(outputs, labels)
        if torch.isnan(loss):
            print(f"⚠️ NaN loss detected at batch {batch_idx}, skipping...")
            continue

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # 保存预测与真实值以便统计真实误差
        all_preds.append(outputs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    # === 计算平均损失 ===
    epoch_loss = running_loss / len(data_loader.dataset)

    # === 汇总所有预测 ===
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    # === ✅ 计算真实空间 MSE/MAE（如果归一化信息存在） ===
    mse_real, mae_real = np.nan, np.nan
    if hasattr(data_loader.dataset, "label_mean") and data_loader.dataset.label_mean is not None:
        label_mean = torch.tensor(data_loader.dataset.label_mean).to(preds.device)
        label_std = torch.tensor(data_loader.dataset.label_std).to(preds.device)

        preds_real = preds * label_std + label_mean
        labels_real = labels * label_std + label_mean

        mse_real = torch.mean((preds_real - labels_real) ** 2).item()
        mae_real = torch.mean(torch.abs(preds_real - labels_real)).item()

    print(f"✅ Epoch {epoch}: Train={epoch_loss:.6f}, MSE_real={mse_real:.2f}, MAE_real={mae_real:.2f}")
    return epoch_loss, (mse_real, mae_real)
