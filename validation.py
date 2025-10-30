import torch
from tqdm import tqdm

@torch.no_grad()
def val_epoch(epoch, data_loader, model, criterion, device):
    """
    验证一个 epoch（支持归一化与反归一化计算 MSE/MAE）
    """
    model.eval()
    running_loss = 0.0

    pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                desc=f"验证 Epoch {epoch}", dynamic_ncols=True, leave=True)

    all_preds = []
    all_labels = []

    for batch_idx, batch in pbar:
        # 取输入与标签
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # ✅ 打印第一个 batch 的取值范围
        if epoch == 1 and batch_idx == 0:
            print(f"[Debug] [Val] output range: {outputs.min().item():.2f} ~ {outputs.max().item():.2f}")
            print(f"[Debug] [Val] label range : {labels.min().item():.2f} ~ {labels.max().item():.2f}")

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # 保存结果（不转 numpy，保持 torch）
        all_preds.append(outputs.detach())
        all_labels.append(labels.detach())

    # === 平均 loss ===
    epoch_loss = running_loss / len(data_loader.dataset)

    # === 拼接所有 batch ===
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # === ✅ 计算真实空间 MSE / MAE（如果有归一化信息） ===
    mse_real, mae_real = float("nan"), float("nan")
    if hasattr(data_loader.dataset, "label_mean") and data_loader.dataset.label_mean is not None:
        label_mean = torch.tensor(data_loader.dataset.label_mean, device=device, dtype=preds.dtype)
        label_std = torch.tensor(data_loader.dataset.label_std, device=device, dtype=preds.dtype)

        preds_real = preds * label_std + label_mean
        labels_real = labels * label_std + label_mean

        mse_real = torch.mean((preds_real - labels_real) ** 2).item()
        mae_real = torch.mean(torch.abs(preds_real - labels_real)).item()

    print(f"✅ 验证 Epoch {epoch}: Val={epoch_loss:.6f}, MSE_real={mse_real:.2f}, MAE_real={mae_real:.2f}")

    return epoch_loss, (mse_real, mae_real)
