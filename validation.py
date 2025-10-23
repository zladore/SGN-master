import torch
from tqdm import tqdm

@torch.no_grad()
def val_epoch(epoch, data_loader, model, criterion, device):
    """
    验证一个 epoch（适用于未对 label 归一化的情况）
    """
    model.eval()
    running_loss = 0.0

    pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                desc=f"验证 Epoch {epoch}", dynamic_ncols=True, leave=True)

    for batch_idx, batch in pbar:
        # 获取输入与标签
        images = batch["image"].to(device)   # ⚠️ 注意是 "image"（你的Dataset里是这个key）
        labels = batch["label"].to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # ✅ 在第一个epoch的第一个batch打印数值范围
        if epoch == 1 and batch_idx == 0:
            print(f"[Debug] [Val] output range: {outputs.min().item():.2f} ~ {outputs.max().item():.2f}")
            print(f"[Debug] [Val] label range : {labels.min().item():.2f} ~ {labels.max().item():.2f}")

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # 计算平均loss
    epoch_loss = running_loss / len(data_loader.dataset)
    print(f"✅ 验证 Epoch {epoch} 完成 | Loss: {epoch_loss:.4f}")

    return epoch_loss, None
