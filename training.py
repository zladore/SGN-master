import torch
from tqdm import tqdm
def train_epoch(epoch, data_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                desc=f"训练 Epoch {epoch}", dynamic_ncols=True, leave=True)

    for batch_idx, batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # ✅ 仅在第1个epoch第1个batch打印范围
        if epoch == 1 and batch_idx == 0:
            print(f"[Debug] output range: {outputs.min().item():.2f} ~ {outputs.max().item():.2f}")
            print(f"[Debug] label range : {labels.min().item():.2f} ~ {labels.max().item():.2f}")

        loss = criterion(outputs, labels)
        if torch.isnan(loss):
            print(f"⚠️ NaN loss detected at batch {batch_idx}, skipping...")
            continue

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(data_loader.dataset)
    print(f"✅ 训练 Epoch {epoch} 完成 | Loss: {epoch_loss:.4f}")
    return epoch_loss, None
