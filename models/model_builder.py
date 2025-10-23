# models/model_builder.py
import torch
from models.resnet import generate_model as generate_resnet
from models.resnext import generate_model as generate_resnext

def build_model(model_cfg):
    """
    根据 model_cfg 构建模型。
    model_cfg 示例:
    {
        "name": "3DResNeXt101",
        "in_channels": 4,
        "out_features": 250
    }
    """
    name = model_cfg.get("name", "3DResNeXt101")
    in_channels = model_cfg.get("in_channels", 4)
    out_features = model_cfg.get("out_features", 250)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🧩 Building model: {name} (in_channels={in_channels}, out_features={out_features})")

    # ResNeXt 系列
    if "ResNeXt" in name or "resnext" in name.lower():
        if "101" in name:
            model = generate_resnext(101, n_input_channels=in_channels, n_classes=out_features)
        elif "50" in name:
            model = generate_resnext(50, n_input_channels=in_channels, n_classes=out_features)
        else:
            raise ValueError(f"Unsupported ResNeXt depth in name '{name}'")

    # ResNet 系列
    elif "ResNet" in name or "resnet" in name.lower():
        if "50" in name:
            model = generate_resnet(50, n_input_channels=in_channels, n_classes=out_features)
        elif "101" in name:
            model = generate_resnet(101, n_input_channels=in_channels, n_classes=out_features)
        elif "18" in name:
            model = generate_resnet(18, n_input_channels=in_channels, n_classes=out_features)
        else:
            raise ValueError(f"Unsupported ResNet depth in name '{name}'")

    else:
        raise ValueError(f"Unknown model name: {name}")

    model = model.to(device)
    print(f"✅ Model built and moved to {device}")
    return model
