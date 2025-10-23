import torch
from models.model_builder import build_model
import time

def print_layer_output_shape(name):
    def hook(module, input, output):
        if isinstance(output, (list, tuple)):
            shape = [o.shape for o in output]
        else:
            shape = output.shape
        print(f"🔹 {name}: output {shape}")
    return hook

def test_model_build():
    cfg = {
        "name": "3DResNeXt101",   # 🔹先用小一点的版本
        "in_channels": 4,
        "out_features": 250
    }

    print("🧩 Building model...")
    model = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("✅ Model moved to", device)

    # 注册 forward hook（打印每层输出形状）
    for name, layer in model.named_children():
        layer.register_forward_hook(print_layer_output_shape(name))

    # 用小输入测试
    x = torch.randn(1, 4, 3, 64, 64, device=device)
    print("📦 Input:", x.shape)

    print("🚀 Running forward...")
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize()
    end = time.time()

    print(f"✅ Forward OK — output shape: {y.shape}")
    print(f"⏱ Time cost: {end - start:.3f} sec")

if __name__ == "__main__":
    test_model_build()
