import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===============================================================
# 1️⃣ 读取测试结果
# ===============================================================
result_path = "results/test_results_full.csv"
if not os.path.exists(result_path):
    raise FileNotFoundError(f"❌ 找不到测试结果文件: {result_path}")

df = pd.read_csv(result_path)
print(f"✅ 已加载结果文件，共 {len(df)} 个样本，{len(df.columns)} 列")

# 获取预测与标签矩阵
pred_cols = [c for c in df.columns if c.startswith("pred_")]
label_cols = [c for c in df.columns if c.startswith("label_")]
preds = df[pred_cols].values
labels = df[label_cols].values
filenames = df["filename"].values

# ===============================================================
# 2️⃣ 计算每个样本的指标
# ===============================================================
mse_list = ((preds - labels) ** 2).mean(axis=1)
mean_pred = preds.mean(axis=1)
mean_label = labels.mean(axis=1)

print(f"�� 样本平均 MSE: {mse_list.mean():.6f}")

# ===============================================================
# 3️⃣ 绘制前若干样本的预测 vs 真实曲线
# ===============================================================
num_show = min(6, len(preds))  # 前6个样本
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(num_show, 1, figsize=(10, 2.5 * num_show), sharex=True)

if num_show == 1:
    axes = [axes]

for i in range(num_show):
    axes[i].plot(preds[i], label="Prediction", color='tab:blue', linewidth=1.2)
    axes[i].plot(labels[i], label="Ground Truth", color='tab:orange', linestyle='--', linewidth=1)
    mse_i = mse_list[i]
    axes[i].set_title(f"{os.path.basename(filenames[i])} | MSE={mse_i:.6f}", fontsize=10)
    axes[i].grid(True, linestyle="--", alpha=0.4)
    if i == 0:
        axes[i].legend(loc="upper right", fontsize=9)

plt.tight_layout()
os.makedirs("results/plots", exist_ok=True)
save_path = "results/plots/test_samples_curve.png"
plt.savefig(save_path, dpi=300)
plt.close()
print(f"�� 已保存前 {num_show} 个样本的曲线图: {save_path}")

# ===============================================================
# 4️⃣ 绘制样本平均预测 vs 平均真实值 散点图
# ===============================================================
plt.figure(figsize=(6, 6))
plt.scatter(mean_label, mean_pred, color='tab:blue', alpha=0.7)
lims = [min(mean_label.min(), mean_pred.min()), max(mean_label.max(), mean_pred.max())]
plt.plot(lims, lims, 'r--', label='y=x')
plt.xlabel("True Mean Value", fontsize=12)
plt.ylabel("Predicted Mean Value", fontsize=12)
plt.title("Predicted vs True Mean (per sample)", fontsize=13)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
save_path = "results/plots/pred_vs_true_mean.png"
plt.savefig(save_path, dpi=300)
plt.close()
print(f"�� 已保存平均预测 vs 平均真实值散点图: {save_path}")

# ===============================================================
# 5️⃣ 绘制样本 MSE 排序条形图
# ===============================================================
sorted_idx = np.argsort(mse_list)
plt.figure(figsize=(10, 4))
plt.bar(range(len(mse_list)), mse_list[sorted_idx], color='tab:purple')
plt.xlabel("Sample Index (sorted by MSE)", fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.title("Sample-wise MSE Distribution", fontsize=13)
plt.grid(True, linestyle="--", alpha=0.3, axis='y')
plt.tight_layout()
save_path = "results/plots/mse_distribution.png"
plt.savefig(save_path, dpi=300)
plt.close()
print(f"�� 已保存样本 MSE 分布图: {save_path}")

print("\n✅ 可视化完成！生成的图像文件位于 results/plots/")
