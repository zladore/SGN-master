import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json


class ParticleDataset(Dataset):
    def __init__(self, filenames, transform=None,
                 normalize_input=True, normalize_label=True,
                 input_mean=None, input_std=None,
                 label_mean=None, label_std=None):
        """
        粒子数据加载器（使用 4 个输入特征）
        输入格式 (22500, 4): [vx, vy, vz, density/ratio]
        reshape → (C=4, D=3, H=250, W=30)
        """
        self.filenames = filenames
        self.transform = transform
        self.normalize_input = normalize_input
        self.normalize_label = normalize_label

        self.input_mean = np.array(input_mean) if input_mean is not None else None
        self.input_std = np.array(input_std) if input_std is not None else None
        self.label_mean = np.array(label_mean) if label_mean is not None else None
        self.label_std = np.array(label_std) if label_std is not None else None

        # -------- 输入归一化参数 --------
        if self.normalize_input and (self.input_mean is None or self.input_std is None):
            self.compute_normalization_parameters()
            print("√ 启用输入归一化 (4 通道)")
        else:
            print("× 未启用输入归一化")

        # -------- 标签归一化参数 --------
        if self.normalize_label and (self.label_mean is None or self.label_std is None):
            self.compute_label_normalization_parameters()
            print("√ 启用标签归一化")
        else:
            print("× 未启用标签归一化")

    def __len__(self):
        return len(self.filenames)

    # ============================================================
    # ✅ 计算输入归一化参数 (4 通道)
    # ============================================================
    def compute_normalization_parameters(self):
        print("计算 input 的标准化参数...")
        num_channels = 4

        input_sums = np.zeros(num_channels)
        input_sq_sums = np.zeros(num_channels)
        input_counts = np.zeros(num_channels)

        for filename_info in self.filenames:
            input_filename = filename_info["image"]
            if not os.path.exists(input_filename):
                continue

            input_data = np.loadtxt(input_filename)  # shape (22500, 4)
            input_data = self._reshape_input_data(input_data)  # shape (4,3,250,30)

            for c in range(4):
                flat = input_data[c].flatten()
                input_sums[c] += np.sum(flat)
                input_sq_sums[c] += np.sum(flat**2)
                input_counts[c] += len(flat)

        self.input_mean = input_sums / input_counts
        self.input_std = np.sqrt(input_sq_sums / input_counts - self.input_mean**2)
        self.input_std = np.maximum(self.input_std, 1e-8)

        print("✅ input_mean:", self.input_mean)
        print("✅ input_std :", self.input_std)

    # ============================================================
    # ✅ 计算标签归一化参数
    # ============================================================
    def compute_label_normalization_parameters(self):
        print("计算 label 的标准化参数...")

        all_labels = []

        for filename_info in self.filenames:
            label_path = filename_info["label"]
            if not os.path.exists(label_path):
                continue

            label_data = np.loadtxt(label_path)

            if label_data.ndim == 2 and label_data.shape[1] >= 2:
                label_data = label_data[:, 1]

            if len(label_data) == 22500:
                label_data = label_data.reshape(250, 90).mean(axis=1)

            all_labels.append(label_data)

        all_labels = np.stack(all_labels, axis=0)

        self.label_mean = np.mean(all_labels, axis=0)
        self.label_std = np.std(all_labels, axis=0)
        self.label_std = np.maximum(self.label_std, 1e-8)

        print("✅ label_mean shape:", self.label_mean.shape)
        print("✅ label_std  shape:", self.label_std.shape)

    # ============================================================
    # ✅ reshape 函数 (4 通道)
    # ============================================================
    def _reshape_input_data(self, input_data):
        """
        输入应为 (22500, 4)
        输出 (C=4, D=3, H=250, W=30)
        """
        if input_data.shape != (22500, 4):
            raise ValueError(f"Expected input shape (22500, 4), got {input_data.shape}")

        # reshape → (D,H,W,C)
        input_data = input_data.reshape(3, 250, 30, 4)

        # transpose → (C,D,H,W)
        input_data = input_data.transpose(3, 0, 1, 2)
        return input_data

    # ============================================================
    # ✅ __getitem__
    # ============================================================
    def __getitem__(self, idx):
        info = self.filenames[idx]
        input_path = info["image"]
        label_path = info["label"]

        # -------- 输入 --------
        input_data = np.loadtxt(input_path)
        input_data = self._reshape_input_data(input_data)

        # -------- 标签 --------
        output_data = np.loadtxt(label_path)

        if output_data.ndim == 2 and output_data.shape[1] >= 2:
            output_data = output_data[:, 1]

        if len(output_data) == 22500:
            output_data = output_data.reshape(250, 90).mean(axis=1)

        # -------- 输入归一化 --------
        if self.normalize_input:
            for c in range(4):
                input_data[c] = (input_data[c] - self.input_mean[c]) / self.input_std[c]

        # -------- 标签归一化 --------
        if self.normalize_label:
            output_data = (output_data - self.label_mean) / self.label_std

        return {
            "image": torch.from_numpy(input_data).float(),
            "label": torch.from_numpy(output_data).float(),
            "filename": os.path.splitext(os.path.basename(input_path))[0]
        }

    # ============================================================
    # ✅ 工具函数
    # ============================================================
    def get_normalization_params(self):
        return {
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'label_mean': self.label_mean,
            'label_std': self.label_std
        }

    @classmethod
    def load_normalization_params(cls, path):
        with open(path, "r") as f:
            params = json.load(f)
        for k, v in params.items():
            if isinstance(v, list):
                params[k] = np.array(v)
        return params

    def save_normalization_params(self, path):
        params = {
            'input_mean': self.input_mean.tolist(),
            'input_std': self.input_std.tolist(),
            'label_mean': self.label_mean.tolist(),
            'label_std': self.label_std.tolist()
        }
        with open(path, "w") as f:
            json.dump(params, f, indent=4)
        print(f"Normalization parameters saved to {path}")
