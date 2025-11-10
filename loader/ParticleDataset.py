import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json


class ParticleDataset(Dataset):
    def __init__(self, filenames, transform=None,
                 normalize_input=True, normalize_label=True,  # ← 标签不归一化
                 input_mean=None, input_std=None,
                 label_mean=None, label_std=None):
        """
        粒子数据加载器
        Args:
            filenames: 包含输入和标签文件路径的字典列表
            transform: 数据变换
            normalize: 是否对输入进行归一化
            normalize_label: 是否对标签进行归一化（默认False）
            input_mean/std: 输入归一化参数
        """
        self.filenames = filenames
        self.transform = transform
        self.normalize_input = normalize_input
        self.normalize_label = normalize_label

        self.input_mean = np.array(input_mean) if input_mean is not None else None
        self.input_std = np.array(input_std) if input_std is not None else None
        self.label_mean = np.array(label_mean) if label_mean is not None else None
        self.label_std = np.array(label_std) if label_std is not None else None

        # 仅当 normalize_input=True 且未提供统计参数时计算
        if self.normalize_input and (self.input_mean is None or self.input_std is None):
            self.compute_normalization_parameters()
            print("√启用input归一化")
        elif self.normalize_label is None:
            print("×禁用input归一化")

        # 如果启用标签归一化且未提供参数，则计算标签统计
        if self.normalize_label and (self.label_mean is None or self.label_std is None):
            self.compute_label_normalization_parameters()
            print("√启用标签归一化")
        elif self.normalize_label is None:
            print("×禁用标签归一化")

    def __len__(self):
        return len(self.filenames)

    # ------------------------------
    # 计算输入归一化参数
    # ------------------------------
    def compute_normalization_parameters(self):
        print("计算input的标准化参数...")
        num_channels = 3
        input_sums = np.zeros(num_channels)
        input_sq_sums = np.zeros(num_channels)
        input_counts = np.zeros(num_channels)

        for filename_info in self.filenames:
            input_filename = filename_info["image"]
            if not os.path.exists(input_filename):
                continue
            # input_data = np.loadtxt(input_filename)
            # input_data = self._reshape_input_data(input_data)
            ## 改成取前3列
            input_data = np.loadtxt(input_filename)
            input_data = input_data[:, :3]  # 保留3列
            input_data = self._reshape_input_data(input_data)

            for c in range(3):
                flat = input_data[c].flatten()
                input_sums[c] += np.sum(flat)
                input_sq_sums[c] += np.sum(flat ** 2)
                input_counts[c] += len(flat)

        self.input_mean = input_sums / input_counts
        self.input_std = np.sqrt(input_sq_sums / input_counts - self.input_mean ** 2)
        self.input_std = np.maximum(self.input_std, 1e-8)
        print("✅ Normalization parameters computed for input.")

    # ------------------------------
    # 计算标签归一化参数
    # ------------------------------
    def compute_label_normalization_parameters(self):
        """计算标签归一化参数"""
        print("计算label的标准化参数...")

        all_labels = []

        for filename_info in self.filenames:
            label_path = filename_info["label"]
            if not os.path.exists(label_path):
                continue

            label_data = np.loadtxt(label_path)

            # ===============================
            # 与 __getitem__ 的处理逻辑保持一致
            # ===============================
            if label_data.ndim == 2 and label_data.shape[1] >= 2:
                label_data = label_data[:, 1]
            if len(label_data) == 22500:
                label_data = label_data.reshape(250, 90).mean(axis=1)  # -> (250,)

            all_labels.append(label_data)

        # 堆叠成 (num_samples, 250)
        all_labels = np.stack(all_labels, axis=0)

        # 对每个列（位置点）求 mean/std
        self.label_mean = np.mean(all_labels, axis=0)  # shape (250,)
        self.label_std = np.std(all_labels, axis=0)
        self.label_std = np.maximum(self.label_std, 1e-8)

        print("✅ Label normalization parameters computed.")
        print(f"Label mean shape: {self.label_mean.shape}")
        print(f"Label std  shape: {self.label_std.shape}")

    # ------------------------------
    # 数据加载与变换
    # ------------------------------
    def _reshape_input_data(self, input_data):
        """将输入(22500, 4) reshape 为 (C=4, D=3, H=250, W=30)"""
        if input_data.shape != (22500, 3):
            raise ValueError(f"Expected input shape (22500, 3), got {input_data.shape}")

        # reshape 为 (D, H, W, C) = (3,250,30,3)
        input_data = input_data.reshape(3, 250, 30, 3)

        # 转置为 (C, D, H, W)
        input_data = input_data.transpose(3, 0, 1, 2)
        return input_data

    def _normalize_data(self, data, mean, std):
        return (data - mean) / std

    def _denormalize_data(self, data, mean, std):
        """反归一化：data * std + mean"""
        return data * std + mean

    def denormalize_label(self, label_data):
        """反归一化标签数据"""
        if self.normalize_label and self.label_mean is not None and self.label_std is not None:
            label_data = self._denormalize_data(label_data, self.label_mean, self.label_std)
        return label_data

    def __getitem__(self, idx):
        info = self.filenames[idx]
        input_path = info["image"]
        label_path = info["label"]

        # --- 加载输入 ---
        input_data = np.loadtxt(input_path)
        input_data = input_data[:, :3]  # 裁剪，只保留前三列
        input_data = self._reshape_input_data(input_data)

        # --- 加载标签 ---
        output_data = np.loadtxt(label_path)

        # 如果标签是二维的，取第二列
        if output_data.ndim == 2 and output_data.shape[1] >= 2:
            output_data = output_data[:, 1]

        # 如果标签和输入一样长(22500)，则按规则聚合成250个点
        if len(output_data) == 22500:
            output_data = output_data.reshape(250, 90).mean(axis=1)

        # --- 输入归一化 ---
        if self.normalize_input and self.input_mean is not None and self.input_std is not None:
            for c in range(3):
                input_data[c] = self._normalize_data(input_data[c], self.input_mean[c], self.input_std[c])
        # --- 标签归一化 ---
        if self.normalize_label and self.label_mean is not None and self.label_std is not None:
            output_data = self._normalize_data(output_data, self.label_mean, self.label_std)

        input_tensor = torch.from_numpy(input_data).float()
        output_tensor = torch.from_numpy(output_data).float()

        return {
            "image": input_tensor,
            "label": output_tensor,
            "filename": os.path.splitext(os.path.basename(input_path))[0]
        }

    # ------------------------------
    # 工具函数
    # ------------------------------
    def get_normalization_params(self):
        """返回输入和标签归一化参数"""
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
        """保存输入归一化参数"""
        params = {
            'input_mean': self.input_mean.tolist() if self.input_mean is not None else None,
            'input_std': self.input_std.tolist() if self.input_std is not None else None,
            'label_mean': self.label_mean.tolist() if self.label_mean is not None else None,
            'label_std': self.label_std.tolist() if self.label_std is not None else None
        }
        with open(path, "w") as f:
            json.dump(params, f, indent=4)
        print(f"Normalization parameters saved to {path}")