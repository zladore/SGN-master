from __future__ import print_function, division

import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 导入你已修改好的 ParticleDataset
from loader.ParticleDataset import ParticleDataset


class ParticleDataLoader:
    """粒子数据加载器 - 管理训练、验证和测试数据加载器"""

    def __init__(self, config):
        """
        初始化数据加载器

        Args:
            config: 配置字典，包含所有配置参数
        """
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.input_dir = config['dataset']['input_dir']
        self.output_dir = config['dataset']['output_dir']

        # 构建数据加载器
        self.build()

    def _resolve_file_paths(self, file_list):
        """将相对路径解析为绝对路径"""
        resolved_list = []
        for file_info in file_list:
            # 处理输入文件路径
            if not os.path.isabs(file_info["image"]):
                image_path = os.path.join(self.input_dir, os.path.basename(file_info["image"]))
            else:
                image_path = file_info["image"]

            # 处理输出文件路径
            if not os.path.isabs(file_info["label"]):
                label_path = os.path.join(self.output_dir, os.path.basename(file_info["label"]))
            else:
                label_path = file_info["label"]

            resolved_list.append({
                "image": image_path,
                "label": label_path
            })
        return resolved_list

    def build(self):
        """构建数据加载器"""
        print("Building data loaders...")

        # --- 解析文件路径 ---
        train_files = self._resolve_file_paths(self.config.get("training_filenames", []))
        val_files = self._resolve_file_paths(self.config.get("validation_filenames", []))
        test_files = self._resolve_file_paths(self.config.get("test_filenames", []))

        if not train_files:
            raise ValueError("❌ 训练集文件列表为空，请检查配置文件")

        print(f"找到 {len(train_files)} 个训练样本, {len(val_files)} 个验证样本, {len(test_files)} 个测试样本")

        # --- 数据加载器参数 ---
        dataset_config = self.config.get("dataset", {})
        data_loader_config = dataset_config.get("data_loader", {})
        batch_size = data_loader_config.get("batch_size", 4)
        shuffle = data_loader_config.get("shuffle", True)
        num_workers = data_loader_config.get("num_workers", 0)

        training_config = self.config.get("training", {})
        training_batch_size = training_config.get("batch_size", batch_size)
        val_batch_size = training_config.get("validation_batch_size", batch_size)

        # --- 构建训练集 (只归一化输入) ---
        self.train_dataset = ParticleDataset(
            filenames=train_files,
            # normalize_input=True,  # ✅ 只归一化 input
            # normalize_label=False  # ❌ 不归一化 label
        )

        # 获取训练集的 input 归一化参数
        norm_params = self.train_dataset.get_normalization_params()

        # --- 验证集 (共享训练集的 input 归一化参数) ---
        self.val_dataset = ParticleDataset(
            filenames=val_files,
            # normalize_input=True,
            # normalize_label=False,
            **norm_params
        )

        # --- 测试集 (共享训练集的 input 归一化参数) ---
        self.test_dataset = ParticleDataset(
            filenames=test_files,
            # normalize_input=True,
            # normalize_label=False,
            **norm_params
        )

        # --- DataLoader ---
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=training_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

        print(f"✅ 数据加载器创建完成:")
        print(f"  训练集: {len(self.train_dataset)} 个样本, 批次大小: {training_batch_size}")
        print(f"  验证集: {len(self.val_dataset)} 个样本, 批次大小: {val_batch_size}")
        print(f"  测试集: {len(self.test_dataset)} 个样本, 批次大小: {val_batch_size}")

        # --- 保存 input 的归一化参数 ---
        norm_params_path = os.path.join(dataset_config.get("input_dir", "."), 'normalization_params.json')
        self.train_dataset.save_normalization_params(norm_params_path)

    def collate_fn(self, batch):
        """自定义批次处理函数"""
        # ✅ Dataset 已改为返回 images / labels / filenames
        valid_batch = [item for item in batch if not torch.isnan(item["image"]).any()]

        if len(valid_batch) == 0:
            # 如果整个批次都有错误，返回空批次
            return {
                "image": torch.empty(0, 4, 3, 250, 30),
                "label": torch.empty(0, 250),
                "filename": []
            }

        # 提取有效数据
        images = torch.stack([item["image"] for item in valid_batch])
        labels = torch.stack([item["label"] for item in valid_batch])
        filenames = [item["filename"] for item in valid_batch]

        return {
            "image": images,
            "label": labels,
            "filename": filenames
        }

    def get_loaders(self):
        """获取所有数据加载器"""
        return self.train_loader, self.val_loader, self.test_loader

    def get_datasets(self):
        """获取所有数据集"""
        return self.train_dataset, self.val_dataset, self.test_dataset


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


if __name__ == "__main__":
    # 测试代码
    config_path = "/home/hqu/PycharmProjects/SGN-master/data/particle_config/particle_config.json"
    config = load_config(config_path)

    loader = ParticleDataLoader(config)
    train_loader, val_loader, test_loader = loader.get_loaders()

    # 测试一个批次
    for batch_idx, batch in enumerate(train_loader):
        print(f"批次 {batch_idx}:")
        print(f"  输入形状: {batch['image'].shape}")   # (batch_size, 4, 3, 250, 30)
        print(f"  标签形状: {batch['label'].shape}")   # (batch_size, 250)
        print(f"  文件名: {batch['filename'][:2]}...")  # 显示前两个文件名
        break  # 只测试第一个批次
