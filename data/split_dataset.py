import os
import json
import random
import argparse


def split_dataset(input_dir, output_dir, test_ratio=0.2, output_config_filename="dataset_split.json"):
    """
    随机划分数据集为训练集和测试集

    参数:
        input_dir: 输入数据目录
        output_dir: 输出数据目录
        test_ratio: 测试集比例 (默认0.2)
        output_config_filename: 输出配置文件名
    """
    # 收集所有样本
    samples = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".dat"):
            base_name = filename.split(".")[0]
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path):
                samples.append({
                    "image": input_path,
                    "label": output_path
                })

    # 随机打乱样本
    random.shuffle(samples)

    # 计算测试集大小
    total_samples = len(samples)
    test_size = int(total_samples * test_ratio)

    # 划分数据集
    test_set = samples[:test_size]
    train_set = samples[test_size:]

    # 创建配置字典
    config = {
        "training_filenames": train_set,
        "validation_filenames": test_set
    }

    # 保存配置文件
    with open(output_config_filename, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"数据集划分完成:")
    print(f"总样本数: {total_samples}")
    print(f"训练集大小: {len(train_set)}")
    print(f"测试集大小: {len(test_set)}")
    print(f"配置文件已保存到: {output_config_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="随机划分数据集为训练集和测试集")
    parser.add_argument("--input_dir", required=True, help="输入数据目录")
    parser.add_argument("--output_dir", required=True, help="输出数据目录")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="测试集比例 (默认0.2)")
    parser.add_argument("--output_config", default="dataset_split.json", help="输出配置文件名")

    args = parser.parse_args()

    split_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        output_config_filename=args.output_config
    )