import json
import os
import glob


def create_particle_config_complete():
    config = {
        "model": {
            "name": "3DUnet",
            "in_channels": 3,
            "out_channels": 1,
            "spatial_dims": 3,
            "deep_supervision": False,
            "f_maps": 64,
            "num_levels": 4
        },
        "optimizer": {
            "name": "Adam",
            "lr": 0.001
        },
        "loss": {
            "name": "MSELoss"
        },
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "patience": 10,
            "factor": 0.5,
            "min_lr": 1e-08
        },
        "dataset": {
            "name": "ParticleDataset",
            "input_dir": "/home/hqu/PycharmProjects/3D-UNet/data/input",
            "output_dir": "/home/hqu/PycharmProjects/3D-UNet/data/output",
            "data_loader": {
                "batch_size": 1,
                "shuffle": True,
                "num_workers": 2
            }
        },
        "training": {
            "batch_size": 1,
            "validation_batch_size": 1,
            "amp": False,
            "early_stopping_patience": 10,
            "n_epochs": 200,
            "save_every_n_epochs": None,
            "save_last_n_models": None,
            "save_best": True
        },
        "training_filenames": []
    }

    # 获取输入和输出目录
    input_dir = "/home/hqu/PycharmProjects/3D-UNet/data/input"
    output_dir = "/home/hqu/PycharmProjects/3D-UNet/data/output"

    # 查找输入文件
    input_files = glob.glob(os.path.join(input_dir, "insert-*.dat"))

    # 提取文件名并排序
    def extract_number(filepath):
        # 从完整路径中提取文件名
        filename = os.path.basename(filepath)
        # 从 "insert-5000.dat" 中提取 5000
        return int(filename.split('-')[1].split('.')[0])

    sorted_input_files = sorted(input_files, key=extract_number)

    training_filenames = []
    for input_file in sorted_input_files:
        filename = os.path.basename(input_file)
        # 构建输出文件路径
        output_file = os.path.join(output_dir, filename)

        # 检查输出文件是否存在
        if os.path.exists(output_file):
            training_filenames.append({
                "image": f"data/input/{filename}",
                "label": f"data/output/{filename}"
            })
        else:
            print(f"警告: 输出文件 {output_file} 不存在，跳过此样本")

    config["training_filenames"] = training_filenames

    # 保存配置文件
    with open("particle_config.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"Generated configuration with {len(training_filenames)} training files")

    # 显示文件范围
    if training_filenames:
        first_file = training_filenames[0]["image"]
        last_file = training_filenames[-1]["image"]
        print(f"Files range: {first_file} to {last_file}")

        # 显示一些样本文件
        print("\nSample files:")
        for i in range(0, min(10, len(training_filenames))):
            print(f"  {training_filenames[i]['image']} -> {training_filenames[i]['label']}")
        if len(training_filenames) > 10:
            print("  ...")
            for i in range(-5, 0):
                print(f"  {training_filenames[i]['image']} -> {training_filenames[i]['label']}")


if __name__ == "__main__":
    create_particle_config_complete()