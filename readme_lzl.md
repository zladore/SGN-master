
machine_configs配置：
     这个文件是必须的，它主要配置：
        use_multiprocessing: true - 启用多进程数据加载
        n_workers: 20 - 数据加载的工作进程数（根据你的CPU核心数设置）
        n_gpus: 2 - 使用2个GPU（V100）
        pin_memory: false - 内存锁定设置
    与CUDA的关系：
        ✅ 直接相关，这个配置会决定：
        使用哪些GPU设备
        数据如何从CPU传输到GPU
        是否使用多GPU训练
        数据加载的并行策略

训练：

`
 python /home/hqu/PycharmProjects/3D-UNet/unet3d/scripts/train.py --config_filename /home/hqu/PycharmProjects/3D-UNet/data/particle_config.json

 python -m unet3d.scripts.train --config_filename data/create_particle_config.json

python -m unet3d.scripts.train --config_filename data/particle_config.json`

我现在要做一个回归任务，首先我的每一个样本例如input：insert-5000.dat是22500行，每行有4列，分别代表粒子在x y z 方向上的速度，以及第4列可能是比例或者密度，
之后我们的数据采集是以网格的方式首先从x轴30个格子，之后y轴250排（一共250排），再z轴（一共3张），所以形成了4*3*250*3，
第一个维度就是他们的4个特征数，之后分别是x y z，输出是每90行形成一个密度值，所以原本的22500行最后变成了250行1列，最后是计算SMOOTHL1LOSS误差
同时将batch_size设置为4，
原本：
标准的3D卷积输入形状：(batch_size,channels, depth, height, width)，对应(批数，特征数, z, y, x)。
(4,4, 3, 250, 30)。这样，y维度（高度）被保留为空间维度之一，与输出（250个值对应每个y）对齐，模型更容易学习y方向的特征。
输出是（batch_size,250），每个样本的输出是250个值，每个值对应一个y方向的密度值。

目录结构：
SGN-master/
├── data/
│   ├── input/
│   ├── machine_configs/
│   ├── norm_params/
│   ├── output/
│   ├── particle_config/particle_config.json
│   └── split_dataset.py
├── filePathCSV
├── loader/
│   ├── __init__.py
│   ├── data_loader.py
│   └── ParticleDataset.py
├── models/
│   ├── __init__.py
│   ├── model_builder.py
│   ├── resnet.py
│   ├── resnext.py
│   └── test_model_build.py
├── splits/
│   └── 结果/
├── .gitignore
├── LICENSE
├── main.py
├── readme_lzl.md
├── requirements.txt
├── test.py
├── training.py
└── utils.py

下面是部分json文件代码：
class ParticleLoaderConfig这个里面，到底应该放置什么内容，我还有一个data/particle_config/particle_config.json文件，部分内容如下：
"model": {
        "name": "3DResNeXt101",
        "in_channels": 4
    },
    "optimizer": {
        "name": "Adam",
        "lr": 0.0001
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
        "input_dir": "/home/hqu/PycharmProjects/SGN-master/data/input",
        "output_dir": "/home/hqu/PycharmProjects/SGN-master/data/output",
        "data_loader": {
            "batch_size": 1,
            "shuffle": true,
            "num_workers": 2
        }
    },
        "training": {
        "batch_size": 4,
        "validation_batch_size": 4,
        "amp": false,
        "early_stopping_patience": 30,
        "n_epochs": 200,
        "save_every_n_epochs": null,
        "save_last_n_models": null,
        "save_best": true,
        "test_input": 0,
        "training_iterations_per_epoch": null
    },
    "training_filenames": [
        {
            "image": "data/input/insert-2195000.dat",
            "label": "data/output/insert-2195000.dat"
        },
        {
            "image": "data/input/insert-1340000.dat",
            "label": "data/output/insert-1340000.dat"
        },
"validation_filenames": [
        {
            "image": "data/input/insert-1950000.dat",
            "label": "data/output/insert-1950000.dat"
        },
        {
            "image": "data/input/insert-1455000.dat",
            "label": "data/output/insert-1455000.dat"
        },
"test_filenames": [
        {
            "image": "data/input/insert-2145000.dat",
            "label": "data/output/insert-2145000.dat"
        },
        {
            "image": "data/input/insert-350000.dat",
            "label": "data/output/insert-350000.dat"
        },

但是我是在晚上照的SGN处理vedio的还有词汇表之后的分类问题网络，我现在如何一步步把这个项目改成我自己的


第五版：25.11.11
修改目的：我之前用了4列数据，这次只用前3列，第4列可能是干扰数据  但是我发现只用3列并不好效果，所以又改成了4列，但其实效果不如第三版（176）  这个第五版是180多
