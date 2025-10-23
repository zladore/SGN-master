# SGN-master
对于示例examples文件夹下的配置文件：
    1. 主配置文件 (brats2020_config.json)
    包含所有训练参数、模型参数、数据集参数
    定义了交叉验证设置
    列出了所有训练文件名

    2. 交叉验证配置文件 (fold1.json, fold2.json等)
    从主配置文件自动生成
    每个文件包含特定fold的训练数据
    共享相同的模型和训练参数

 对于我的粒子数据，我只需要一个主配置文件，目前先不设置交叉验证

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
 python /home/hqu/PycharmProjects/3D-UNet/unet3d/scripts/train.py --config_filename /home/hqu/PycharmProjects/3D-UNet/data/particle_config.json

 python -m unet3d.scripts.train --config_filename data/create_particle_config.json

python -m unet3d.scripts.train --config_filename data/particle_config.json



"name": "BasicUnet",
        "in_channels": 4,
        "out_channels": 1,
        "spatial_dims": 3,
        "deep_supervision": false,
        "f_maps": 64,
        "num_levels": 4


我现在要做一个回归任务，首先我的每一个样本例如input：insert-5000.dat是22500行，每行有4列，分别代表粒子在x y z 方向上的速度，以及第4列可能是比例或者密度，这里我们可以先抛弃第4列，如果不抛弃第4列是不是应该要修改PaticDataset和json配置文件，
之后我们的数据采集是以网格的方式首先从x轴30个格子，之后y轴250排（一共250排），再z轴（一共3张），所以形成了3*3*250*3，但我认为第一个维度应该是每一行抛弃了最后一列产生的，之后分别是x y z，输出是每90行形成一个密度值，所以原本的22500行最后变成了250行1列，最后是计算MSE均方值误差
我给你上传了配置文件，

标准的3D卷积输入形状：(channels, depth, height, width)，对应(特征数, z, y, x)。
(4, 3, 250, 30)。这样，y维度（高度）被保留为空间维度之一，与输出（250个值对应每个y）对齐，模型更容易学习y方向的特征。

由于您要做的是回归任务而不是视频描述生成，需要进行以下调整：
    修改数据加载部分：重写特征提取流程，适配您的粒子速度数据格式
    替换模型头：将语言生成解码器改为回归输出层（输出250个密度值）
    调整损失函数：从语言模型的交叉熵损失改为MSE损失
    自定义数据划分：根据您的数据特点重新设计训练/验证/测试划分逻辑


我现在要做一个回归任务，首先我的每一个样本例如input：insert-5000.dat是22500行，每行有4列，分别代表粒子在x y z 方向上的速度，以及第4列可能是比例或者密度，这里我们可以先抛弃第4列，如果不抛弃第4列是不是应该要修改PaticDataset和json配置文件，
之后我们的数据采集是以网格的方式首先从x轴30个格子，之后y轴250排（一共250排），再z轴（一共3张），所以形成了3*3*250*3，但我认为第一个维度应该是每一行抛弃了最后一列产生的，之后分别是x y z，输出是每90行形成一个密度值，所以原本的22500行最后变成了250行1列，最后是计算MSE均方值误差
我给你上传了配置文件，

标准的3D卷积输入形状：(channels, depth, height, width)，对应(特征数, z, y, x)。
(4, 3, 250, 30)。这样，y维度（高度）被保留为空间维度之一，与输出（250个值对应每个y）对齐，模型更容易学习y方向的特征。

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
