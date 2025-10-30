
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

本次改变：
增加对input和label归一化，同时在test时，对label反归一化实验
变更学习率，在整个模型训练过程中，学习率不应该是一成不变的   Cosine Annealing + Warmup
修改模型结构
    让模型学出一个长度为 250 的特征曲线。
    AdaptiveAvgPool3d((1, None, 1))   # 只在 z/x 上池化，保留 y
    → flatten (得到 [B, C, 250])
    → Conv1d 或 Linear 映射到 [B, 250]
    保持原始的输入维度不变，原始输入更加适配3D卷积网络，不会丢失局部空间信息，贴合原数据的三维格点（扁平化可能导致失去空间意义）


