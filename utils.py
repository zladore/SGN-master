import csv
import random
from functools import partialmethod

import torch
import numpy as np


class AverageMeter(object):
    """计算并存储当前值和平均值（用于记录loss、指标等）"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    """简单日志记录器，将训练过程中的指标写入到文件"""

    def __init__(self, path, header):
        self.log_file = path.open('w')
        self.logger = csv.writer(self.log_file, delimiter='\t')
        self.logger.writerow(header)
        self.header = header

    def __del__(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values, f"Missing column '{col}' in logged values"
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(outputs, targets):
    """分类任务中使用的准确率函数，如果你是回归任务可忽略或改为计算RMSE"""
    with torch.no_grad():
        mse = torch.mean((outputs - targets) ** 2).item()
        rmse = mse ** 0.5
        return rmse  # 对回归任务返回RMSE误差


def worker_init_fn(worker_id):
    """DataLoader中保证每个worker初始化不同随机种子"""
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)
    np.random.seed((torch_seed + worker_id) % (2**32 - 1))


def get_lr(optimizer):
    """获取当前学习率（适配多参数组优化器）"""
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)
    return max(lrs)


def partialclass(cls, *args, **kwargs):
    """用于偏函数化类构造（相当于partial，但针对类）"""

    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return PartialClass
