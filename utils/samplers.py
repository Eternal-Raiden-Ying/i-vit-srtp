# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.distributed as dist
import math


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    
    限制数据加载到数据集的子集进行分布式重复增强的采样器。
    它确保样本的每个增强版本将对不同的进程（GPU）可见。
    在 torch.utils.data.DistributedSampler 的基础上进行了大量修改。
    
    该采样器的作用是限制数据加载到数据集的子集以进行分布式训练，并提供重复的数据增强。
    它确保数据集中每个样本的每个增强版本都会被不同的进程（GPU）看到。

    构造函数 __init__ 接受以下参数：

    dataset: 数据集对象。
    num_replicas: 分布式训练中的进程数，默认为 None。
    rank: 当前进程的排名，默认为 None。
    shuffle: 是否对数据进行洗牌，默认为 True。
    num_repeats: 数据增强的重复次数，默认为 3。
    在 __iter__ 方法中，首先根据是否进行洗牌来确定样本的索引顺序。然后，根据 num_repeats 参数对样本索引进行重复。
    接着，通过填充额外的样本使得样本数量能够被 num_replicas 均匀整除。最后，根据当前进程的排名和总样本数来选择该进程所需的样本。

    __len__ 方法返回该采样器所提供的样本数量。

    set_epoch 方法用于设置当前 epoch 的值。
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, num_repeats: int = 3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if num_repeats < 1:
            raise ValueError("num_repeats should be greater than 0")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * self.num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))

        # add extra samples to make it evenly divisible
        indices = torch.repeat_interleave(indices, repeats=self.num_repeats, dim=0).tolist()
        padding_size: int = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])
        #return iter(indices)

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
