import platform
import random
from functools import partial

import numpy as np
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchvision.transforms import Compose

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
DATA_SOURCES = Registry('data_source')
PIPELINES = Registry('pipeline')
SAMPLERS = Registry('sampler')


def build_pipeline(pipeline):
    if isinstance(pipeline, list):
        pipeline = [build_from_cfg(_pipe, PIPELINES) for _pipe in pipeline]
        pipeline = Compose(pipeline)
    elif isinstance(pipeline, dict):
        pipeline = build_from_cfg(pipeline, PIPELINES)
    else:
        f'pipeline should be a list or dict, but got: {type(pipeline)}'

    return pipeline


def build_sampler(sampler,
                  dataset,
                  batch_size,
                  shuffle=False,
                  distributed=True):
    """Build a sampler."""
    if sampler is not None:
        # Specified sampler
        sampler_type = sampler['type']
        assert sampler_type == 'IdentitySampler', \
            f'only `IdentitySampler` is supported, but got: {sampler_type}'
        if distributed:
            sampler['type'] = f'Distributed{sampler_type}'

        sampler['dataset'] = dataset
        sampler['batch_size'] = batch_size
        sampler = build_from_cfg(sampler, SAMPLERS)
    else:
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            sampler = RandomSampler(dataset) if shuffle else None

    return sampler


def build_data_source(cfg):
    return build_from_cfg(cfg, DATA_SOURCES)


def build_dataset(cfg):
    return build_from_cfg(cfg, DATASETS)


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     sampler=None,
                     seed=None,
                     **kwargs):
    """Build a PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int, optional): Number of GPUs. Only used in non-distributed
            training. Default: 1.    random.seed(worker_seed)
        dist (bool, optional): Distributed training/test or not.
            Default: True.
        shuffle (bool, optional): Whether to shuffle the data at every epoch.
            Default: True.
        sampler (dict, optional): Config used to build sampler.
            Default: None.
        seed (int, optional): Random seed. Default: None.
        kwargs: any keyword argument to beataset, used to initialize
            DataLoader.

    Returns:
        DataLoader: PyTorch DataLoader
    """
    rank, world_size = get_dist_info()
    if dist:
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    sampler = build_sampler(
        sampler, dataset, batch_size, shuffle=shuffle, distributed=dist)

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
