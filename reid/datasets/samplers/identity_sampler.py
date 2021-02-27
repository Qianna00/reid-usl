import copy
from collections import defaultdict

import numpy as np
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler

from ..builder import SAMPLERS


@SAMPLERS.register_module()
class IdentitySampler(Sampler):
    """Randomly sample P identities. For each identity, randomly sample
    K instances. Batch size is P*K.

    Args:
        dataset (Dataset): Dataset to sample.
        batch_size (int): Number of samples in a batch.
        num_instances (int, optional): Number of instances per instance,
            i.e., K. Default: 4.
    """

    def __init__(self, dataset, batch_size, num_instances=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.n_pids_per_batch = self.batch_size // self.num_instances

        self.flag = dataset.flag
        self.group_size = np.bincount(self.flag)

        self.pids = dataset.pids
        self.pid_pool = defaultdict(int)
        self.pid_indice_dict = defaultdict(list)
        for i, size in enumerate(self.group_size):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            self.pid_indice_dict[i] = indice

            # how many pieces of K instances
            self.pid_pool[i] = int(np.ceil(size / self.num_instances))

        self.num_samples = 0
        for _, val in self.pid_pool.items():
            self.num_samples += (val * self.num_instances)

    def _gen_indices(self):
        pid_pool = copy.deepcopy(self.pid_pool)
        avai_pids = list(pid_pool.keys())
        batched_pids = []

        while len(avai_pids) >= self.n_pids_per_batch:
            # randomly select P pids
            selected_pids = np.random.choice(
                avai_pids, self.n_pids_per_batch, replace=False).tolist()
            for pid in selected_pids:
                pid_pool[pid] -= 1
                if pid_pool[pid] == 0:
                    avai_pids.remove(pid)

            batched_pids.append(selected_pids)
        batched_pids = np.concatenate(batched_pids).tolist()

        avai_pid_indice = copy.deepcopy(self.pid_indice_dict)
        # extend indice of each pid
        for pid, indice in avai_pid_indice.items():
            np.random.shuffle(indice)
            num_extra = self.pid_pool[pid] * self.num_instances - len(indice)
            indice = np.concatenate(
                [indice, np.random.choice(indice, num_extra)])
            avai_pid_indice[pid] = indice.tolist()

        indices = []
        for pid in batched_pids:
            selected_indice = avai_pid_indice[pid][:self.num_instances]
            indices.append(selected_indice)
            # del selected_indice
            del avai_pid_indice[pid][:self.num_instances]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()

        return indices

    def __iter__(self):
        self.indices = self._gen_indices()
        self.num_samples = len(self.indices)

        return iter(self.indices)

    def __len__(self):
        return self.num_samples


@SAMPLERS.register_module()
class DistributedIdentitySampler(IdentitySampler):

    def __init__(self,
                 dataset,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 num_instances=4,
                 seed=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.num_replicas = num_replicas
        self.rank = rank

        batch_size = batch_size * num_replicas
        super().__init__(dataset, batch_size, num_instances)

        self.num_samples = int(np.ceil(self.num_samples / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        np.random.seed(self.seed + self.epoch)
        indices = self._gen_indices()
        self.total_size = len(indices)
        self.num_samples = self.total_size // self.num_replicas

        # subsample
        _indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(_indices) == self.num_samples

        return iter(_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
