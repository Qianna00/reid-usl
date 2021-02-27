import os.path as osp
import pickle
import tempfile
import time
from collections import defaultdict

import mmcv
import torch
from mmcv.runner import get_dist_info
from mmcv.utils import ProgressBar


def single_gpu_extract(model, data_loader, progress=False):
    model.eval()

    results = defaultdict(list)
    dataset = data_loader.dataset
    if progress:
        prog_bar = ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            feats = model(mode='test', **data)

        results['feats'].append(feats.cpu())
        results['pids'].append(data['pid'])
        results['camids'].append(data['camid'])

        if progress:
            prog_bar.update(data['img'].shape[0])

    for key, val in results.items():
        results[key] = torch.cat(val, dim=0)

    return results


def multi_gpu_extract(model,
                      data_loader,
                      tmpdir=None,
                      gpu_collect=False,
                      progress=False):
    model.eval()

    results = defaultdict(list)
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0 and progress:
        prog_bar = ProgressBar(len(dataset))

    time.sleep(2)  # prevent deadlock problem in some cases
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            feats = model(mode='test', **data)

        results['feats'].append(feats.cpu())
        results['pids'].append(data['pid'])
        results['camids'].append(data['camid'])

        if rank == 0 and progress:
            prog_bar.update(data['img'].shape[0] * world_size)

    for key, val in results.items():
        results[key] = torch.cat(val, dim=0)

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)

    return results


def collect_results_cpu(results_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist')
            tmpdir = tempfile.mkdtemp(dir='.dist')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        torch.distributed.broadcast(dir_tensor, src=0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part results to the dir
    mmcv.dump(results_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    torch.distributed.barrier()

    # collect all parts
    if rank == 0:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{rank}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = _sort_results(part_list, size)

        return ordered_results


def collect_results_gpu(results_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(results_part)),
        dtype=torch.uint8,
        device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    torch.distributed.gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    torch.distributed.gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = _sort_results(part_list, size)

        return ordered_results


def _sort_results(part_list, size):
    raise NotImplementedError
