import warnings

import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info
from mmcv.utils import print_log
from tabulate import tabulate

from ..distances import cosine_dist, euclidean_dist
from .extract import multi_gpu_extract, single_gpu_extract

try:
    from ._rank import ranking
except ImportError:
    from .rank import ranking
    warnings.warn('Cython version of ranking method is not imported.')


class Evaluator:

    def __init__(self,
                 metric='euclidean',
                 feat_norm=False,
                 max_rank=50,
                 topk=(1, 5, 10),
                 progress=True,
                 **kwargs):
        assert metric in ('cosine', 'euclidean'), \
            f'distance {metric} is unsupported'
        self.metric = cosine_dist if metric == 'cosine' else euclidean_dist
        self.feat_norm = feat_norm

        self.max_rank = max_rank
        self.topk = topk
        self.progress = progress

    def extract(self, model, data_loader):
        if dist.is_available() and dist.is_initialized():
            return multi_gpu_extract(
                model, data_loader, progress=self.progress)
        else:
            return single_gpu_extract(
                model, data_loader, progress=self.progress)

    def _eval(self, results, num_query, logger=None):
        feats = results['feats']
        pids = results['pids']
        camids = results['camids']

        if self.feat_norm:
            feats = F.normalize(feats, dim=1)

        # query
        query_feats = feats[:num_query]
        query_pids = pids[:num_query]
        query_camids = camids[:num_query]

        # gallery
        gallery_feats = feats[num_query:]
        gallery_pids = pids[num_query:]
        gallery_camids = camids[num_query:]

        # compute distances
        dist = self.metric(query_feats, gallery_feats)

        # convert to np
        cmc, all_AP = ranking(dist.numpy(), query_pids.numpy(),
                              gallery_pids.numpy(), query_camids.numpy(),
                              gallery_camids.numpy(), self.max_rank)
        mAP = np.mean(all_AP)

        _results = {}
        for k in self.topk:
            _results[f'Rank-{k}'] = cmc[k - 1]
        _results['mAP'] = mAP

        return _results

    def eval(self, model, data_loader, logger=None):
        dataset = data_loader.dataset
        assert dataset.test_mode  # test mode
        num_query = dataset.num_query

        # extract
        results = self.extract(model, data_loader)
        rank, _ = get_dist_info()
        if rank == 0:
            _results = self._eval(results, num_query, logger=logger)
            self.report_results(
                _results, dataset_name=dataset.DATA_SOURCE, logger=logger)

    def report_results(self, results, dataset_name='none', logger=None):
        headers = ['dataset']
        headers.extend(list(results.keys()))

        _results = [dataset_name]
        _results += [val for _, val in results.items()]

        table = tabulate([_results],
                         tablefmt='github',
                         floatfmt='.2%',
                         headers=headers,
                         numalign='left')
        print_log('\n====> Results:\n' + table, logger=logger)
