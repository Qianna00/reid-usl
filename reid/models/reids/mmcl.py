import torch
import torch.nn.functional as F

from ..builder import REIDS
from ..utils import MemoryLayer
from .baseline import Baseline


@REIDS.register_module()
class MMCL(Baseline):
    """MMCL.

    Implementation of ``Unsupervised Person Re-identification via Mulit-label
    Classification <https://arxiv.org/abs/2004.09228>``.

    Args:
        backbone (dict): Config dict of backbone.
        neck (dict): Config dict of neck.
        head (dict): Config dict of head.
        pretrained (str, optional): Path to pre-trained weights.
            Default: None.
        feat_dim (int, optional): Dimension of features in the memory.
            Default: 2048.
        memory_isze (int, optional): Number of instances in the memory,
            which equals to the dataset size. Default: 65536.
        base_momentum (float, optional): Base momentum coefficient.
            Default: 0.5.
        t (float, optional): Threshold used for label predicting.
            Default: 0.6.
        start_epoch (int, optional): Epoch that starts using MPLP for labels.
            Default: 5.
    """

    def __init__(self,
                 backbone,
                 neck,
                 head,
                 pretrained=None,
                 feat_dim=2048,
                 memory_size=65536,
                 base_momentum=0.5,
                 t=0.6,
                 start_epoch=5):
        super(MMCL, self).__init__(backbone, neck, head, pretrained=pretrained)
        # register memory
        self.register_buffer('memory', torch.zeros(memory_size, feat_dim))

        self.base_momentum = base_momentum
        self.momentum = base_momentum  # updated by hook
        self.t = t
        self.start_epoch = start_epoch

        self._epoch = None

    def set_epoch(self, epoch):
        """Set runnint epoch by hook.
        """
        self._epoch = epoch

    @torch.no_grad()
    def generate_label(self, idx):
        """Generate multi labels by MPLP.

        Args:
            idx (Tensor): Indices of images in the dataset.

        Returns:
            Tensor: Multi-labels.
        """
        feats = self.memory.clone().detach()

        s = feats[idx].mm(feats.t())
        s_sorted, idx_sorted = torch.sort(s, dim=1, descending=True)
        multi_labels = torch.zeros(s.size(), dtype=torch.long).cuda()
        mask_num = torch.sum(s_sorted > self.t, dim=1)

        for i in range(s.shape[0]):
            topk = int(mask_num[i].item())
            topk = max(topk, 10)
            topk_idx = idx_sorted[i, :topk]

            _s = feats[topk_idx].mm(feats.t())
            _, _idx_sorted = torch.sort(_s, dim=1, descending=True)
            step = 1
            for j in range(topk):
                # print(_idx_sorted[j], idx[i])
                pos = torch.nonzero(_idx_sorted[j] == idx[i], as_tuple=False)
                pos = pos.reshape(-1).item()
                if pos > topk:
                    break
                step = max(step, j)
            step += 1
            step = min(step, mask_num[i].item())
            if step <= 0:
                continue
            multi_labels[i, idx_sorted[i, :step]] = 1

        multi_labels.scatter_(1, idx.unsqueeze(1), 1)

        return multi_labels

    def forward_train(self, img, idx, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Inputed images of shape (N, C, H, W).
            idx (Tensor): Indices of images in the dataset.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.backbone(img)
        z = self.neck(x)[0]
        z = F.normalize(z, dim=1)
        logits = MemoryLayer.apply(z, idx, self.memory, self.momentum)

        with torch.no_grad():
            if self._epoch >= self.start_epoch:
                labels = self.generate_label(idx)
            else:
                labels = logits.new_zeros(logits.size(), dtype=torch.long)
                labels.scatter_(1, idx.unsqueeze(1), 1)

        # compute loss
        losses = self.head(logits, labels)

        return losses

    def forward_test(self, img, **kwargs):
        x = self.backbone(img)
        z = self.neck(x, loc='both')[0]  # MMCL uses pooling-5 feature

        return z
