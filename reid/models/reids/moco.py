import torch
import torch.nn as nn
import torch.nn.functional as F

from reid.utils import concat_all_gather
from ..base import BaseModel
from ..builder import REIDS, build_backbone, build_head, build_neck


@REIDS.register_module()
class MoCo(BaseModel):
    """MoCo.

    Implementation of ``Momentum Contrast for Unsupervised Visual
    Representation Learning``.

    Args:
        queue_size (int): Number of negative keys. Default: 65536.
        feat_dim (int): Dimension of feature. Default: 128.
        momentum (float): Momentum coefficient for updating encoder.
            Default: 0.999.
    """

    def __init__(self,
                 backbone,
                 neck,
                 head,
                 pretrained=None,
                 queue_size=65536,
                 feat_dim=128,
                 momentum=0.999):
        super(MoCo, self).__init__()
        self.encoder_q = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.encoder_k = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        self.head = build_head(head)
        self.init_weights(pretrained=pretrained)

        self.queue_size = queue_size
        self.feat_dim = feat_dim
        self.momentum = momentum

        # create the queue
        self.register_buffer('queue', torch.randn(feat_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    def init_weights(self, pretrained=None):
        """Initialize weights.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        super().init_weights(pretrained=pretrained)
        self.encoder_q[0].init_weights(pretrained=pretrained)
        self.encoder_q[1].init_weights(init_linear='kaiming')
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Batch shuffle, for making use of BN.
        *** Only support DDP model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle.
        *** Only support DDP model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_train(self, img, **kwargs):
        """Forward during training.

        Args:
            img (Tensor): Input of two concatenated images of shape
                (N, 2, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, f'img must have 5 dims, but got: {img.dim()}'
        im_q = img[:, 0, ...].contiguous()
        im_k = img[:, 1, ...].contiguous()
        # compute queue features
        q = self.encoder_q(im_q)[0]  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)[0]  # keys: NxC
            k = F.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits: Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        losses = self.head(l_pos, l_neg)

        return losses

    def forward_test(self, img, **kwargs):
        return self.encoder_q(img)[0]
