import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Function

from reid.utils import concat_all_gather


class MemoryLayer(Function):

    @staticmethod
    def forward(ctx, inputs, idxs, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        outputs = inputs.mm(ctx.features.t())

        if dist.is_available() and dist.is_initialized():
            all_inputs = concat_all_gather(inputs)
            all_idxs = concat_all_gather(idxs)
        else:
            all_inputs = inputs
            all_idxs = idxs
        ctx.save_for_backward(all_inputs, all_idxs)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, idxs = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        feats = ctx.momentum * ctx.features[idxs] + \
            (1. - ctx.momentum) * inputs
        ctx.features[idxs] = F.normalize(feats, dim=1)

        return grad_inputs, None, None, None
