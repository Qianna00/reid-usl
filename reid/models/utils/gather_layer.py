import torch
import torch.distributed as dist
from torch.autograd import Function


class GatherLayer(Function):
    """Gather tensors from all processes, supporting backward propagation.
    """

    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        outputs = [
            torch.zeros_like(inputs) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(outputs, inputs)

        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grads):
        inputs, = ctx.saved_tensors
        grad_out = torch.zeros_like(inputs)
        grad_out[:] = grads[dist.get_rank()]

        return grad_out
