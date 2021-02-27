import torch
import torch.nn.functional as F


def cosine_dist(x, y):
    """Compute cosine distances between features.

    Args:
        x (Tensor): Features.
        y (Tensor): Features.

    Returns:
        Tensor: Distances.
    """
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    dist = 2. - 2. * torch.mm(x, y.t())

    return dist


def euclidean_dist(x, y):
    """Compute euclidean distances between features.

    Args:
        x (Tensor): Features.
        y (Tensor): Features.

    Returns:
        Tensor: Distances.
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    return dist
