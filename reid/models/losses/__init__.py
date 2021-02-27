from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy)
from .triplet_loss import TripletLoss
from .utils import reduce_loss, weight_reduce_loss

__all__ = [
    'CrossEntropyLoss', 'binary_cross_entropy', 'cross_entropy', 'TripletLoss',
    'reduce_loss', 'weight_reduce_loss'
]
