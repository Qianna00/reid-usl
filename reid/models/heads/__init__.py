from .baseline_head import StandardBaselineHead, StrongBaselineHead
from .contrastive_head import ContrastiveHead
from .latent_pred_head import LatentPredictHead
from .mmcl_head import MMCLHead

__all__ = [
    'ContrastiveHead', 'StandardBaselineHead', 'StrongBaselineHead',
    'MMCLHead', 'LatentPredictHead'
]
