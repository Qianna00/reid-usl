from ..builder import REIDS
from .baseline import Baseline


@REIDS.register_module()
class BoT(Baseline):

    def forward_train(self, img, label, **kwargs):
        x = self.backbone(img)
        f_t, f_i = self.neck(x, loc='both')
        losses = self.head(f_t, f_i, label)

        return losses
