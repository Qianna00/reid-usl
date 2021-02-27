from ..base import BaseModel
from ..builder import REIDS, build_backbone, build_head, build_neck


@REIDS.register_module()
class Baseline(BaseModel):

    def __init__(self, backbone, neck, head=None, pretrained=None):
        super(Baseline, self).__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(Baseline, self).init_weights(pretrained=pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights()
        self.head.init_weights()

    def forward_test(self, img, **kwargs):
        x = self.backbone(img)
        z = self.neck(x)[0]

        return z

    def forward_train(self, img, label, **kwargs):
        x = self.backbone(img)
        z = self.neck(x)[0]
        losses = self.head(z, label)

        return losses
