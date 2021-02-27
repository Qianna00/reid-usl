import torch

from .builder import DATASETS
from .reid_dataset import ReIDDataset


@DATASETS.register_module()
class BYOLDataset(ReIDDataset):

    def __init__(self, data_source, pipeline1, pipeline2):
        super(BYOLDataset, self).__init__(data_source, pipeline=None)
        self.pipeline1 = self.build_pipeline(pipeline1)
        self.pipeline2 = self.build_pipeline(pipeline2)

    def __getitem__(self, idx):
        img1, _, _ = self.get_sample(idx)
        img2, _, _ = self.get_sample(idx)
        img1 = self.pipeline1(img1)
        img2 = self.pipeline2(img2)
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)

        return dict(img=img_cat, idx=idx)
