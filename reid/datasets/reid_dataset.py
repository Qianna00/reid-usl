import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .builder import DATASETS, build_data_source, build_pipeline


@DATASETS.register_module()
class ReIDDataset(Dataset):

    def __init__(self, data_source, pipeline=None, test_mode=False):
        self.data_source = build_data_source(data_source)
        self.DATA_SOURCE = self.data_source.DATA_SOURCE
        self.test_mode = test_mode

        if self.test_mode:
            # query + gallery
            (self.img_items, self.num_query,
             self.num_gallery) = self.data_source.prep_test()
        else:
            # train
            (self.img_items, self.pids, self.num_pids, self.cams,
             self.num_cams) = self.data_source.prep_train()

        if not self.test_mode:
            # set identity flag for the sampler
            self._set_group_flag()
            # relabel
            self.pid_dict = {p: i for i, p in enumerate(self.pids)}

        # build pipeline
        if pipeline is not None:
            self.pipeline = self.build_pipeline(pipeline)
        else:
            self.pipeline = None

    def _set_group_flag(self):
        """Set flag according to pid of image.
        """
        self.flag = np.zeros(len(self), dtype=np.int64)
        for i in range(len(self)):
            _, pid, _ = self.img_items[i]
            self.flag[i] = pid

    def __len__(self):
        return len(self.img_items)

    def build_pipeline(self, pipeline):
        self.with_camstyle = False
        # CamStyle must be the first transform if used
        if pipeline[0]['type'] == 'CamStyle':
            camstyle_cfg = pipeline.pop(0)
            camstyle_cfg['dataset'] = self
            self.with_camstyle = True
            self.camstyle_aug = build_pipeline(camstyle_cfg)

        return build_pipeline(pipeline)

    def get_sample(self, idx):
        img, pid, camid = self.img_items[idx]
        if self.with_camstyle:
            img = self.camstyle_aug(img, camid)

        img = Image.open(img)
        img = img.convert('RGB')

        return img, pid, camid

    def __getitem__(self, idx):
        img, pid, camid = self.get_sample(idx)
        img = self.pipeline(img)
        label = self.pid_dict[pid] if not self.test_mode else pid

        return dict(img=img, label=label, pid=pid, camid=camid, idx=idx)
