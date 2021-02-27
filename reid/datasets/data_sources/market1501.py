import os.path as osp
import re
from glob import glob

from ..builder import DATA_SOURCES
from .reid_data_source import ReIDDataSource


@DATA_SOURCES.register_module()
class Market1501(ReIDDataSource):
    """Market1501 dataset:
        - identities: 1501
        - images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    DATA_SOURCE = 'Market1501'
    PATTERN = r'([-\d]+)_c(\d)'

    def __init__(self, data_root, **kwargs):
        self.data_root = data_root

        self.train_dir = osp.join(self.data_root, 'bounding_box_train')
        self.query_dir = osp.join(self.data_root, 'query')
        self.gallery_dir = osp.join(self.data_root, 'bounding_box_test')

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)

        super(Market1501, self).__init__(train, query, gallery)

    def _valid_instance(self, pid, camid):
        valid_flag = True

        if pid == -1:
            valid_flag = False  # junk image
        else:
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6

        return valid_flag

    def process_dir(self, dir_path):
        img_paths = glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(self.PATTERN)

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if not self._valid_instance(pid, camid):
                continue
            data.append((img_path, pid, camid))

        return data
