from ..builder import DATA_SOURCES
from .market1501 import Market1501


@DATA_SOURCES.register_module()
class DukeMTMC(Market1501):
    """DukeMTMC-reID dataset:
        - identities: 1404
        - images: 16522 (train) + 2228 (query) + 17661 (gallery)
    """
    DATA_SOURCE = 'DukeMTMC-reID'
    PATTERN = r'([-\d]+)_c(\d)'

    def _valid_instance(self, pid, camid):
        valid_flag = True
        assert 1 <= camid <= 8

        return valid_flag
