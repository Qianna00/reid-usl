from ..builder import DATA_SOURCES
from .market1501 import Market1501


@DATA_SOURCES.register_module()
class MSMT17(Market1501):
    """MSMT17 datasets:
        - identities: 4101
        - images: 32621 (train) + 11659 (query) + 82161 (gallery)
    """
    DATA_SOURCE = 'MSMT17'
    PATTERN = r'([-\d]+)_c(\d)'

    def _valid_instance(self, pid, camid):
        valid_flag = True
        assert 1 <= camid <= 15

        return valid_flag
