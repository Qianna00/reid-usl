from tabulate import tabulate


class ReIDDataSource(object):
    """Base dataset class for person re-ID.
    """
    DATA_SOURCE = None
    PATTERN = None

    def __init__(self, train, query, gallery):
        self.train = train
        self.query = query
        self.gallery = gallery

    def _parse_data(self, data):
        pids = set()
        cams = set()
        for _, pid, camid in data:
            pids.add(pid)
            cams.add(camid)

        return pids, len(pids), cams, len(cams)

    def _print_info(self, info):
        headers = ['subset', '# images', '# pids', '# cameras']
        table = tabulate(
            info, tablefmt='github', headers=headers, numalign='left')
        print(f'\n====> Loaded {self.DATA_SOURCE}: \n' + table)

    def prep_train(self):
        pids, n_pids, cams, n_cams = self._parse_data(self.train)
        info = [['train', len(self.train), n_pids, n_cams]]
        self._print_info(info)

        return self.train, pids, n_pids, cams, n_cams

    def prep_test(self):
        _, n_q_pids, _, n_q_cams = self._parse_data(self.query)
        _, n_g_pids, _, n_g_cams = self._parse_data(self.gallery)
        # yapf: disable
        info = [
            ['query', len(self.query), n_q_pids, n_q_cams],
            ['gallery', len(self.gallery), n_g_pids, n_q_cams]
        ]
        # yapf: enable
        self._print_info(info)

        return self.query + self.gallery, len(self.query), len(self.gallery)
