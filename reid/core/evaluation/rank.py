import warnings

import numpy as np


def rank(dist, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Peron re-identification evaluation.

    Args:
        dist: Distances between queries and samples in gallery.
        q_pids: Identities of queries.
        g_pids: Identities of samples in gallery.
        q_camids: Camera IDs of queries.
        g_camids: Camera IDs of samples in gallery.

    Returns:
    """
    num_q, num_g = dist.shape
    if num_g < max_rank:
        max_rank = num_g
        warnings.warn(f'Number of gallery samples is quite small, got {num_g}')

    indice = np.argsort(dist, axis=1)
    matches = (g_pids[indice] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0

    for q_idx in range(num_q):
        # query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indice[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()

        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1

        # compute average precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / float(num_valid_q)

    return all_cmc, all_AP
