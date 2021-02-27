import cython
import numpy as np

cimport numpy as np


cpdef rank(float[:,:] distmat, 
           long[:] q_pids,
           long[:] g_pids,
           long[:] q_camids,
           long[:] g_camids,
           long max_rank):
    
    cdef long num_q = distmat.shape[0]
    cdef long num_g = distmat.shape[1]

    if num_g < max_rank:
        max_rank = num_g
        print(f'Note: number of gallery samples is quite small, got: {num_g}')

    cdef long[:,:] indices = np.argsort(distmat, axis=1)
    cdef long[:,:] matches = (np.asarray(g_pids)[np.asarray(indices)] == np.asarray(q_pids)[:, np.newaxis]).astype(np.int64)

    cdef float[:,:] all_cmc = np.zeros((num_q, max_rank), dtype=np.float32)
    cdef float[:] all_AP = np.zeros(num_q, dtype=np.float32)

    cdef long num_valid_q = 0
    cdef long valid_index = 0

    cdef long q_idx
    cdef long q_pid
    cdef long q_camid
    cdef long g_idx

    cdef long[:] order = np.zeros(num_g, dtype=np.int64)
    cdef long keep
    
    cdef float[:] raw_cmc = np.zeros(num_g, dtype=np.float32)
    cdef float[:] cmc = np.zeros(num_g, dtype=np.float32)

    cdef long num_g_real
    cdef long rank_idx
    cdef unsigned long meet_condition

    cdef float num_rel
    cdef float[:] tmp_cmc = np.zeros(num_g, dtype=np.float32)
    cdef float tmp_cmc_summ

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        for g_idx in range(num_g):
            order[g_idx] = indices[q_idx, g_idx]
        num_g_real = 0
        meet_condition = 0

        # remove gallery samples that have the same pid and camid with query
        for g_idx in range(num_g):
            if (g_pids[order[g_idx]] != q_pid) or (g_camids[order[g_idx]] != q_camid):
                raw_cmc[num_g_real] = matches[q_idx][g_idx]
                num_g_real += 1
                # this condition is true if query appear in gallery
                if matches[q_idx][g_idx] > 1e-31:
                    meet_condition = 1
        
        if not meet_condition:
            continue

        # compute cmc
        function_cumsum(raw_cmc, cmc, num_g_real)
        
        for g_idx in range(num_g_real):
            if cmc[g_idx] > 1:
                cmc[g_idx] = 1
        
        for rank_idx in range(max_rank):
            all_cmc[q_idx, rank_idx] = cmc[rank_idx]
        
        num_valid_q += 1

        # compute AP
        function_cumsum(raw_cmc, tmp_cmc, num_g_real)
        num_real = 0 
        tmp_cmc_sum = 0

        for g_idx in range(num_g_real):
            tmp_cmc_sum += (tmp_cmc[g_idx] / (g_idx + 1.)) * raw_cmc[g_idx]
            num_real += raw_cmc[g_idx]
        all_AP[valid_index] = tmp_cmc_sum / num_real
        valid_index += 1

    # compute average cmc
    cdef float[:] avg_cmc = np.zeros(max_rank, dtype=np.float32)
    for rank_idx in range(max_rank):
        for q_idx in range(num_q):
            avg_cmc[rank_idx] += all_cmc[q_idx, rank_idx]
        avg_cmc[rank_idx] /= num_valid_q

    return np.asarray(avg_cmc).astype(np.float32), np.asarray(all_AP[:valid_index])
        

cdef void function_cumsum(cython.numeric[:] src, cython.numeric[:] dst, long n):
    cdef long i
    dst[0] = src[0]
    for i in range(1, n):
        dst[i] = src[i] + dst[i - 1]
