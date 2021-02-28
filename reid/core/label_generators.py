import torch


class MPLP(object):

    def __init__(self, t=0.6):
        self.t = t

    @torch.no_grad()
    def generate_labels(self, feats, idx):
        """Generate multi labels by MPLP.
        """

        s = feats[idx].mm(feats.t())
        s_sorted, idx_sorted = torch.sort(s, dim=1, descending=True)
        multi_labels = torch.zeros(s.size(), dtype=torch.long).cuda()
        mask_num = torch.sum(s_sorted > self.t, dim=1)

        for i in range(s.shape[0]):
            topk = int(mask_num[i].item())
            topk = max(topk, 10)
            topk_idx = idx_sorted[i, :topk]

            _s = feats[topk_idx].mm(feats.t())
            _, _idx_sorted = torch.sort(_s, dim=1, descending=True)
            step = 1
            for j in range(topk):
                # print(_idx_sorted[j], idx[i])
                pos = torch.nonzero(_idx_sorted[j] == idx[i], as_tuple=False)
                pos = pos.reshape(-1).item()
                if pos > topk:
                    break
                step = max(step, j)
            step += 1
            step = min(step, mask_num[i].item())
            if step <= 0:
                continue
            multi_labels[i, idx_sorted[i, :step]] = 1

        multi_labels.scatter_(1, idx.unsqueeze(1), 1)

        return multi_labels
