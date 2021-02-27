# Unsupervised Person Re-identification via Multi-label Classification


## Introduction

```
@inproceedings{wang2020unsupervised,
  title={Unsupervised person re-identification via multi-label classification},
  author={Wang, Dongkai and Zhang, Shiliang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={10981--10990},
  year={2020}
}
```

## Results

### Market1501

|                            | mAP  | R1   |
| -------------------------- | :--: | :--: |
| MMCL w/o CamStyle (paper)  | 35.3 | 66.6 |
| MMCL w/o CamStyle (pool-5) | 30.5 | 61.0 |
| MMCL w/o CamStyle          | 33.7 | 62.2 |
| MMCL (paper)               | 45.5 | 80.3 |
| MMCL (pool-5 )             | 49.4 (49.1/49.2/50.0) | 79.0 (79.4/78.5/79.2) |
| MMCL                       | 50.1 (49.9/49.9/50.6) | 79.3 (79.3/79.0/79.6) |

**Notes:**

- pool-5: the original MMCL uses pooling-5 feature (before BN layer)

### DukeMTMC-reID

|                           | mAP  | R1   |
| ------------------------- | :--: | :--: |
| MMCL (paper)              | 40.2 | 65.2 |
| MMCL w/o CamStyle (paper) | 36.3 | 58.0 |

### MSMT17

|                           | mAP  | R1   |
| ------------------------- | :--: | :--: |
| MMCL (paper)              | 11.2 | 35.4 |
