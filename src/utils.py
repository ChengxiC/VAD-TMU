import torch
import numpy as np
import random
import os


def random_perturb(feature_len, length):
    r = np.linspace(0, feature_len, length + 1, dtype=np.uint16)
    return r


def norm(data):
    l2 = torch.norm(data, p=2, dim=-1, keepdim=True)
    return torch.div(data, l2)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_annotation(anno_path):  # annotation 必须统一形式
    ann = {}
    with open(anno_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            fullpath = parts[0]
            video_name = os.path.basename(fullpath)  # e.g. "Arson016_x264.mp4"
            total = int(parts[1])
            label = int(parts[2])
            nums = list(map(int, parts[3:]))
            intervals = []
            for i in range(0, len(nums), 2):
                s, e = nums[i], nums[i+1]
                if s >= 0 and e >= 0:
                    intervals.append((s, e))

            if label == 0:
                intervals = [(0, total-1)]
            ann[video_name] = (total, intervals)
    return ann



