import csv
import math
from pathlib import Path

test_file = 'MSAD_Testlist_original.txt'
annotation_file = 'MSAD_anomaly_annotation.csv'

# 读取注释，建立 name -> 原始 total_frames 映射
annotations = {}
with open(annotation_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        annotations[row['name']] = int(row['total frames'])

total_padded = 0
abnormal_info = []

# 遍历 test splits，只处理 abnormal-testing
with open(test_file, 'r', encoding='utf-8') as f:
    for line in f:
        path = line.strip()
        if 'abnormal-testing' not in path:
            continue

        stem = Path(path).stem               # e.g. 'Assault_2_i3d'
        video = stem.replace('_i3d', '')     # e.g. 'Assault_2'

        orig = annotations.get(video)

        padded = math.ceil(orig / 16) * 16

        abnormal_info.append((video, orig, padded))
        total_padded += padded

print(f"\n所有异常视频的帧数：{total_padded}")




