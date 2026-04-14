import numpy as np


# def find_ano_frames(path: str):
#     ano_frames = 0
#     nor_frames = 0
#     with open(path, 'r') as file:
#         for line in file:
#             parts = line.strip().split(' ')
#             fname = parts[0]
#             t_frames = int(parts[1])
#             if 'abnormal' in fname or 'a' in fname:
#                 ano_frames += t_frames
#             elif 'normal' in fname or 'n' in fname:
#                 nor_frames += t_frames
#     print(f'{path}: normal frames: {nor_frames}\nanomalous frames:{ano_frames}')


def make_gt_padding(annotation_file: str, output_file: str):
    all_gt = []
    total_ano_frames = 0
    total_norm_frames = 0

    with open(annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            video = parts[0]
            orig_len = int(parts[1])
            label = int(parts[2])
            times = list(map(int, parts[3:]))

            gt = np.zeros(orig_len, dtype=np.uint8)
            if label != 0:
                for i in range(0, len(times), 2):
                    start, end = times[i], times[i+1]
                    gt[start:end+1] = 1

            pad_len = (16 - (orig_len % 16)) % 16       # 计算需要 pad 的长度
            if pad_len > 0:
                last_label = gt[-1]
                pad_array = np.full(pad_len, last_label, dtype=np.uint8)
                gt = np.concatenate([gt, pad_array], axis=0)

            n_frames = gt.shape[0]
            if label != 0:
                total_ano_frames += n_frames
                print(f"[ANOMALY] {video}: {n_frames} frames")
            else:
                total_norm_frames += n_frames
                print(f"[NORMAL] {video}: {n_frames} frames")

            all_gt.append(gt)

    frame_gt = np.concatenate(all_gt, axis=0)
    np.save(output_file, frame_gt)
    print(f"anomaly videos(含padding): {total_ano_frames}")
    print(f"normal videos(含padding): {total_norm_frames}")
    print(f"total: {frame_gt.shape[0]}")

if __name__ == '__main__':

    # make_gt_padding(annotation_file='UBnormal_annotation.txt', output_file='gt_ubnormal.npy')
    # anomaly visualization(含padding): 70896
    # normal visualization(含padding): 24224
    # total: 95120

    make_gt_padding(annotation_file='SDN_annotation.txt', output_file='../frame_gt/gt_sdn.npy')
    # anomaly videos(含padding): 34032
    # normal videos(含padding): 17696
    # total: 51728

