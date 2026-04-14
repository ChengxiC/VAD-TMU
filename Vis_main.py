import math
import numpy as np
import torch
import cv2
import os
from I3D.i3dpt import I3D
from tqdm import tqdm

from src.model import TMwVAD
from time import time
import matplotlib.pyplot as plt
from src.utils import load_annotation


def forward_batch(b_data, net):
    b_data = b_data.transpose([0, 4, 1, 2, 3])
    b_data = torch.from_numpy(b_data)  # b,c,t,h,w  # 40x3x16x224x224
    with torch.no_grad():
        b_data = b_data.cuda().float()
        b_features, _ = net(b_data, feature_layer=5)
    b_features = b_features[:, :, 0, 0, 0]
    return b_features


def load_video(path: str):
    frames = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("video capture open fail")
        exit(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("read over")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (340, 256))
        frame = np.array(frame)
        frame = frame.astype(float)
        frame = (frame * 2 / 255) - 1
        frame = frame[16:240, 58:282, :]
        frames.append(frame)
    return frames


def load_video_dir(path_dir):
    frames = []
    fs = os.listdir(path_dir)
    fs = fs.sort()
    for f in fs:
        frame = cv2.imread(os.path.join(path_dir, f))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = np.array(frame)
        frame = frame.astype(float)
        frame = (frame * 2 / 255) - 1
        frame = frame[16:240, 58:282, :]
        frames.append(frame)
    return frames


def batch_split(clipped_length, batch_size, chunk_size):
    frame_indices = []
    for i in range(clipped_length):
        frame_indices.append(
            [j for j in range(i * 16, i * 16 + chunk_size)])

    frame_indices = np.array(frame_indices)
    chunk_num = frame_indices.shape[0]
    batch_num = int(np.ceil(chunk_num / batch_size))
    frame_indices = np.array_split(frame_indices, batch_num, axis=0)
    return frame_indices, batch_num


def cv2show(video_path, score_list):
    frame_num = 1
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("video capture open fail")
        exit(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame = cap.read()
        if not ret:
            print("read over")
            break
        frame = cv2.resize(frame, (340, 256))
        frame = frame[16:240, 58:282, :]
        score = score_list[frame_num - 1]
        left_x_up = 10
        left_y_up = 10
        right_x_down = int(left_x_up + 200)
        right_y_down = int(left_y_up + 60)
        word_x = left_x_up + 10
        word_y = left_y_up + 20
        cv2.rectangle(frame, (left_x_up, left_y_up), (right_x_down, right_y_down), (55, 255, 155), 2)
        cv2.putText(frame, f'frame_num:{frame_num}', (word_x, word_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 255, 155), 1)
        if score > 0.5:
            cv2.putText(frame, f'frame_score:{score:.2f}', (word_x, word_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 155), 1)
        else:
            cv2.putText(frame, f'frame_score:{score:.2f}', (word_x, word_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 255, 155), 1)
        frame_num += 1
        cv2.imshow('det_res', frame)
        key = cv2.waitKey(25)
        if key == ord('q'):
            cap.release()
            break

def draw_score(ckpt_path: str, input_dir: str, anno_path: str, a_nums: int, n_nums: int, x_nums: int):
    """
    :param ckpt_path: the path of your best pretrained model, e.g. 'ckpt/ucf_best.pth'
    :param input_dir: the video you want to test, e.g. 'visualization/Shooting021_x264.mp4'
    :param anno_path: the path of the annotation, e.g. 'splits/UCF_Annotation.txt'
    :param a_nums, n_nums, x_nums: the params for your model
    :return: score fig and visualization
    """
    start_time = time()
    batch_size = 10
    i3d = I3D(400, modality='rgb', dropout_prob=0, name='inception')
    i3d.eval()
    i3d.load_state_dict(torch.load("I3D/model_rgb.pth"))
    i3d.cuda()

    model = TMwVAD(input_size=1024, flag="Test", a_nums=a_nums, n_nums=n_nums, x_nums=x_nums)
    model.load_state_dict(torch.load(ckpt_path))
    model.cuda()
    if os.path.isdir(input_dir):
        frames = load_video_dir(input_dir)
    else:
        frames = load_video(input_dir)
    frames_cnt = len(frames)  # 一个视频的总帧数，但是没有进行帧padding
    clipped_length = math.ceil(frames_cnt / 16)
    copy_length = (clipped_length * 16) - frames_cnt
    if copy_length != 0:
        copy_img = [frames[frames_cnt - 1]] * copy_length
        frames = frames + copy_img
    frame_indices, batch_num = batch_split(clipped_length, batch_size=batch_size, chunk_size=16)
    full_features = torch.zeros(0).cuda()
    for batch_id in tqdm(range(batch_num)):
        batch_data = np.zeros(frame_indices[batch_id].shape + (224, 224, 3))
        for i in range(frame_indices[batch_id].shape[0]):
            for j in range(frame_indices[batch_id].shape[1]):
                batch_data[i, j] = frames[frame_indices[batch_id][i][j]]
        full_features = torch.cat([full_features, forward_batch(batch_data, i3d)], dim=0)
    print(f"{input_dir} has been extracted. Its shape:{full_features.size()}")
    print("---------------------start detecting---------------------")
    full_features = full_features.unsqueeze(0)
    res = model(full_features)

    raw_scores = res["frame"].cpu().detach().numpy()
    scores = np.repeat(raw_scores, 16)[:frames_cnt]  # 重复到逐帧，并截断到原始帧数

    end_time = time()
    cost_time = end_time - start_time
    print(f"cost:{cost_time}")
    print(f"fps:{frames_cnt / cost_time}")

    # 解析 annotation
    ann_dict = load_annotation(anno_path)

    # 找到当前视频
    video_name = os.path.basename(input_dir)  # e.g. "Arson016_x264.mp4"
    if ckpt_path == 'ckpt/ubnormal_best.pth':  # 区别于UCF-Crime数据集
        video_name = video_name.split('.')[0]
        total, gt_intervals = ann_dict[video_name]
    else:
        total, gt_intervals = ann_dict[video_name]  # 异常 GT 格式为 [(start1, end1), (start2, end2)...]

    # sanity check： total 应该 == frames_cnt
    assert total == frames_cnt  # "annotation 中的 total 可能 != 实际帧数 frames_cnt"

    x = np.arange(frames_cnt)
    fig, ax = plt.subplots(figsize=(12, 5))
    # 先绘制高亮的红色背景
    for (s, e) in gt_intervals:
        ax.axvspan(s, e, color='red', alpha=0.3, zorder=0)
    # 再绘制得分折线
    # ax.plot(x, scores, color='blue', linewidth=2, label='Anomaly score', zorder=1)
    ax.plot(x, scores, color='blue')

    fig_title = os.path.basename(input_dir).split('_')[0]
    # ax.set_title(fig_title)
    ax.set_xlabel('Frame idx')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    # ax.legend(loc='upper right')

    ax.tick_params(axis='both', labelsize=16)  # 尺度的大小

    plt.tight_layout()

    out_plot = os.path.splitext(input_dir)[0] + '_score.jpg'
    plt.savefig(out_plot)
    plt.close(fig)
    print(f"Score plot saved to {out_plot}")

    save_visualization_video(input_dir, scores)
    # cv2show(input_dir, scores)
    # cv2.destroyAllWindows()

def save_visualization_video(video_path, score_list, output_path=None):
    """
    将每帧叠加分数后保存成新的视频文件。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频 {video_path}")

    # 先读一帧，算出裁剪后的尺寸
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("读取第一帧失败")
    frame = cv2.resize(frame, (340, 256))
    frame = frame[16:240, 58:282]   # 224×224, BGR
    H, W = frame.shape[:2]

    # 视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if output_path is None:
        base, _ = os.path.splitext(video_path)
        output_path = base + '_vis.mp4'

    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # 回到开头，开始写
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 裁剪到 224×224（BGR）
        frame = cv2.resize(frame, (340, 256))
        crop = frame[16:240, 58:282]

        # 叠加背景框 + 分数文字
        score = float(score_list[idx])
        cv2.rectangle(crop, (10, 10), (210, 70), (55, 255, 155), 2)
        cv2.putText(crop, f'Frame:{idx+1}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 255, 155), 1)
        col = (0, 0, 155) if score > 0.5 else (55, 255, 155)
        cv2.putText(crop, f'Score:{score:.2f}', (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # 写入
        writer.write(crop)
        idx += 1

    cap.release()
    writer.release()
    print(f"可视化视频已保存到 {output_path}")


if __name__ == "__main__":
    # draw_score('ckpt/ucf_best.pth', 'visualization/Arson016_x264.mp4', 'splits/UCF_Annotation.txt', 60, 60, 20)
    # draw_score('ckpt/ucf_best.pth', 'visualization/Shoplifting028_x264.mp4', 'splits/UCF_Annotation.txt', 60, 60, 20)
    # draw_score('ckpt/ucf_best.pth', 'visualization/RoadAccidents017_x264.mp4', 'splits/UCF_Annotation.txt', 60, 60, 20)
    draw_score('ckpt/ucf_best.pth', 'visualization/Robbery102_x264.mp4', 'splits/UCF_Annotation.txt', 60, 60, 20)
    # draw_score('ckpt/ucf_best.pth', 'visualization/Shooting015_x264.mp4', 'splits/UCF_Annotation.txt', 60, 60, 20)

    # visualization for UBnormal
    # draw_score('ckpt/ubnormal_best.pth', 'visualization/abnormal_scene_4_scenario_1.mp4', 'splits/UBnormal_annotation.txt', 10, 10, 8)
    # draw_score('ckpt/ubnormal_best.pth', 'visualization/abnormal_scene_4_scenario_1_fire.mp4', 'splits/UBnormal_annotation.txt', 10, 10, 8)
    # draw_score('ckpt/ubnormal_best.pth', 'visualization/abnormal_scene_9_scenario_2_fog.mp4', 'splits/UBnormal_annotation.txt', 10, 10, 8)
    # draw_score('ckpt/ubnormal_best.pth', 'visualization/abnormal_scene_17_scenario_1_smoke.mp4', 'splits/UBnormal_annotation.txt', 10, 10, 8)
    # draw_score('ckpt/ubnormal_best.pth', 'visualization/abnormal_scene_27_scenario_2_fire.mp4', 'splits/UBnormal_annotation.txt', 10, 10, 8)


