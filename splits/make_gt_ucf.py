import os
import torch
import numpy as np


def get_init_gt():
    frames_gt = torch.zeros(0)
    with open('./UCF_Annotation.txt', 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            video_path = parts[0]
            vid_len = int(parts[1])
            label = parts[2]
            anomaly_times = list(map(int, parts[3:]))  # 提取异常的开始和结束帧，-1表示没有异常
            # 假设 parts[3:] 为 ['1080', '1560', '-1', '-1'], 利用map映射为[1080, 1560, -1, -1]，
            # map(int, parts[3:]) 会返回一个迭代器，将每个字符串转换为整数, list() 将 map() 返回的迭代器转化为一个列表
            assert all(anomaly_times[i] <= vid_len for i in range(4))

            if vid_len % 16 == 0:  # 整分
                vid_len = vid_len
            else:
                vid_len = vid_len - vid_len % 16
            temp = torch.zeros(vid_len)  # 初始化当前视频的GT为全0，长度为视频的帧数

            if 'Normal' in label:
                frames_gt = torch.concatenate([frames_gt, temp], dim=0)
            else:
                for i in range(0, len(anomaly_times), 2):   # 这里的步进是2，因为异常开始和异常结束是成对的. 有可能有两段或3段异常视片段.
                    start_frame = anomaly_times[i]
                    end_frame = anomaly_times[i + 1]
                    if start_frame != -1 and end_frame != -1:
                        temp[start_frame:end_frame + 1] = 1  # 左闭右开
                frames_gt = torch.concatenate([frames_gt, temp], dim=0)

    np.save(frames_gt.cpu().numpy(), 'gt_ucf.npy')


def find_vids():
    normal_vids = abnormal_vids = 0
    with open('./UCF_Annotation.txt', 'r') as file:
        for line in file:
            if 'Normal' in line.strip().split(' ')[2]:
                normal_vids += 1
            else:
                abnormal_vids += 1

    return normal_vids, abnormal_vids


def get_vid_list():
    nor_list = []
    ano_list = []
    with open('./UCF_Annotation.txt', 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            vid_path = parts[0]
            label = parts[2]
            if 'Normal' in label:
                nor_list.append(vid_path.split('/')[1].split('.')[0])
            else:
                ano_list.append(vid_path.split('/')[1].split('.')[0])
    return nor_list, ano_list


def check_list():
    _path = r'E:\toi3d'
    nor_list, ano_list = get_vid_list()
    for file in os.listdir(_path):
        file = file.split('.')[0]
        if file in nor_list:
            nor_list.remove(file)
        if file in ano_list:
            ano_list.remove(file)

    print(f'find normal videos {nor_list} not contained')
    print(f'find normal videos {ano_list} not contained')


def main_rm_e_frames():
    frame_path = 'F:\\code\\i3d_extractor\\UCF_Crime_Frames\\UCF_Crime\\Training_Normal_Videos_Anomaly'
    for folder in os.listdir(frame_path):
        video_f_path = os.path.join(frame_path, folder)

        if os.path.isdir(video_f_path):

            image_files = [f for f in os.listdir(video_f_path) if f.endswith('.jpg')]
            image_files.sort()  #

            num_images = len(image_files)

            # 如果图片数量不能被16整除，则删除最后几张图片
            if num_images % 16 != 0:

                num_to_delete = num_images % 16
                for i in range(num_to_delete):

                    image_to_delete = os.path.join(video_f_path, image_files[-1])
                    os.remove(image_to_delete)
                    image_files.pop()
                print(f"Deleted {num_to_delete} images from {folder} to make the number of images divisible by 16.")
            else:
                print(f"The number of images in {folder} is already divisible by 16. No images deleted.")


def get_test_vid_info():
    nor_vid_list = []
    ano_vid_list = []
    with open('./UCF_Test.list', 'r') as file:
        for line in file:
            parts = line.strip().split('/')
            labels = parts[1]
            file_name = parts[2].split('__')[0]
            if 'Normal' in labels:
                if file_name not in nor_vid_list:
                    nor_vid_list.append(file_name)
            else:
                if file_name not in ano_vid_list:
                    ano_vid_list.append(file_name)
    # all_vid_list = ano_vid_list.copy()
    # all_vid_list.extend(nor_vid_list)
    all_vid_list = ano_vid_list + nor_vid_list

    return ano_vid_list, nor_vid_list, all_vid_list


def make_dict(annotation_path=''):
    if annotation_path is None:
        raise 'check the path!'
    npy_dict = dict()
    try:
        with open(annotation_path, 'r') as file:
            for line in file:
                parts = line.strip().split(' ')
                video_name = parts[0].split('/')[1].split('.')[0]
                vid_len = int(parts[1])
                label = parts[2]
                anomaly_times = list(map(int, parts[3:]))  # 提取异常的开始和结束帧，-1表示没有异常
                # 假设 parts[3:] 为 ['1080', '1560', '-1', '-1'], 利用map映射为[1080, 1560, -1, -1]，
                # map(int, parts[3:]) 会返回一个迭代器，将每个字符串转换为整数, list() 将 map() 返回的迭代器转化为一个列表
                assert all(anomaly_times[i] <= vid_len for i in range(4))

                if vid_len % 16 == 0:  # 整分
                    vid_len = vid_len
                else:
                    vid_len = vid_len - vid_len % 16
                temp = np.zeros(vid_len)  # 初始化当前视频的GT为全0，长度为视频的帧数

                if 'Normal' in label:
                    npy_dict[video_name] = temp

                else:
                    for i in range(0, len(anomaly_times), 2):  # 这里的步进是2，因为异常开始和异常结束是成对的. 有可能有两段或3段异常视频.
                        start_frame = anomaly_times[i]
                        end_frame = anomaly_times[i + 1]
                        if start_frame != -1 and end_frame != -1:
                            temp[start_frame:end_frame + 1] = 1
                    npy_dict[video_name] = temp
    except Exception as e:
        print(f'error: {e}')

    return npy_dict


if __name__ == '__main__':
    # main_rm_e_frames()
    frame_gt = np.zeros(0)
    ano_frame_gt = np.zeros(0)
    nor_frame_gt = np.zeros(0)
    ano_vid_list, nor_vid_list, all_vid_list = get_test_vid_info()
    npy_dict = make_dict(annotation_path='./UCF_Annotation_old.txt')
    # 获得总的frame gt
    for i in range(len(all_vid_list)):
        if all_vid_list[i] in npy_dict.keys():
            frame_gt = np.concatenate([frame_gt, npy_dict[all_vid_list[i]]], axis=0)

    np.save(file='./gt_ucf.npy', arr=frame_gt)
    print('ground truth numpy array have been saved successfully !')






























