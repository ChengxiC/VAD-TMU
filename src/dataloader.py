from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from src.utils import setup_seed


class UCFCrime(Dataset):
    def __init__(self, mode, num_segments, seed=-1, is_normal=None):
        super().__init__()
        if seed >= 0:
            setup_seed(seed)
        self.mode = mode
        self.num_segments = num_segments
        split_path = os.path.join('splits', f'UCF_{self.mode}.list')

        with open(split_path, 'r') as file:
            self.vid_list = []
            for line in file:
                self.vid_list.append(line.split())

        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[8100:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:8100]
            else:
                assert (is_normal is None)
                print("Please sure is_normal=[True/False]")
                self.vid_list = []

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        
        if self.mode == "Test":
            data, label, name = self.get_data(index)
            return data, label, name
        else:
            data, label = self.get_data(index)
            return data, label

    def get_data(self, index):
        vid_info = self.vid_list[index][0]  
        name = vid_info.split("/")[-1].split("_x264")[0]
        video_feature = np.load(vid_info).astype(np.float32)   

        if "Normal" in vid_info.split("/")[-1]:
            label = 0
        else:
            label = 1
        if self.mode == "Train":
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype=np.int32)
            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i, :] = np.mean(video_feature[r[i]:r[i+1], :], 0)
                else:
                    new_feat[i:i+1, :] = video_feature[r[i]:r[i]+1, :]
            video_feature = new_feat
        if self.mode == "Test":
            return video_feature, label, name      
        else:
            return video_feature, label


class UB(Dataset):
    def __init__(self, mode, num_segments, seed=-1, is_normal=None):

        if seed >= 0:
            setup_seed(seed)
        self.mode = mode
        self.num_segments = num_segments
        split_path = os.path.join('splits', 'UBnormal_{}.txt'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())

        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[1860:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:1860]
            else:
                assert (is_normal is None)
                print("Please sure is_normal=[True/False]")
                self.vid_list = []

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):

        if self.mode == "Test":
            data, label, name = self.get_data(index)
            return data, label, name
        else:
            data, label = self.get_data(index)
            return data, label

    def get_data(self, index):

        vid_info = self.vid_list[index][0]
        vid_info = os.path.normpath(vid_info)
        name = os.path.splitext(os.path.basename(vid_info))[0]

        video_feature = np.load(vid_info).astype(np.float32)

        prefix = name.split('_')[0]  # “abnormal” or “normal”
        if prefix == 'abnormal':
            label = 1
        elif prefix == 'normal':
            label = 0

        if self.mode == "Train":
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype=np.int32)
            for i in range(self.num_segments):
                if r[i] != r[i + 1]:
                    new_feat[i, :] = np.mean(video_feature[r[i]:r[i + 1], :], 0)
                else:
                    new_feat[i:i + 1, :] = video_feature[r[i]:r[i] + 1, :]
            video_feature = new_feat
        if self.mode == "Test":
            return video_feature, label, name
        else:
            return video_feature, label


class MSAD(Dataset):
    def __init__(self, mode, num_segments, seed=-1, is_normal=None):
        if seed >= 0:
            setup_seed(seed)
        self.mode = mode
        self.num_segments = num_segments
        split_path = os.path.join('splits', 'MSAD_{}.txt'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[3600:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:3600]
            else:
                assert (is_normal is None)
                print("Please sure is_normal=[True/False]")
                self.vid_list = []

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):

        if self.mode == "Test":
            data, label, name = self.get_data(index)
            return data, label, name
        else:
            data, label = self.get_data(index)
            return data, label

    def get_data(self, index):

        vid_info = self.vid_list[index][0]
        vid_info = os.path.normpath(vid_info)
        name = os.path.splitext(os.path.basename(vid_info))[0]

        video_feature = np.load(vid_info).astype(np.float32)

        if "normal" in vid_info:
            label = 0
        else:
            label = 1
        if self.mode == "Train":
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype=np.int32)
            for i in range(self.num_segments):
                if r[i] != r[i + 1]:
                    new_feat[i, :] = np.mean(video_feature[r[i]:r[i + 1], :], 0)
                else:
                    new_feat[i:i + 1, :] = video_feature[r[i]:r[i] + 1, :]
            video_feature = new_feat
        if self.mode == "Test":
            return video_feature, label, name
        else:
            return video_feature, label

class SDN(Dataset):
    def __init__(self, mode, num_segments, seed=-1, is_normal=None):

        if seed >= 0:
            setup_seed(seed)
        self.mode = mode
        self.num_segments = num_segments
        split_path = os.path.join('splits', f'SDN_{self.mode}.txt')
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())

        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[1580:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:1580]
            else:
                assert (is_normal is None)
                print("Please sure is_normal=[True/False]")
                self.vid_list = []

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        if self.mode == "Test":
            data, label, name = self.get_data(index)
            return data, label, name
        else:
            data, label = self.get_data(index)
            return data, label

    def get_data(self, index):

        vid_info = self.vid_list[index][0]
        vid_info = os.path.normpath(vid_info)
        name = os.path.splitext(os.path.basename(vid_info))[0]

        video_feature = np.load(vid_info).astype(np.float32)

        prefix = name.split('_')[0]  # “abnormal” or “normal”
        if prefix == 'normal' or prefix == 'Normal':
            label = 0
        else:
            label = 1

        if self.mode == "Train":
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype=np.int32)
            for i in range(self.num_segments):
                if r[i] != r[i + 1]:
                    new_feat[i, :] = np.mean(video_feature[r[i]:r[i + 1], :], 0)
                else:
                    new_feat[i:i + 1, :] = video_feature[r[i]:r[i] + 1, :]
            video_feature = new_feat
        if self.mode == "Test":
            return video_feature, label, name
        else:
            return video_feature, label



