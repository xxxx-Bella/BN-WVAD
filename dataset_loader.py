import torch
import torch.utils.data as data
import os
import numpy as np
import utils 

class XDVideo(data.DataLoader):
    def __init__(self, data_path, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path = data_path
        self.mode = mode
        self.num_segments = num_segments  # 将视频分割成多少个片段
        self.len_feature = len_feature    # 每个片段特征的长度
        
        # 读取 list 文件（videos' dir list），获取 vid_list
        split_path = os.path.join("list",'XD_{}.list'.format(self.mode))
        split_file = open(split_path, 'r',encoding="utf-8")
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        
        if self.mode == "train":
            if is_normal is True:
                self.vid_list = self.vid_list[9525:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:9525]
            else:
                assert (is_normal == None)
                print("Please sure is_normal = [True/False]")
                self.vid_list=[]
        
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid_name = self.vid_list[index][0]
        label = 0 
        if "_label_A" not in vid_name:  # label_A = normal
            label=1
        
        # load video feature
        video_feature = np.load(os.path.join(self.data_path, vid_name)).astype(np.float32)
        
        # 将视频特征分割成 num_segments 个片段，并且如果片段之间存在重叠，则计算平均值以填充特征矩阵 new_feature
        if self.mode == "train":
            new_feature = np.zeros((self.num_segments, self.len_feature)).astype(np.float32)
            sample_index = np.linspace(0, video_feature.shape[0], self.num_segments+1, dtype=np.uint16)

            for i in range(len(sample_index)-1):
                if sample_index[i] == sample_index[i+1]:
                    new_feature[i,:] = video_feature[sample_index[i],:]
                else:
                    new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1],:].mean(0)
                    
            video_feature = new_feature
        return video_feature, label    


class DroneAnomaly(data.DataLoader):
    def __init__(self, data_path, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path = data_path  # ./I3D/output/drone_anomaly
        self.mode = mode
        self.num_segments = num_segments  # 将视频分割成多少个片段
        self.len_feature = len_feature    # 每个片段特征的长度
        
        # 读取 list 文件（videos' dir list），获取 vid_list
        split_path = os.path.join("list",'DA-i3d-{}.list'.format(self.mode))
        split_file = open(split_path, 'r',encoding="utf-8")
        self.vid_list = []
        self.nvid_list = []
        self.avid_list = []
        for line in split_file:
            self.vid_list.append(line.split())  # test set
            
            if 'label_0' in line:
                self.nvid_list.append(line.split())
            elif 'label_1' in line:
                self.avid_list.append(line.split())
        split_file.close()
        # breakpoint()
        
        if self.mode == "train":
            if is_normal is True:  # normal train
                print("Loading training set... (Normal)")
                self.vid_list = self.nvid_list
            elif is_normal is False:  # abnormal train
                print("Loading training set... (Abnormal)")
                self.vid_list = self.avid_list
            else:
                assert (is_normal == None)
                print("Please sure is_normal = [True/False]")
                self.vid_list=[]
        
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):  # call this when use 'iter()' 
        data, label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid_name = self.vid_list[index][0]  # 'Bike_Roundabout_seq1_01_1_label_0.npy'
        # print(f'vid_name: {vid_name}')
        label = 0 
        if "_label_0" not in vid_name:  # label_0 = normal
            label = 1
        
        # load video feature
        video_feature = np.load(os.path.join(self.data_path, vid_name)).astype(np.float32)  # each video
        print(f'vid_name: {vid_name}. video_feature.shape = {video_feature.shape}')
        
        # # 将 each video's i3d feature 分割成 num_segments 个片段，并且如果片段之间存在重叠，则计算平均值以填充特征矩阵 new_feature
        # if self.mode == "train":
        #     print(f'Before split into num_segments:len(video_feature) = {len(video_feature)}') 
        #     # num_frames = video_feature.shape[0] = 10

        #     # 用于存储分割后的每个片段的特征
        #     new_feature = np.zeros((self.num_segments, self.len_feature)).astype(np.float32)  # (200, 2048)
        #     # 计算采样索引 (划分该视频的这些帧时的 起始和结束索引)
        #     sample_index = np.linspace(0, video_feature.shape[0], self.num_segments+1, dtype=np.uint16)  # 生成 num_segments + 1 个均匀分布的索引值
        #     # sample_index: [0*10, 1*10, ... , 9*10, 10]

        #     # 按片段分割特征，并计算平均值
        #     for i in range(len(sample_index)-1):
        #         if sample_index[i] == sample_index[i+1]:  
        #             # 如果两个相邻的 sample_index 相等，说明这个片段没有多余的帧（即该片段只对应一个帧），直接使用该帧的特征
        #             breakpoint()
        #             new_feature[i,:] = video_feature[sample_index[i],:]  # shape:(10, 2048)
        #         else:
        #             # 如果相邻的 sample_index 不相等，说明这个片段包含多个帧，此时会计算这些帧特征的平均值，作为该片段的特征
        #             new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1],:].mean(0)
        #     video_feature = new_feature
        #     print(f'After split into num_segments:len(video_feature) = {len(video_feature)}')
        return video_feature, label, 