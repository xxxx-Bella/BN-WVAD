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
        self.is_normal = is_normal
        
        # 读取 list 文件（videos' dir list），获取 vid_list
        split_path = os.path.join("list", f'XD_{self.mode}.list')
        split_file = open(split_path, 'r',encoding="utf-8")
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        
        if self.mode == "train":
            if self.is_normal is True:
                self.vid_list = self.vid_list[9525:]
            elif self.is_normal is False:
                self.vid_list = self.vid_list[:9525]
            else:
                assert (self.is_normal == None)
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
        self.mode = mode  # 'train', 'test'
        self.num_segments = num_segments  # 将视频分割成多少个片段
        self.len_feature = len_feature    # feature_dim 2048
        self.is_normal = is_normal
        
        # 读取 list 文件（videos' dir list），获取 vid_list
        split_path = os.path.join("list", f'DA-i3d-{self.mode}.list')
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
        
        if self.mode == "train":
            if self.is_normal is True:  # normal train
                print("Loading training set... (Normal)")
                self.vid_list = self.nvid_list
            elif self.is_normal is False:  # abnormal train
                print("Loading training set... (Abnormal)")
                self.vid_list = self.avid_list
            else:
                assert (self.is_normal == None)
                print("Please sure is_normal = [True/False]")
                self.vid_list=[]
        
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):  # call this when use 'iter()' 
        data, label = self.get_data(index)  # data.shape (10, 32, 2048)
        # data = data.transpose(1, 0, 2)
        # breakpoint()
        return data, label


    def get_label(self):
        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
        return label


    def get_data(self, index):
        vid_name = self.vid_list[index][0]  # 'Bike_Roundabout_seq1_01_1_label_0.npy'
        label = self.get_label()  # get video level label 0/1

        # label = 0 
        # if "_label_0" not in vid_name:  # label_0 = normal
        #     label = 1
        
        # load video feature
        video_feature = np.load(os.path.join(self.data_path, vid_name)).astype(np.float32)  # each video  (18, 10, 2048)
        video_feature = video_feature.transpose(1, 0, 2) # [10, 18, 2048]
        # print(f'vid_name: {vid_name}')
        # breakpoint()

        # 将 each video's i3d feature 分割成 200 个片段，并且如果片段之间存在重叠，则计算平均值以填充特征矩阵 new_feature
        if self.mode == "train":
            # print(f'Before split into 200 snippets: shape {video_feature.shape}') 
            # num_frames = video_feature.shape[0] = 10
            
            divided_features = []
            for feature in video_feature:
                # feature.shape = (18, 2048)
                feature = utils.process_feat(feature, 32)   # divide a video into 32 segments (T=32)
                divided_features.append(feature)
            
            divided_features = np.array(divided_features, dtype=np.float32)  # (10, 32, 2048)

            # new_feature = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)  # (T, 2048)
            # sample_index = np.linspace(0, video_feature.shape[0], self.num_segments+1, dtype=int)  # (start, stop, num)=(0, 18, 201) 生成 T+1 个均匀分布的索引值
            # # 按片段分割特征，并计算平均值
            # for i in range(len(sample_index)-1):  # T
            #     if sample_index[i] == sample_index[i+1]:  
            #         # 两个相邻的sample_index相等，说明该片段只对应一个帧，直接取出该帧的特征
            #         # if sample_index[i]=0, then video_feature[0, :].shape = (10, 2048)
            #         # new_feature[i, :].shape = (2048,)
            #         new_feature[i, :] = video_feature[sample_index[i], :]  # 取单帧特征
            #     else:
            #         # 相邻的sample_index不相等，说明这个片段包含多个帧，取出多帧特征并计算平均值
            #         new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1], :].mean(0)

            #         new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)  # 计算每个区间内的均值
            # video_feature = new_feature
            
            video_feature = divided_features
            # print(f'After split into 200 snippets: shape {video_feature.shape}')
        # breakpoint()
        
        return video_feature, label