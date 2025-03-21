import torch
import numpy as np
import random
import os

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def save_best_record(test_info, file_path):
    fo = open(file_path, "a")
    fo.write("Epoch: {}\n".format(test_info["epoch"][-1]))
    fo.write("AUC: {:.4f}\n".format(test_info["AUC"][-1]))
    fo.write("AP: {:.4f}\n".format(test_info["AP"][-1]))
    fo.write("\n")  # 换行，方便下次记录


def process_feat(feat, length):
    '''对输入的特征 feat 进行重新采样，将其处理为固定长度. 
    length - 输出的目标长度
    feat: (18, 2048)
    '''
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    # 均匀划分特征的区间 (32, 2048)
    indexs = np.linspace(0, len(feat), length+1, dtype=int)  # (start, stop, num)=(0, 37, 33)  #   将0~18个划分为33个
    # breakpoint()
    for i in range(length):  # 32
        if indexs[i]!=indexs[i+1]:
            new_feat[i,:] = np.mean(feat[indexs[i]:indexs[i+1],:], 0)  # 计算每个区间内的均值
        else:
            new_feat[i,:] = feat[indexs[i],:]
    return new_feat
