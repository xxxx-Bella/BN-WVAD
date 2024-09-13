import torch
from options import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

def get_predicts(test_loader, net):
    print('-----------------\nGeting test_load_iter...\n')
    load_iter = iter(test_loader)
    frame_predict = []
    test_num_frames = 0
    # print('\npredict...\n')
    # print(f'len(test_loader.dataset) = {len(test_loader.dataset)}')
    
    for i in range(len(test_loader.dataset)):  # 72  # ori: len(test_loader.dataset)//5
        print('loading next test_load_iter...')
        _data, _label = next(load_iter)  # _label.shape = torch.Size([1]), _data.shape = torch.Size([1, 37, 10, 2048])  # (bs, num_frame, n_crop, feature_dim)
        # breakpoint()
        test_num_frames += _data.shape[1]
        
        _data = _data.cuda()
        _label = _label.cuda()
        res = net(_data)  
        
        a_predict = res.cpu().numpy().mean(0)   

        fpre_ = np.repeat(a_predict, 16)
        frame_predict.append(fpre_)

        # print(f'fpre_.shape:{fpre_.shape}')  # (160,)
    
    print(f'test_num_frames: {test_num_frames}')  # 2322
    frame_predict = np.concatenate(frame_predict, axis=0)
    return frame_predict


def get_metrics(frame_predict, frame_gt):
    metrics = {}
    
    # fpr,tpr,_ = roc_curve(frame_gt, frame_predict)

    # frame_gt = frame_gt[:len(frame_predict)]
    print(f'len(frame_gt)={len(frame_gt)}, len(frame_predict)={len(frame_predict)}')
    fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
    metrics['AUC'] = auc(fpr, tpr)
    
    precision, recall, th = precision_recall_curve(frame_gt, frame_predict)
    metrics['AP'] = auc(recall, precision)
    
    return metrics

def test(net, test_loader, test_info, step, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))
        
        # print('-----------------\nGeting test_load_iter...\n')
        # load_iter = iter(test_loader)
        frame_predicts = get_predicts(test_loader, net)

        frame_gt = np.load("list/gt-da.npy")  # copy from ./RTFM/list  # len(frame_gt)=37152

        metrics = get_metrics(frame_predicts, frame_gt)
        
        test_info['step'].append(step)
        for score_name, score in metrics.items():
            metrics[score_name] = score * 100
            test_info[score_name].append(metrics[score_name])

        return metrics
