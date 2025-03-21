import pdb
import numpy as np
import torch
import torch.utils.data as data
import utils
import time
import wandb
from options import *  # init_args
from train import train
from losses import LossComputer
from test import test
from models import WSAD

from dataset_loader import XDVideo, DroneAnomaly
from tqdm import tqdm

localtime = time.localtime()
time_ymd = time.strftime("%Y-%m-%d", localtime)
time_hms = time.strftime("%H:%M:%S", localtime)


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    wandb.init(
        project="BN-WVAD",
        name=args.version,
        config={
            'optimization:lr': args.lr[0],
            'optimization:iters': args.num_iters,
            'dataset:dataset': args.dataset,
            'model:kernel_sizes': args.kernel_sizes,
            'model:channel_ratios': args.ratios,
            'triplet_loss:abn_ratio_sample': args.ratio_sample,
            'triplet_loss:abn_ratio_batch': args.ratio_batch,
        },
        settings=wandb.Settings(code_dir=os.path.dirname(os.path.abspath(__file__))),
        save_code=True,
    )

    worker_init_fn = None

    if args.seed >= 0:
        utils.set_seed(args.seed)
        worker_init_fn = np.random.seed(args.seed)
    
    net = WSAD(args.len_feature, flag = "train", args=args)
    net = net.cuda()
    if args.dataset == 'XDViolence':
        data_name = 'xd'
        normal_train_loader = data.DataLoader(
            XDVideo(data_path = args.data_path, mode = 'train', num_segments = args.num_segments, len_feature = args.len_feature, is_normal = True),
            batch_size = args.batch_size,
            shuffle = True, num_workers = args.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
        abnormal_train_loader = data.DataLoader(
            XDVideo(data_path = args.data_path, mode='train', num_segments = args.num_segments, len_feature = args.len_feature, is_normal = False),
            batch_size = args.batch_size,
            shuffle = True, num_workers = args.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
        test_loader = data.DataLoader(
            XDVideo(data_path = args.data_path, mode = 'test', num_segments = args.num_segments, len_feature = args.len_feature),
            batch_size = 5,
            shuffle = False, num_workers = args.num_workers,
            worker_init_fn = worker_init_fn)
    elif args.dataset == 'DroneAnomaly':
        data_name = 'da'
        train_nset = DroneAnomaly(data_path = args.data_path, mode = 'train', num_segments = args.num_segments, len_feature = args.len_feature, is_normal = True)
        train_aset = DroneAnomaly(data_path = args.data_path, mode='train', num_segments = args.num_segments, len_feature = args.len_feature, is_normal = False)
        test_set = DroneAnomaly(data_path = args.data_path, mode = 'test', num_segments = args.num_segments, len_feature = args.len_feature)
        print(f'train_Nset: {len(train_nset)}')
        print(f'train_Aset: {len(train_aset)}')
        print(f'test_set: {len(test_set)}')

        normal_train_loader = data.DataLoader(train_nset,
                                            batch_size = args.batch_size,
                                            shuffle = True, num_workers = args.num_workers,
                                            worker_init_fn = worker_init_fn, drop_last = True)
        abnormal_train_loader = data.DataLoader(train_aset,
                                            batch_size = args.batch_size,
                                            shuffle = True, num_workers = args.num_workers,
                                            worker_init_fn = worker_init_fn, drop_last = True)
        test_loader = data.DataLoader(test_set,
                                    batch_size = 1,
                                    shuffle = False, num_workers = args.num_workers,
                                    worker_init_fn = worker_init_fn)

    print(f'train_Nloader: {len(normal_train_loader)}')
    print(f'train_Aloader: {len(abnormal_train_loader)}')
    print(f'test_loader: {len(test_loader)}')
    # breakpoint()

    args.log_path = os.path.join(args.log_path, time_ymd, data_name, args.version)
    args.model_path = os.path.join(args.model_path, time_ymd, data_name, args.version)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    test_info = {'epoch': [], 'AUC': [], 'AP': []}
    best_auc = 0 
    criterion = LossComputer()
    optimizer = torch.optim.Adam(net.parameters(), lr = args.lr[0],
        betas = (0.9, 0.999), weight_decay = args.weight_decay) 
    best_scores = {
        'best_AUC': -1,
        'best_AP': -1,
    }
    
    metric = test(net, test_loader, test_info, 0)

    for epoch in tqdm(
            range(1, args.num_iters + 1),
            total = args.num_iters,
            dynamic_ncols = True
        ):
        if epoch > 1 and args.lr[epoch - 1] != args.lr[epoch - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr[epoch - 1]
        if (epoch - 1) % len(normal_train_loader) == 0:
            # print('-----------------\nGeting normal_loader_iter...\n')
            normal_loader_iter = iter(normal_train_loader)

        if (epoch - 1) % len(abnormal_train_loader) == 0:
            # print('-----------------\nGeting abnormal_loader_iter...\n')
            abnormal_loader_iter = iter(abnormal_train_loader)
        
        # print('-----------------\nStart training...\n')
        losses = train(net, normal_loader_iter, abnormal_loader_iter, optimizer, criterion)
        wandb.log(losses, step=epoch)

        if epoch % args.plot_freq == 0 and epoch > 0:
            print('-----------------\nStart testing...\n')
            metric = test(net, test_loader, test_info, epoch)

            if test_info["AP"][-1] > best_scores['best_AP']:
                utils.save_best_record(test_info, os.path.join(args.log_path, f"{data_name}_best_record_{args.seed}.txt"))

                torch.save(net.state_dict(), os.path.join(args.model_path, f"{data_name}_best_{args.seed}.pkl"))
            
            for n, v in metric.items():
                best_name = 'best_' + n
                best_scores[best_name] = v if v > best_scores[best_name] else best_scores[best_name]

        wandb.log(metric, step=epoch) 
        wandb.log(best_scores, step=epoch) 