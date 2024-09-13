import torch

def train(net, normal_loader, abnormal_loader, optimizer, criterion):
    net.train()
    net.flag = "train"
    
    # print(f'normal_loader: {normal_loader}')
    print('loading next normal_loader_iter...')
    ninput, nlabel = next(normal_loader)
    breakpoint()

    print('loading next abnormal_loader_iter...')
    ainput, alabel = next(abnormal_loader)

    _data = torch.cat((ninput, ainput), 0)
    _label = torch.cat((nlabel, alabel), 0)
    _data = _data.cuda()
    _label = _label.cuda()
    res = net(_data)
    cost, loss = criterion(res)
    optimizer.zero_grad()
    cost.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
    optimizer.step()
    
    return loss