import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import paddle
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
from apex import amp  # uncomment lines related with `amp` to use apex
from torch.optim import lr_scheduler
# from dataset import MyDataset
from Dateset import MyDataset
# from model import DeepDTAF, test
from cnn import CNN, test

print(sys.argv)

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# 显示进度条
SHOW_PROCESS_BAR = True
# data_path = r'E:\Python\TrAdaboost\canruntr\MyTrAdaBoost\labelData\AllDATA\TransferData\labelcnn'
# data_path = r'F:\test'
data_path = r'E:\Python\TrAdaboost\canruntr\MyTrAdaBoost\labelData\AllDATA\TransferData\origin'
# data_path = r'E:\Python\TrAdaboost\canruntr\MyTrAdaBoost\labelData\AllDATA\TransferData\dta'
seed = np.random.randint(33927, 33928)  ##random
path = Path(f'../runs/CNN_{datetime.now().strftime("%Y%m%d%H%M%S")}_{seed}')
device = torch.device("cuda:0")  # or torch.device('cpu')
# device = torch.device('cpu')

max_seq_len = 600  # 蛋白质序列长度
max_pkt_len = 63  # pocket长度
max_smi_len = 150  # SMILES长度

batch_size = 16
n_epoch = 20
interrupt = None
save_best_epoch = 13  # when `save_best_epoch` is reached and the loss starts to decrease, save best model parameters
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑

# GPU uses cudnn as backend to ensure repeatable by setting the following (in turn, use advances function to speed up training)
# 保证卷积操作的一致性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed)
np.random.seed(seed)  # 固定随机种子

writer = SummaryWriter(path)  # 生成runs目录
f_param = open(path / 'parameters.txt', 'w')

print(f'device={device}')
print(f'seed={seed}')
print(f'write to {path}')
f_param.write(f'device={device}\n'
              f'seed={seed}\n'
              f'write to {path}\n')

print(f'max_seq_len={max_seq_len}\n'
      f'max_pkt_len={max_pkt_len}\n'
      f'max_smi_len={max_smi_len}')

f_param.write(f'max_seq_len={max_seq_len}\n'
              f'max_pkt_len={max_pkt_len}\n'
              f'max_smi_len={max_smi_len}\n')

assert 0 < save_best_epoch < n_epoch

# model = DeepDTAF()
# Modeling...
model = CNN()
model = model.to(device)
print(model)
f_param.write('model: \n')
f_param.write(str(model) + '\n')
f_param.close()


def training(model, training_loader, loss_function, optim):
    """Training script for DeepDTA backbone model.

    Args:
        model: DeepDTA backbone model.
        training_loader: Dataloader of training set.
        optim: Optimizer.

    Returns:
        res_loss: Ouput training loss.
    """
    model.train()
    n = 0
    for _, (seq, smile, label, pkt, y) in enumerate(training_loader):
        for seq, smile, label, pkt, y in training_loader:
            seq = seq.to(device)
            smile = smile.to(device)
            y = y.to(device)
            pkt = pkt.to(device)
        output = model(seq, smile, pkt)
        output = output.to(torch.float32)
        loss = loss_function(output.view(-1), y.view(-1))
        loss = loss.to(torch.float32)

        optim.zero_grad()
        loss.backward()
        optim.step()
        res_loss = loss.cpu().detach().numpy()
        n += 1
        lr_scheduler.step()
    return res_loss, n


# print(len)
data_loaders = {phase_name:
                    DataLoader(MyDataset(data_path, phase_name, max_seq_len=max_seq_len,
                                         max_pkt_len=max_pkt_len, max_smi_len=max_smi_len),
                               batch_size=batch_size,
                               shuffle=True)
                for phase_name in ['training', 'validation', 'test']}

loss_function = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=1e-2)

lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# fp16
# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

start = datetime.now()
print('start at ', start)

best_epoch = -1
best_val_loss = 100000000
loss_train = []
print("===============Go for Training===============")
# for epoch in range(n_epoch):
#     # tbar = tqdm(enumerate(data_loaders['training']), disable=not SHOW_PROCESS_BAR, total=len(data_loaders['training']))
#     train_loss, n = training(model, data_loaders['training'], loss_function, optimizer)
#     loss_train.append(train_loss)
#     print(f' * Train Epoch {epoch} Loss={train_loss.item() / n:.3f}')
#
#     for _p in ['training', 'validation']:
#         performance = test(model, data_loaders[_p], loss_function, device, False)
#         for i in performance:
#             writer.add_scalar(f'{_p} {i}', performance[i], global_step=epoch)
#         if _p == 'validation' and epoch >= save_best_epoch and performance['loss'] < best_val_loss:
#             best_val_loss = performance['loss']
#             best_epoch = epoch
#             torch.save(model.state_dict(), path / 'best_model.pt')
#
# model.load_state_dict(torch.load(path / 'best_model.pt'))
with open(path / 'result.txt', 'w') as f:
    f.write(f'best model found at epoch NO.{best_epoch}\n')
    for _p in ['training', 'validation', 'test', ]:
        performance = test(model, data_loaders[_p], loss_function, device, SHOW_PROCESS_BAR)
        f.write(f'{_p}:\n')
        print(f'{_p}:')
        for k, v in performance.items():
            f.write(f'{k}: {v}\n')
            print(f'{k}: {v}\n')
        f.write('\n')
        print()

print('training finished')

end = datetime.now()
print(loss_train)
print('end at:', end)
print('time used:', str(end - start))

if __name__ == '__main__':
    import torch
    print(torch.version.cuda)

