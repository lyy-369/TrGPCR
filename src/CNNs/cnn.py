import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src import metrics
from src.metrics import *
from Dateset import *

class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        smi_embed_size = 128
        seq_embed_size = 128
        PT_FEATURE_SIZE = 25
        SM_FEATURE_SIZE = 64
        PKT_FEATURE_SIZE = 4
        seq_oc = 128
        pkt_oc = 128
        smi_oc = 128

        # (N, H=32, L)
        conv_smi = []
        conv_seq = []

        ic = seq_embed_size
        # self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size)  # (N, *, H_{in}) -> (N, *, H_{out})
        self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size,bias=False) # 这边出问题了，seq[9,1000,25]-->seq[9,1000,25,128]
        self.smi_embed = nn.Linear(SM_FEATURE_SIZE, smi_embed_size,bias=False)  # (N, *, H_{in}) -> (N, *, H_{out})
        # self.pkt_embed = nn.Embedding(PKT_FEATURE_SIZE, seq_embed_size)
        self.pkt_embed = nn.Linear(PKT_FEATURE_SIZE,smi_embed_size,bias=False)
        self.cat_dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(seq_oc + smi_oc + pkt_oc, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64, 1),
            nn.PReLU())
        for oc in [32, 64, smi_oc]:
            conv_smi.append(nn.Conv1d(ic, oc, 3))  # (N,C,L)
            conv_smi.append(nn.BatchNorm1d(oc))
            conv_smi.append(nn.PReLU())
            ic = oc
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)  # (N,oc)

        # self.Conv1d1 = nn.Conv1d(128, 32, 3)
        # self.BatchNorm1d1 = nn.BatchNorm1d(32)
        # self.PRelu = nn.PReLU()
        # self.AdaptiveMaxPool1d = nn.AdaptiveMaxPool1d(1)
        # self.Squeeze = Squeeze()

    def forward(self, seq, smi,pkt):
        seq_embed = self.seq_embed(seq)  # (N,L,32)
        seq_embed = torch.transpose(seq_embed, 1, 2)
        seq_conv = self.conv_smi(seq_embed)  # (N,128)
        # output = self.Conv1d1(seq_embed)
        # output = self.BatchNorm1d1(output)
        # output = self.PRelu(output)
        # output = self.AdaptiveMaxPool1d(output)
        # seq_conv = self.Squeeze(output)

        smi_embed = self.smi_embed(smi)  # (N,L,32)
        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)  # (N,128)

        pkt_embed = self.pkt_embed(pkt)  # (N,L,32)
        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_embed = self.conv_smi(pkt_embed)  # (N,128)

        cat = torch.cat([seq_conv, smi_conv,pkt_embed], dim=1)  # (N,128*3)
        cat = self.cat_dropout(cat)
        output = self.classifier(cat)
        return output
def test(model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for idx, (seq,smile,label,pkt,y) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            seq = seq.to(device)
            smile = smile.to(device)
            y = y.to(device)
            pkt = pkt.to(device)

            y_hat = model(seq,smile,pkt)

            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))
    print("outputs",outputs)
    print("targets",targets)

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    test_loss /= len(test_loader.dataset)

    evaluation = {
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
    }

    return evaluation

# if __name__ == '__main__':
#     seq = torch.rand([2,1000,25])
#     smile = torch.rand([2,150,64])
#     pkt = torch.rand([2,10]).long()
#     model = CNN()
#     pred = model(seq, smile,pkt)
#     print(pred)
#     print(model)
