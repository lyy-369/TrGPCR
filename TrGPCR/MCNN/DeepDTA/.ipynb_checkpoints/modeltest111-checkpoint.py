import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict
import metrics
from myDataset import *
# 蛋白质编码共25字符
# CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
#                "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
#                "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
#                "U": 19, "T": 20, "W": 21,
#                "V": 22, "Y": 23, "X": 24,
#                "Z": 25}
# CHARPROTLEN = 25
# # SMILEs编码共64字符
# CHARISOSMISET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
#                  "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
#                  "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
#                  "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
#                  "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
#                  "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
#                  "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
#                  "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}
# CHARISOSMILEN = 64
class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()

class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)
# GCN-CNN based model
class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):

        return self.inc(x).squeeze(-1)

class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        #self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)    
        self.embed = nn.Linear(25,128,bias=False)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx + 1, embedding_num, 96, 3)
            )

        self.linear = nn.Linear(block_num * 96, 96)

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        #x = self.embed(x)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x
class DTA(nn.Module):
    def __init__(self):
        super().__init__()
        smi_embed_size = 128
        seq_embed_size = 128
        PT_FEATURE_SIZE = 25
        SM_FEATURE_SIZE = 64
        seq_oc = 128
        #pkt_oc = 128
        smi_oc = 128
        block_num=3
        # (N, H=32, L)
        conv_pkt = []
        conv_seq = []

        ic = seq_embed_size
        # self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size)  # (N, *, H_{in}) -> (N, *, H_{out})
        self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size,bias=False) # 这边出问题了，seq[9,1000,25]-->seq[9,1000,25,128]
        self.smi_embed = nn.Linear(SM_FEATURE_SIZE, smi_embed_size,bias=False)  # (N, *, H_{in}) -> (N, *, H_{out})
        self.cat_dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(96 + 96, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64, 1),
            nn.PReLU())
        # for oc in [32, 64, 96]:
        #     conv_pkt.append(nn.Conv1d(ic, oc, 3))  # (N,C,L)
        #     conv_pkt.append(nn.BatchNorm1d(oc))
        #     conv_pkt.append(nn.PReLU())
        #     ic = oc
        # conv_pkt.append(nn.AdaptiveMaxPool1d(1))
        # conv_pkt.append(Squeeze())
        # self.conv_pkt = nn.Sequential(*conv_pkt)  # (N,oc)
        self.protein_encoder = TargetRepresentation(block_num, PT_FEATURE_SIZE, seq_embed_size)
#         self.conv_seq = nn.Sequential(
#             nn.Conv1d(128, 32, 4),
#             nn.BatchNorm1d(32),
#             nn.PReLU(),
#             nn.Conv1d(32, 64, 8),
#             nn.BatchNorm1d(64),
#             nn.PReLU(),
#             nn.Conv1d(64, 96, 12),
#             nn.BatchNorm1d(96),
#             nn.PReLU(),
#             nn.AdaptiveMaxPool1d(1),
#             Squeeze()
#         )
        self.conv_pkt = nn.Sequential(
            nn.Conv1d(128, 32, 4),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Conv1d(32, 64, 6),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Conv1d(64, 96, 8),
            nn.BatchNorm1d(96),
            nn.PReLU(),
            nn.AdaptiveMaxPool1d(1),
            Squeeze()
        )

    def forward(self, seq, pkt):
#         seq_embed = self.seq_embed(seq)  # (N,L,32)
#         seq_embed = torch.transpose(seq_embed, 1, 2)
#         seq_conv = self.conv_seq(seq_embed)  # (N,128)
        seq_conv = self.protein_encoder(seq)
        

        pkt_embed = self.smi_embed(pkt)  # (N,L,32)
        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_conv = self.conv_pkt(pkt_embed)  # (N,128)

        cat = torch.cat([seq_conv, pkt_conv], dim=1)  # (N,128*3)
        cat = self.cat_dropout(cat)
        output = self.classifier(cat)
        return output
def test(model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for idx,(seq,smile,y) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):#label,pkt,
            seq = seq.to(device)
            smile = smile.to(device)
            y = y.to(device)
            # pkt = pkt.to(device)

            y_hat = model(seq, smile)

            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

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

