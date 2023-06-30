
import torch.nn as nn

from tqdm import tqdm
from collections import OrderedDict
import metrics

from Dateset import *

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
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        smi_embed_size = 128
        seq_embed_size = 128
        PT_FEATURE_SIZE = 25
        SM_FEATURE_SIZE = 64
        PKT_FEATURE_SIZE = 8
        seq_oc = 128
        pkt_oc = 128
        smi_oc = 128
        
        block_num=3
        #vocab_protein_size=26

        # (N, H=32, L)
        conv_smi = []
        conv_seq = []

        ic = seq_embed_size #128
        # self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size)  # (N, *, H_{in}) -> (N, *, H_{out})
        #self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size,bias=False) # 这边出问题了，seq[9,1000,25]-->seq[9,1000,25,128]
        #(4,128)
        self.smi_embed = nn.Linear(SM_FEATURE_SIZE, smi_embed_size,bias=False)  # (N, *, H_{in}) -> (N, *, H_{out})
        #(64,128)
        # self.pkt_embed = nn.Embedding(PKT_FEATURE_SIZE, seq_embed_size)
        self.pkt_embed = nn.Linear(PKT_FEATURE_SIZE,smi_embed_size,bias=False)
        #(4,128)
        self.cat_dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(224 , 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64, 1),
            nn.PReLU())
        for oc in [32, 64, smi_oc]:
            conv_smi.append(nn.Conv1d(ic, oc, 3))  # (N,C,L)  (128,OC,3)
            conv_smi.append(nn.BatchNorm1d(oc))
            conv_smi.append(nn.PReLU())
            ic = oc
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)  # (N,oc)
        self.protein_encoder = TargetRepresentation(block_num, PT_FEATURE_SIZE, seq_embed_size)   #(3,25,128)

        # self.Conv1d1 = nn.Conv1d(128, 32, 3)
        # self.BatchNorm1d1 = nn.BatchNorm1d(32)
        # self.PRelu = nn.PReLU()
        # self.AdaptiveMaxPool1d = nn.AdaptiveMaxPool1d(1)
        # self.Squeeze = Squeeze()

    def forward(self, seq, smi):#,pkt
#         seq_embed = self.seq_embed(seq)  # (N,L,32)
#         seq_embed = torch.transpose(seq_embed, 1, 2)
#         seq_conv = self.protein_encoder(seq_embed)  # (N,128)
        seq_conv = self.protein_encoder(seq)
        # seq_conv = self.Squeeze(output)

        smi_embed = self.smi_embed(smi)  # (N,L,32)
        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)  # (N,128)

#         pkt_embed = self.pkt_embed(pkt)  # (N,L,32)
#         pkt_embed = torch.transpose(pkt_embed, 1, 2)
#         pkt_embed = self.conv_smi(pkt_embed)  # (N,128)

        cat = torch.cat([seq_conv, smi_conv], dim=1)  # (N,128*3),pkt_embed

        cat = self.cat_dropout(cat)
        output = self.classifier(cat)
        return output
def test(model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for idx, (seq,smile,y) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):#,pkt
            seq = seq.to(device)
            smile = smile.to(device)
            y = y.to(device)
            #pkt = pkt.to(device)

            y_hat = model(seq,smile)#,pkt

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
#     pkt = torch.rand([2,40,4])#.float()
#     model = CNN()
#     pred = model(seq, smile,pkt)
#     print(pred)
#     print(model)