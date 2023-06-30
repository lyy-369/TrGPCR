import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from collections import OrderedDict
'''
MGraphDTA: Deep Multiscale Graph Neural Network for Explainable Drug-target binding affinity Prediction
'''


class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx + 1, embedding_num, 96, 3)
            )

        self.linear = nn.Linear(block_num * 96, 96)

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x
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
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx + 1, embedding_num, 96, 3)
            )

        self.linear = nn.Linear(block_num * 96, 96)

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x


class MCNN_GATGCN(torch.nn.Module):
    def __init__(self,block_num=3, vocab_protein_size=26, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, embedding_size=128, filter_num=32, out_dim=1):

        super(MCNN_GATGCN, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)#?512是怎么来的
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        # self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        # self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        # self.fc1_xt = nn.Linear(32*121, output_dim)
        self.protein_encoder = TargetRepresentation(block_num, vocab_protein_size, embedding_size)

        # combined layers
        self.fc1 = nn.Linear(224, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        # print('x shape = ', x.shape)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        #Protein
        # embedded_xt = self.embedding_xt(target)
        # conv_xt = self.conv_xt_1(embedded_xt)
        # # flatten
        # xt = conv_xt.view(-1, 32 * 121)
        # xt = self.fc1_xt(xt)
        protein_x = self.protein_encoder(target)

        # concat
        xc = torch.cat((x, protein_x), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
