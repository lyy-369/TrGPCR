from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

# 蛋白质编码共25字符
CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21,
               "V": 22, "Y": 23, "X": 24,
               "Z": 25}
CHARPROTLEN = 25
# SMILEs编码共64字符
CHARISOSMISET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                 "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                 "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                 "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                 "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                 "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                 "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                 "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}
CHARISOSMILEN = 64
# SSE编码共4字符
# SSESET = {"E": 1, "C": 2, "T": 3, "H": 4,"B":5,"S":6,"G":7,"I":8}
# SSELEN = 8

def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind)), dtype="int8")  # +1

    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i, (smi_ch_ind[ch] - 1)] = 1

    return X  # .tolist()


def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind)), dtype="int8")
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i, (smi_ch_ind[ch]) - 1] = 1

    return X  # .tolist()

# def one_hot_sse(line, MAX_SEQ_LEN, smi_ch_ind):
#     X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind)), dtype="int8")
#     for i, ch in enumerate(line[:MAX_SEQ_LEN]):
#         X[i, (smi_ch_ind[ch]) - 1] = 1

#     return X  # .tolist()
# def label_sse(lines, MAX_SSE_LEN, SSESET):
#     X = np.zeros(MAX_SSE_LEN)
#     for i, ch in enumerate(lines[:MAX_SSE_LEN]):
#         X[i] = SSESET[ch]
#     # print(lines)
#     return X.tolist()  # .tolist()

def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # x, smi_ch_ind, y
        X[i] = smi_ch_ind[ch]

    return X  # .tolist()


def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]

    return X  # .tolist()

class MyDataset(Dataset):
    def __init__(self, data_path, phase, max_seq_len,  max_smi_len):          #max_pkt_len,
        data_path = Path(data_path)
        self.SEQLEN = max_seq_len
        self.SMILEN = max_smi_len
        #self.PKTLEN = max_pkt_len
        self.charseqset = CHARPROTSET
        self.charsmiset = CHARISOSMISET
        #self.charsseset = SSESET
        self.train_path = data_path / f"{phase}.txt"
        length = []
        with open(self.train_path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n').split()
                length.append(float(line[2]))
        self.length = len(length)

        # seq
        seqlist = []
        # smiles
        smileslist = []
        # label
        #label = []
        # sse
        #sse = []
        # y
        y = []

        with open(self.train_path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n').split()  # 去掉列表中每一个元素的换行符
                seqlist.append(one_hot_sequence(line[1], self.SEQLEN, self.charseqset))
                smileslist.append(one_hot_smiles(line[0], self.SMILEN, self.charsmiset))
                #label.append(int(line[2]))
                # sse.append([round(a) for a in label_sse(line[3],self.SEQLEN,SSESET)])
                #sse.append(one_hot_sse(line[2],self.SEQLEN,self.charsseset))
                y.append(float(line[2]))

            seqtensor = torch.Tensor(seqlist)
            smiletensor = torch.Tensor(smileslist)
            #labeltensor = torch.Tensor(label)
            #ssetensor = torch.Tensor(sse)
            ytensor = torch.Tensor(y)
        self.seqtensor = seqtensor
        # print("seqtensor", seqtensor)
        self.smiletensor = smiletensor
        #self.labeltensor = labeltensor
        #self.sse = ssetensor
        self.y = ytensor

    def __getitem__(self, idx):

        return self.seqtensor[idx], self.smiletensor[idx],self.y[idx]#self.labeltensor[idx], self.sse[idx],

    def __len__(self):
        return self.length

if __name__ == '__main__':
    data_path = r'F:\test'
    data_loaders = {phase_name:
                        DataLoader(MyDataset(data_path, phase_name, max_seq_len=100,
                                             max_smi_len=100),              # max_pkt_len=100,
                                   batch_size=1,
                                   shuffle=True)
                    for phase_name in ['training', 'validation', 'test']}
    for i, ele in enumerate(data_loaders['training']):
        print(i, ele)
