from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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


def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind)),dtype="int8")  # +1

    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i, (smi_ch_ind[ch] - 1)] = 1

    return X  # .tolist()


def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind)),dtype="int8")
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i, (smi_ch_ind[ch]) - 1] = 1

    return X  # .tolist()


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
    def __init__(self, data_path, phase, max_seq_len, max_pkt_len, max_smi_len, pkt_window=None, pkt_stride=None):

        self.SEQLEN = max_seq_len
        self.SMILEN = max_smi_len
        self.PKTLEN = max_pkt_len

        self.charseqset = CHARPROTSET
        self.charsmiset = CHARISOSMISET


    def __getitem__(self, idx):
        # seq
        seqlist = []
        # smiles
        smileslist = []
        # label
        label = []
        with open(r"E:\Python\TrAdaboost\canruntr\MyTrAdaBoost\labelData\AllDATA\TransferData\combine.txt", 'r') as f:
            for line in f.readlines():
                line = line.strip('\n').split()  # 去掉列表中每一个元素的换行符
                seqlist.append(one_hot_sequence(line[1], self.SEQLEN, self.charseqset))
                smileslist.append(one_hot_smiles(line[0], self.SMILEN, self.charsmiset))
                label.append(int(line[2]))
            seqtensor = torch.Tensor(seqlist)
            smiletensor = torch.Tensor(smileslist)
            labeltensor = torch.LongTensor(label)
        self.seqtensor = seqtensor
        self.smiletensor = smiletensor
        self.labeltensor = labeltensor
        return self.seqtensor[idx], self.seqtensor[idx], self.smiletensor[idx], self.labeltensor[idx]
        # assert seq.name == pkt.name
        #
        # return (seq_tensor.astype(np.float32),
        #         pkt_tensor.astype(np.float32),
        #         label_smiles(self.smi[seq.name.split('.')[0]], self.max_smi_len),  # 根据seq的id获取对应的smiles,并标签编码
        #         np.array(self.affinity[seq.name.split('.')[0]], dtype=np.float32))

    def __len__(self):
        return len(self.labeltensor)
