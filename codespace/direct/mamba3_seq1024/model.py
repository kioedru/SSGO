# --------------------------------------------------------
# part of code borrowed from Quert2Label
# Written by Zhourun Wu
# --------------------------------------------------------
import sys
import os
import os.path as osp
from transformers import AutoTokenizer, AutoModel

import torch
from torch import Tensor
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import copy

from codespace.model.multihead_attention_mamba3 import (
    build_transformerEncoder,
    _get_activation_fn,
)

from mamba_ssm import Mamba


class Our_Model(nn.Module):
    def __init__(self, Encoder, Predictor):
        super().__init__()
        self.Encoder = Encoder
        self.Predictor = Predictor

    def forward(self, src):
        hs = self.Encoder(src)
        out = self.Predictor(hs)
        return out


class Predictor(nn.Module):
    def __init__(self, num_class, dim_feedforward, activation, dropout, input_num=3):
        super().__init__()
        self.num_class = num_class

        self.output_layer1 = nn.Linear(
            dim_feedforward * input_num, dim_feedforward // input_num
        )
        self.activation1 = torch.nn.Sigmoid()
        self.dropout1 = torch.nn.Dropout(dropout)

        self.output_layer3 = nn.Linear(dim_feedforward // input_num, num_class)

    def forward(self, hs):  # hs[3, 32, 512]
        # 维度转换 第0维和第1维互换
        hs = hs.permute(1, 0, 2)  # [32, 3, 512]
        # 按第一维度展开
        hs = hs.flatten(1)  # [32,1536]
        # 默认(512*2,512//2)，//表示下取整
        hs = self.output_layer1(hs)
        # sigmoid
        hs = self.activation1(hs)
        hs = self.dropout1(hs)
        # (512//2,GO标签数)
        out = self.output_layer3(hs)
        # 后面还需要一个sigmoid，在测试输出后面直接加了
        return out


class Encoder(nn.Module):
    def __init__(self, transformerEncoder, feature_len, activation, dropout):
        # feature_len=modefeature_lens = [X.shape[1], Z.shape[1]]即X的列数和Z的列数
        # 其中X是PPI信息的矩阵，Z是蛋白质结构域和亚细胞位置信息的矩阵

        super().__init__()
        self.transformerEncoder = transformerEncoder
        # dim_feedforward是自己设定的 512
        # dim_feedforward = transformerEncoder.dim_feedforward
        dim_feedforward = transformerEncoder.dim_feedforward

        self.input_proj_x1 = nn.Linear(feature_len[0], dim_feedforward * 2)
        self.input_proj_z1 = nn.Linear(feature_len[1], dim_feedforward * 2)
        self.input_proj_s1 = nn.Linear(feature_len[2], dim_feedforward * 2)

        self.input_proj_x2 = nn.Linear(dim_feedforward * 2, dim_feedforward)
        self.input_proj_z2 = nn.Linear(dim_feedforward * 2, dim_feedforward)
        self.input_proj_s2 = nn.Linear(dim_feedforward * 2, dim_feedforward)

        self.norm_x1 = nn.LayerNorm(dim_feedforward * 2)
        self.norm_z1 = nn.LayerNorm(dim_feedforward * 2)
        self.norm_s1 = nn.LayerNorm(dim_feedforward * 2)
        self.norm_x2 = nn.LayerNorm(dim_feedforward)
        self.norm_z2 = nn.LayerNorm(dim_feedforward)
        self.norm_s2 = nn.LayerNorm(dim_feedforward)

        self.dropout_x1 = nn.Dropout(dropout)
        self.dropout_z1 = nn.Dropout(dropout)
        self.dropout_s1 = nn.Dropout(dropout)
        self.dropout_x2 = nn.Dropout(dropout)
        self.dropout_z2 = nn.Dropout(dropout)
        self.dropout_s2 = nn.Dropout(dropout)

        self.activation_x1 = copy.deepcopy(activation)
        self.activation_z1 = copy.deepcopy(activation)
        self.activation_s1 = copy.deepcopy(activation)
        self.activation_x2 = copy.deepcopy(activation)
        self.activation_z2 = copy.deepcopy(activation)
        self.activation_s2 = copy.deepcopy(activation)

        self.W_x1 = nn.Linear(dim_feedforward, dim_feedforward * 2)
        self.W_z1 = nn.Linear(dim_feedforward, dim_feedforward * 2)
        self.W_s1 = nn.Linear(dim_feedforward, dim_feedforward * 2)
        self.W_x2 = nn.Linear(dim_feedforward * 2, feature_len[0])
        self.W_z2 = nn.Linear(dim_feedforward * 2, feature_len[1])
        self.W_s2 = nn.Linear(dim_feedforward * 2, feature_len[2])

    def forward(self, src):
        in_s = src[2]  # 32,2000

        # ----------------------------多层感知机MLP---------------------------------------
        in_s = self.input_proj_s1(in_s)  # 线性层，输入S的列数2000，输出512*2
        in_s = self.norm_s1(in_s)
        in_s = self.activation_s1(in_s)
        in_s = self.dropout_s1(in_s)
        in_s = in_s

        # src[0]是PPI信息的矩阵
        # batch_size=32因此共32行，19385列
        in_x = src[0]  # 32,19385
        in_x = self.input_proj_x1(in_x)  # 线性层，输入X的列数19385，输出512*2
        in_x = self.norm_x1(in_x)  # 标准化层，512*2
        in_x = self.activation_x1(in_x)  # 激活层 gelu
        in_x = self.dropout_x1(in_x)  # dropout
        in_x = in_x  # 32,1024

        # src[1]是蛋白质结构域和亚细胞位置信息的矩阵
        # batch_size=32因此共32行，1389列
        in_z = src[1]  # 32,1389
        in_z = self.input_proj_z1(in_z)  # 线性层，输入Z的列数1389，输出512*2
        in_z = self.norm_z1(in_z)
        in_z = self.activation_z1(in_z)
        in_z = self.dropout_z1(in_z)
        in_z = in_z  # 32,1024

        in_s = in_s
        in_s = self.input_proj_s2(in_s)
        in_s = self.norm_s2(in_s)
        in_s = self.activation_s2(in_s)
        in_s = self.dropout_s2(in_s)
        in_s = in_s

        in_x = in_x  # 32,1024
        in_x = self.input_proj_x2(in_x)  # 线性层，输入512*2，输出512
        in_x = self.norm_x2(in_x)  # 标准化层，512
        in_x = self.activation_x2(in_x)  # 激活层 gelu
        in_x = self.dropout_x2(in_x)  # dropout
        in_x = in_x  # 32,512

        in_z = in_z
        in_z = self.input_proj_z2(in_z)
        in_z = self.norm_z2(in_z)
        in_z = self.activation_z2(in_z)
        in_z = self.dropout_z2(in_z)
        in_z = in_z  # 32,512

        # ----------------------------多头注意力层---------------------------------------
        in_s = in_s.unsqueeze(0)
        in_x = in_x.unsqueeze(0)  # 1,32,512
        in_z = in_z.unsqueeze(0)  # 1,32,512

        # 按第0维拼接
        # in_put = torch.cat([in_x, in_z], 0)  # 2,32,512

        in_put = torch.cat([in_x, in_z, in_s], 0)  # 2,32,512

        # 编码器
        hs = self.transformerEncoder(in_put)  # B,K,d

        return hs


# 创建预训练模型
def build_our_model(args):

    encoder = Encoder(
        transformerEncoder=build_transformerEncoder(args),
        feature_len=args.modesfeature_len,
        activation=_get_activation_fn(args.activation),
        dropout=args.dropout,
    )
    predictor = Predictor(
        num_class=args.num_class,
        dim_feedforward=args.dim_feedforward,
        activation=_get_activation_fn(args.activation),
        dropout=args.dropout,
        input_num=args.input_num,
    )
    our_model = Our_Model(encoder, predictor)
    return our_model


if __name__ == "__main__":
    # 创建模型实例
    model = Encoder(
        transformerEncoder=None,
        feature_len=1024,
        activation=_get_activation_fn("gelu"),
        dropout=0.1,
    )
    # 打印模型结构
    print(model)
