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

from codespace.model.multihead_attention_bimamba_concat import (
    build_transformerEncoder,
    _get_activation_fn,
)

from mamba_ssm import Mamba


class FC_Decoder(nn.Module):
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


class Pre_Train_Model(nn.Module):
    def __init__(
        self, transformerEncoder, feature_len, activation, dropout, num_class, input_num
    ):
        # feature_len=modefeature_lens = [X.shape[1], Z.shape[1]]即X的列数和Z的列数
        # 其中X是PPI信息的矩阵，Z是蛋白质结构域和亚细胞位置信息的矩阵

        super().__init__()
        self.transformerEncoder = transformerEncoder
        # dim_feedforward是自己设定的 512
        # dim_feedforward = transformerEncoder.dim_feedforward
        dim_feedforward = transformerEncoder.dim_feedforward
        self.input_proj_x1 = nn.Linear(feature_len[0], dim_feedforward * 2)
        self.input_proj_x2 = nn.Linear(dim_feedforward * 2, dim_feedforward)
        self.norm_x1 = nn.LayerNorm(dim_feedforward * 2)
        self.norm_x2 = nn.LayerNorm(dim_feedforward)
        self.dropout_x1 = nn.Dropout(dropout)
        self.dropout_x2 = nn.Dropout(dropout)
        self.activation_x1 = copy.deepcopy(activation)
        self.activation_x2 = copy.deepcopy(activation)

        self.fc_decoder = FC_Decoder(
            num_class=num_class,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            input_num=input_num,
        )

    def forward(self, src):

        # ----------------------------多层感知机MLP---------------------------------------

        # src[0]是PPI信息的矩阵
        # batch_size=32因此共32行，19385列
        in_x = src[0]  # 32,19385
        in_x = self.input_proj_x1(in_x)  # 线性层，输入X的列数19385，输出512*2
        in_x = self.norm_x1(in_x)  # 标准化层，512*2
        in_x = self.activation_x1(in_x)  # 激活层 gelu
        in_x = self.dropout_x1(in_x)  # dropout
        in_x = in_x  # 32,1024

        in_x = in_x  # 32,1024
        in_x = self.input_proj_x2(in_x)  # 线性层，输入512*2，输出512
        in_x = self.norm_x2(in_x)  # 标准化层，512
        in_x = self.activation_x2(in_x)  # 激活层 gelu
        in_x = self.dropout_x2(in_x)  # dropout
        in_x = in_x  # 32,512

        in_x = in_x.unsqueeze(0)  # 1,32,512
        # 编码器
        hs = self.transformerEncoder(in_x)  # B,K,d

        predict_out = self.fc_decoder(hs)
        return predict_out, hs


# 创建预训练模型
def build_Pre_Train_Model(args):

    model = Pre_Train_Model(
        transformerEncoder=build_transformerEncoder(args),
        feature_len=args.modesfeature_len,
        activation=_get_activation_fn(args.activation),
        dropout=args.dropout,
        num_class=args.num_class,
        input_num=args.input_num,
    )

    return model


# 测试
if __name__ == "__main__":
    # 创建模型实例
    model = Pre_Train_Model(
        transformerEncoder=None,
        feature_len=1024,
        activation=_get_activation_fn("gelu"),
        dropout=0.1,
        num_class=45,
        input_num=3,
    )
    # 打印模型结构
    print(model)
    x = torch.randn(32, 1, 512)
    y = model(x)
    print(y.shape)
