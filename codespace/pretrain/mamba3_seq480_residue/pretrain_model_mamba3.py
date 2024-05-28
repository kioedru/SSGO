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


class Pre_Train_Model(nn.Module):
    def __init__(
        self, transformerEncoder, transformerDecoder, feature_len, activation, dropout
    ):
        # feature_len=modefeature_lens = [X.shape[1], Z.shape[1]]即X的列数和Z的列数
        # 其中X是PPI信息的矩阵，Z是蛋白质结构域和亚细胞位置信息的矩阵

        super().__init__()
        self.transformerEncoder = transformerEncoder
        self.transformerDecoder = transformerDecoder
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

        self.activation_wx = copy.deepcopy(activation)
        self.activation_wz = copy.deepcopy(activation)
        self.activation_ws = copy.deepcopy(activation)

        self.dropout_wx = nn.Dropout(dropout)
        self.dropout_wz = nn.Dropout(dropout)
        self.dropout_ws = nn.Dropout(dropout)

        self.norm_wx = nn.LayerNorm(dim_feedforward * 2)
        self.norm_wz = nn.LayerNorm(dim_feedforward * 2)
        self.norm_ws = nn.LayerNorm(dim_feedforward * 2)
        # 残基自注意力
        self.residue_attn = Mamba(
            d_model=feature_len[2], d_state=16, d_conv=4, expand=2
        )
        self.ins_layernorm = nn.LayerNorm(feature_len[2])

    def forward(self, src, residue):
        in_s = residue  # residue[32,2000,480]
        residue_score = self.residue_attn(in_s)  # residue_score[32,2000,480]
        residue_score = nn.functional.adaptive_avg_pool1d(
            residue_score, output_size=1
        )  # residue_score[32,2000,1]
        residue_score = nn.functional.softmax(
            residue_score, dim=1
        )  # residue_score[32,2000,1]
        # exists_not_equal = (residue_score != 0.0005).any()#输出True
        in_s = torch.sum(in_s * residue_score, dim=1)  # [32,480]
        # in_s = self.ins_layernorm(in_s)  # [32,480]
        ori_3 = in_s

        # ----------------------------多层感知机MLP---------------------------------------
        in_s = self.input_proj_s1(in_s)  # feature_len -> dim_feedforward * 2
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

        # 编码器解码器
        hs = self.transformerEncoder(in_put)  # B,K,d
        # 2,32,512
        rec = self.transformerDecoder(hs)  # B,K,d

        # ----------------------------多层感知机MLP---------------------------------------
        ph_s = self.W_s1(rec[2])
        ph_s = self.norm_ws(ph_s)
        ph_s = self.activation_ws(ph_s)
        ph_s = self.dropout_ws(ph_s)

        ph_x = self.W_x1(rec[0])
        ph_x = self.norm_wx(ph_x)
        ph_x = self.activation_wx(ph_x)
        ph_x = self.dropout_wx(ph_x)

        ph_z = self.W_z1(rec[1])
        ph_z = self.norm_wz(ph_z)
        ph_z = self.activation_wz(ph_z)
        ph_z = self.dropout_wz(ph_z)

        rec_s = self.W_s2(ph_s)
        rec_x = self.W_x2(ph_x)
        rec_z = self.W_z2(ph_z)
        # import ipdb; ipdb.set_trace()

        return (rec_x, rec_z, rec_s), hs, ori_3


# 创建预训练模型
def build_Pre_Train_Model(args):

    model = Pre_Train_Model(
        transformerEncoder=build_transformerEncoder(args),
        transformerDecoder=build_transformerEncoder(args),
        feature_len=args.modesfeature_len,
        activation=_get_activation_fn(args.activation),
        dropout=args.dropout,
    )

    return model


if __name__ == "__main__":
    # 创建模型实例
    model = Pre_Train_Model(
        transformerEncoder=None,
        transformerDecoder=None,
        feature_len=1024,
        activation=_get_activation_fn("gelu"),
        dropout=0.1,
    )
    # 打印模型结构
    print(model)
