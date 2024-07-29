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
    build_transformerEncoder as build_bimamba_encoder,
    _get_activation_fn,
)
from codespace.model.multihead_attention_transformer import (
    build_transformerEncoder as build_transformer_encoder,
)
from codespace.utils.MultiModal.EnhancedSemanticAttentionModule import (
    EnhancedSemanticAttentionModule,
)

# from mamba_ssm import Mamba


class Pre_Train_Model(nn.Module):
    def __init__(
        self,
        transformerEncoder,
        transformerDecoder,
        bimambaEncoder,
        bimambaDecoder,
        dim_feedforward,
        feature_len,
        activation,
        dropout,
    ):
        # feature_len=modefeature_lens = [X.shape[1], Z.shape[1]]即X的列数和Z的列数
        # 其中X是PPI信息的矩阵，Z是蛋白质结构域和亚细胞位置信息的矩阵

        super().__init__()
        self.transformerEncoder = transformerEncoder
        self.transformerDecoder = transformerDecoder
        self.bimambaEncoder = bimambaEncoder
        self.bimambaDecoder = bimambaDecoder
        self.fusionEncoder = EnhancedSemanticAttentionModule(
            dim_feedforward, dim_feedforward, num_heads=8
        )
        self.fusionDecoder = EnhancedSemanticAttentionModule(
            dim_feedforward, dim_feedforward, num_heads=8
        )
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

    def forward(self, src):
        in_s = src[2]  # 32,1024

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

        in_put = torch.cat([in_x, in_z, in_s], 0)  # 3,32,512
        L, B, D = in_put.shape  # 3,32,512

        # 编码器解码器
        hs_transformer = self.transformerEncoder(in_put)  # 3,32,512
        hs_bimamba = self.bimambaEncoder(in_put)  # 3,32,512
        input_fusion = torch.cat([hs_transformer, hs_bimamba], 0)  # 6,32,512
        input_transformer_fusion = input_fusion[:3]  # 32,3,512
        input_bimamba_fusion = input_fusion[3:]  # 32,3,512
        input_transformer_fusion = torch.einsum(
            "LBD->BLD", input_transformer_fusion
        )  # 32,3,512
        input_bimamba_fusion = torch.einsum(
            "LBD->BLD", input_bimamba_fusion
        )  # 32,3,512

        fusion_hs = self.fusionEncoder(
            input_transformer_fusion, input_bimamba_fusion
        )  # 32,3,512*2
        # 调整维度顺序，得到形状为 [3, 32, 512] 的张量
        fusion_hs_transformer, fusion_hs_bimamba = fusion_hs.chunk(
            2, dim=2
        )  # [32,3,512]

        fusion_rec = self.fusionDecoder(
            fusion_hs_transformer, fusion_hs_bimamba
        )  # [32,3,512*2]
        fusion_rec_transformer, fusion_rec_bimamba = fusion_rec.chunk(
            2, dim=2
        )  # [32,3,512]
        fusion_rec_transformer = torch.einsum(
            "BLD->LBD", fusion_rec_transformer
        )  # 3,32,512
        fusion_rec_bimamba = torch.einsum("BLD->LBD", fusion_rec_bimamba)  # 3,32,512
        rec_transformer = self.transformerDecoder(fusion_rec_transformer)
        rec_bimamba = self.bimambaDecoder(fusion_rec_bimamba)

        # ----------------------------多层感知机MLP---------------------------------------
        ph_s_transformer = self.W_s1(rec_transformer[2])
        ph_s_transformer = self.norm_ws(ph_s_transformer)
        ph_s_transformer = self.activation_ws(ph_s_transformer)
        ph_s_transformer = self.dropout_ws(ph_s_transformer)

        ph_x_transformer = self.W_x1(rec_transformer[0])
        ph_x_transformer = self.norm_wx(ph_x_transformer)
        ph_x_transformer = self.activation_wx(ph_x_transformer)
        ph_x_transformer = self.dropout_wx(ph_x_transformer)

        ph_z_transformer = self.W_z1(rec_transformer[1])
        ph_z_transformer = self.norm_wz(ph_z_transformer)
        ph_z_transformer = self.activation_wz(ph_z_transformer)
        ph_z_transformer = self.dropout_wz(ph_z_transformer)

        rec_s_transformer = self.W_s2(ph_s_transformer)
        rec_x_transformer = self.W_x2(ph_x_transformer)
        rec_z_transformer = self.W_z2(ph_z_transformer)

        # ----------------------------多层感知机MLP---------------------------------------
        ph_s_bimamba = self.W_s1(rec_bimamba[2])
        ph_s_bimamba = self.norm_ws(ph_s_bimamba)
        ph_s_bimamba = self.activation_ws(ph_s_bimamba)
        ph_s_bimamba = self.dropout_ws(ph_s_bimamba)

        ph_x_bimamba = self.W_x1(rec_bimamba[0])
        ph_x_bimamba = self.norm_wx(ph_x_bimamba)
        ph_x_bimamba = self.activation_wx(ph_x_bimamba)
        ph_x_bimamba = self.dropout_wx(ph_x_bimamba)

        ph_z_bimamba = self.W_z1(rec_bimamba[1])
        ph_z_bimamba = self.norm_wz(ph_z_bimamba)
        ph_z_bimamba = self.activation_wz(ph_z_bimamba)
        ph_z_bimamba = self.dropout_wz(ph_z_bimamba)

        rec_s_bimamba = self.W_s2(ph_s_bimamba)
        rec_x_bimamba = self.W_x2(ph_x_bimamba)
        rec_z_bimamba = self.W_z2(ph_z_bimamba)
        return (
            rec_x_transformer,
            rec_z_transformer,
            rec_s_transformer,
            rec_x_bimamba,
            rec_z_bimamba,
            rec_s_bimamba,
        ), fusion_hs


# 创建预训练模型
def build_Pre_Train_Model(args):

    model = Pre_Train_Model(
        transformerEncoder=build_transformer_encoder(args),
        transformerDecoder=build_transformer_encoder(args),
        bimambaEncoder=build_bimamba_encoder(args),
        bimambaDecoder=build_bimamba_encoder(args),
        dim_feedforward=args.dim_feedforward,
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
