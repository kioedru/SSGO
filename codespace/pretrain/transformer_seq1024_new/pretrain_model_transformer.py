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

from codespace.model.multihead_attention_transformer import (
    build_transformerEncoder,
    _get_activation_fn,
)

from mamba_ssm import Mamba
from torch.nn import MultiheadAttention


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        activation,
        dim_feedforward=2048,
        nhead=8,
        num_encoder_layers=6,
        dropout=0.1,
    ):
        super().__init__()
        self.dim_feedforward = dim_feedforward
        self.num_encoder_layers = num_encoder_layers
        encoder_layer = TransformerLayer(
            dim_feedforward,  # 512
            nhead,  # 8
            dropout,  # 0.1
            activation,  # gelu
        )

        self.encoder = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_encoder_layers)]
        )

        self.nhead = nhead

    def forward(self, src):
        memory = self.encoder(src)

        return memory


# 用于定义TransformerLayer
class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim_feedforward,
        nhead,  # 8
        dropout,
        activation,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.self_attn = MultiheadAttention(dim_feedforward, nhead, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(dim_feedforward, 2048),
            activation,
            nn.Dropout(dropout),
            nn.Linear(2048, dim_feedforward),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim_feedforward)
        self.norm2 = nn.LayerNorm(dim_feedforward)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src,  # 3,32,512
    ):

        src = self.fc(src)
        src2, corr = self.self_attn(
            query=src,
            key=src,
            value=src,
        )

        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.fc(src)
        src = src + src2
        src = self.norm2(src)
        return src


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, activation, dropout):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation,
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            activation,
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Pre_Train_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        activation = _get_activation_fn(args.activation)
        self.transformerEncoder = build_transformerEncoder(args)
        self.transformerDecoder = build_transformerEncoder(args)
        self.downsample_ppi = MLP(
            args.feature_len[0],
            args.dim_feedforward * 2,
            args.dim_feedforward,
            activation,
            args.dropout,
        )
        self.downsample_feature = MLP(
            args.feature_len[1],
            args.dim_feedforward * 2,
            args.dim_feedforward,
            activation,
            args.dropout,
        )
        self.downsample_seq = MLP(
            args.feature_len[2],
            args.dim_feedforward * 2,
            args.dim_feedforward,
            activation,
            args.dropout,
        )
        self.upsample_ppi = MLP(
            args.dim_feedforward,
            args.dim_feedforward * 2,
            args.feature_len[0],
            activation,
            args.dropout,
        )
        self.upsample_feature = MLP(
            args.dim_feedforward,
            args.dim_feedforward * 2,
            args.feature_len[1],
            activation,
            args.dropout,
        )
        self.upsample_seq = MLP(
            args.dim_feedforward,
            args.dim_feedforward * 2,
            args.feature_len[2],
            activation,
            args.dropout,
        )

    def forward(self, src):
        ppi = src[0]
        feature = src[1]
        seq = src[2]
        ppi = self.downsample_ppi(ppi)
        feature = self.downsample_feature(feature)
        seq = self.downsample_seq(seq)
        hs = torch.cat([ppi, feature, seq], dim=0)  # 3,32,512
        hs = self.transformerEncoder(hs)
        rec = self.transformerDecoder(hs)
        ppi = self.upsample_ppi(rec[0])
        feature = self.upsample_feature(rec[1])
        seq = self.upsample_seq(rec[2])

        return (ppi, feature, seq), hs


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
