import copy
from typing import Optional, List

import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import MultiheadAttention


class EncoderTransformer(nn.Module):
    def __init__(
        self,
        dim_feedforward=2048,
        nhead=8,
        num_encoder_layers=6,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.dim_feedforward = dim_feedforward
        self.num_encoder_layers = num_encoder_layers
        encoder_layer = TransformerEncoderLayer(
            dim_feedforward,  # 512
            nhead,  # 8
            dropout,  # 0.1
            activation,  # gelu
            normalize_before,  # false
        )

        self.encoder = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_encoder_layers)]
        )

        self.nhead = nhead

    def forward(self, src):
        memory = self.encoder(src)

        return memory


# 用于定义Transformer编码器中的编码层
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_feedforward,  # 512
        nhead,  # 8
        dropout=0.1,  # 0.1
        activation="relu",  # gelu
        normalize_before=False,  # false
        last_encoder=False,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.last_encoder = last_encoder
        # 所有头共需要的输入维度512，8头
        self.self_attn = MultiheadAttention(dim_feedforward, nhead, dropout=dropout)
        self.linear1 = nn.Linear(dim_feedforward, 2048)
        self.linear2 = nn.Linear(2048, dim_feedforward)
        self.norm1 = nn.LayerNorm(dim_feedforward)
        self.norm2 = nn.LayerNorm(dim_feedforward)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(
        self,
        src,  # 3,32,512
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ):
        src2, corr = self.self_attn(
            query=src,
            key=src,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def build_transformerEncoder(args):
    return EncoderTransformer(
        dim_feedforward=args.dim_feedforward,  # 512
        activation=args.activation,  # gelu
        dropout=args.dropout,  # 0.1
        nhead=args.nheads,  # 8
        num_encoder_layers=args.attention_layers,  # 6
        normalize_before=args.pre_norm,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "lrelu":
        return F.leaky_relu
    if activation == "gelu":
        return F.gelu
    if activation == "sigmoid":
        return F.sigmoid
    if activation == "elu":
        return F.elu
    if activation == "tanh":
        return F.tanh
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
