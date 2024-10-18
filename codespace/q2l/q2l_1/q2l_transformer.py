import copy
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import MultiheadAttention


class q2l_transformer(nn.Module):
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
        if num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(
                dim_feedforward,  # 512
                nhead,  # 8
                dropout,  # 0.1
                activation,  # gelu
                normalize_before,  # false
            )
            # 层归一化参数
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(
                encoder_layer,  # 编码层
                num_encoder_layers,  # 6
                encoder_norm,  # 层归一化
            )

        self._reset_parameters()
        self.nhead = nhead

        # self.debug_mode = False
        # self.set_debug_mode(self.debug_mode)

    def set_debug_mode(self, status):
        print("set debug mode to {}!!!".format(status))
        self.debug_mode = status
        if hasattr(self, "encoder"):
            for idx, layer in enumerate(self.encoder.layers):
                layer.debug_mode = status
                layer.debug_name = str(idx)
        if hasattr(self, "decoder"):
            for idx, layer in enumerate(self.decoder.layers):
                layer.debug_mode = status
                layer.debug_name = str(idx)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,query_embed, mask=None):
        L,B,D = src.shape
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)#45,32,512
        tgt = torch.zeros_like(query_embed)
        hs = self.encoder(tgt,src, src_key_padding_mask=mask, pos=query_embed)

        return hs


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,tgt,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = tgt

        for layer in self.layers:
            output = layer(
                output,src,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


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
        # Implementation of Feedforward model

        self.dropout = nn.Dropout(dropout)
        self.last_encoder = last_encoder
        # 所有头共需要的输入维度512，8头
        self.self_attn = MultiheadAttention(dim_feedforward, nhead, dropout=dropout)
        self.cross_attn = MultiheadAttention(dim_feedforward, nhead, dropout=dropout)
        self.linear1 = nn.Linear(dim_feedforward, 2048)
        self.linear2 = nn.Linear(2048, dim_feedforward)
        self.norm1 = nn.LayerNorm(dim_feedforward)
        self.norm2 = nn.LayerNorm(dim_feedforward)
        self.norm3 = nn.LayerNorm(dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation
        self.normalize_before = normalize_before

        self.debug_mode = False
        self.debug_name = None

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(#默认使用forward_post
        self,tgt,#45,32,512
        src,#2,32,512
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, pos)
        tgt2, _ = self.self_attn(q, k, value=tgt)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, _ = self.cross_attn(query=self.with_pos_embed(tgt, pos), key=src, value=src)#45,32,512
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
        

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    # 前向传播
    def forward(
        self,tgt,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            # 自注意力层：输入归一化
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
            # 前馈神经网络层：输出归一化
        return self.forward_post(tgt,src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# from timm import create_model


def build_transformerEncoder(args):
    return q2l_transformer(
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
