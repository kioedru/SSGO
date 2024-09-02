import torch
import torch.nn as nn

from codespace.model.multihead_attention_transformer_cross_attention import (
    _get_activation_fn,
    build_transformerEncoder,
)
from codespace.model.gate_net import GateNet


class FusionNet(nn.Module):
    def __init__(self, args, dim_feedforward, input_num=3):
        super().__init__()
        self.fusionlayer = build_transformerEncoder(args)
        self.gatenet = GateNet(dim_feedforward * input_num * 2, input_num * 2)

    def forward(self, hs, hs_src):  # hs[4, 32, 512]

        fusion_hs = self.fusionlayer(hs)  # 4,32,512
        gate_hs = torch.cat([fusion_hs, hs_src], dim=0)  # 8,32,512
        gate_hs_flatten = torch.einsum("LBD->BLD", gate_hs).flatten(1)  # 32,512*8
        weight = self.gatenet(gate_hs_flatten)  # 32,8

        gate_hs_permuted = torch.einsum("LBD->BLD", gate_hs)  # 32, 8, 512

        weight_expanded = weight.unsqueeze(-1)  # 32, 8, 1

        weighted_gate_hs = gate_hs_permuted * weight_expanded  # 32, 8, 512

        weighted_gate_hs_final = torch.einsum(
            "BLD->LBD", weighted_gate_hs
        )  # [8, 32, 512]
        return weighted_gate_hs_final


class FC_Decoder(nn.Module):
    def __init__(self, num_class, dim_feedforward, activation, dropout, input_num=3):
        super().__init__()
        self.num_class = num_class

        self.output_layer1 = nn.Linear(
            dim_feedforward * input_num * 2, dim_feedforward // (input_num * 2)
        )
        self.activation1 = torch.nn.Sigmoid()
        self.dropout1 = torch.nn.Dropout(dropout)

        self.output_layer3 = nn.Linear(dim_feedforward // (input_num * 2), num_class)

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


class Predictor(nn.Module):
    def __init__(
        self,
        args,
        seq_pre_model,
        ppi_feature_pre_model,
        num_class,
        dim_feedforward,
        activation,
        dropout,
        input_num=3,
    ):
        super().__init__()

        self.seq_pre_model = seq_pre_model
        self.ppi_feature_pre_model = ppi_feature_pre_model

        self.fusion = FusionNet(
            args=args, dim_feedforward=dim_feedforward, input_num=input_num
        )
        self.fc_decoder = FC_Decoder(
            num_class=num_class,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            input_num=input_num,
        )

    def forward(self, src):

        _, hs_ppi_feature = self.ppi_feature_pre_model(src)
        _, hs_seq = self.seq_pre_model(src[2].unsqueeze(0))
        hs = torch.cat([hs_ppi_feature, hs_seq, hs_seq], dim=0)  # 4,32,512

        after_fusion_hs = self.fusion(hs)  # 4,32,512

        out = self.fc_decoder(after_fusion_hs)
        return hs, out


def build_predictor(seq_pre_model, ppi_feature_pre_model, args):
    predictor = Predictor(
        args,
        seq_pre_model=seq_pre_model,
        ppi_feature_pre_model=ppi_feature_pre_model,
        num_class=args.num_class,
        dim_feedforward=args.dim_feedforward,
        activation=_get_activation_fn(args.activation),
        dropout=args.dropout,
        input_num=args.input_num + 1,
    )

    return predictor
