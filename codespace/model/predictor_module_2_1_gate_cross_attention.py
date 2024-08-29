import torch
import torch.nn as nn

# import math
from codespace.model.gate_net import GateNet
from codespace.model.multihead_attention_transformer import (
    _get_activation_fn,
    build_transformerEncoder,
)

# from codespace.utils.MultiModal.EnhancedSemanticAttentionModule import (
#     EnhancedSemanticAttentionModule,
# )
# import copy

# from codespace.model.multihead_attention_transformer import build_transformerEncoder


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

        self.fusion = build_transformerEncoder(args)
        self.fc_decoder = FC_Decoder(
            num_class=num_class,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            input_num=input_num,
        )
        self.fc_ppi_feature = nn.Linear(dim_feedforward * 2, dim_feedforward)
        self.act_ppi_feature = activation
        self.drop_ppi_feature = nn.Dropout(dropout)
        self.ppi_feature_fuison = nn.MultiheadAttention(dim_feedforward, args.nheads)

        self.fc_seq = nn.Linear(dim_feedforward * 2, dim_feedforward)
        self.act_seq = activation
        self.drop_seq = nn.Dropout(dropout)
        self.seq_fuison = nn.MultiheadAttention(dim_feedforward, args.nheads)

        self.gatenet1 = GateNet(dim_feedforward * 3, 2)  # for ppi_feature
        self.gatenet2 = GateNet(dim_feedforward * 3, 2)  # for seq

    def forward(self, src):

        # src[0]是PPI信息的矩阵
        # batch_size=32因此共32行，19385列
        in_x = src[0]  # 32,19385
        in_x = self.ppi_feature_pre_model.input_proj_x1(
            in_x
        )  # 线性层，输入X的列数19385，输出512*2
        in_x = self.ppi_feature_pre_model.norm_x1(in_x)  # 标准化层，512*2
        in_x = self.ppi_feature_pre_model.activation_x1(in_x)  # 激活层 gelu
        in_x = self.ppi_feature_pre_model.dropout_x1(in_x)  # dropout

        # src[1]是蛋白质结构域和亚细胞位置信息的矩阵
        # batch_size=32因此共32行，1389列
        in_z = src[1]  # 32,1389
        in_z = self.ppi_feature_pre_model.input_proj_z1(
            in_z
        )  # 线性层，输入Z的列数1389，输出512*2
        in_z = self.ppi_feature_pre_model.norm_z1(in_z)
        in_z = self.ppi_feature_pre_model.activation_z1(in_z)
        in_z = self.ppi_feature_pre_model.dropout_z1(in_z)

        in_x = self.ppi_feature_pre_model.input_proj_x2(
            in_x
        )  # 线性层，输入512*2，输出512
        in_x = self.ppi_feature_pre_model.norm_x2(in_x)  # 标准化层，512
        in_x = self.ppi_feature_pre_model.activation_x2(in_x)  # 激活层 gelu
        in_x = self.ppi_feature_pre_model.dropout_x2(in_x)  # dropout

        in_z = self.ppi_feature_pre_model.input_proj_z2(in_z)
        in_z = self.ppi_feature_pre_model.norm_z2(in_z)
        in_z = self.ppi_feature_pre_model.activation_z2(in_z)
        in_z = self.ppi_feature_pre_model.dropout_z2(in_z)

        in_x = in_x.unsqueeze(0)  # 1,32,512
        in_z = in_z.unsqueeze(0)  # 1,32,512
        ppi_feature_src = torch.cat([in_x, in_z], dim=0)  # 2,32,512

        # src[0]是PPI信息的矩阵
        # batch_size=32因此共32行，19385列
        in_s = src[2]  # 1,32,19385
        in_s = self.seq_pre_model.input_proj_x1(
            in_s
        )  # 线性层，输入X的列数19385，输出512*2
        in_s = self.seq_pre_model.norm_x1(in_s)  # 标准化层，512*2
        in_s = self.seq_pre_model.activation_x1(in_s)  # 激活层 gelu
        in_s = self.seq_pre_model.dropout_x1(in_s)  # dropout

        in_s = self.seq_pre_model.input_proj_x2(in_s)  # 线性层，输入512*2，输出512
        in_s = self.seq_pre_model.norm_x2(in_s)  # 标准化层，512
        in_s = self.seq_pre_model.activation_x2(in_s)  # 激活层 gelu
        in_s = self.seq_pre_model.dropout_x2(in_s)  # dropout

        # ----------------------------多头注意力层---------------------------------------
        in_s = in_s.unsqueeze(0)  # 1,32,512
        seq_src = in_s
        # seq_src = torch.cat([seq_src, seq_src], dim=0)  # 2,32,512
        _, hs_ppi_feature = self.ppi_feature_pre_model(src)
        _, hs_seq = self.seq_pre_model(src[2].unsqueeze(0))
        hs = torch.cat([hs_ppi_feature, hs_seq, hs_seq], dim=0)  # 4,32,512

        gate_ppi_feature_ori = torch.cat([hs_ppi_feature, seq_src], dim=0)  # 3,32,512
        gate_ppi_feature = torch.einsum("LBD->BLD", gate_ppi_feature_ori).flatten(
            1
        )  # 32,1536
        weight_ppi_featrue = self.gatenet1(gate_ppi_feature)  # 32,2
        ppi_vec1 = weight_ppi_featrue[:, 0:1] * gate_ppi_feature_ori[0]  # 32,512
        feature_vec1 = weight_ppi_featrue[:, 0:1] * gate_ppi_feature_ori[1]  # 32,512
        seq_vec1 = weight_ppi_featrue[:, 1:2] * gate_ppi_feature_ori[2]  # 32,512
        gate_ppi_feature_vec = torch.stack(
            [ppi_vec1, feature_vec1, seq_vec1], dim=0
        )  # 3,32,512
        # output = weight[:, 0:1] * self.branch1(inputs[0]) + weight[:, 1:2] * self.branch2(inputs[1])
        q_ppi_feature = gate_ppi_feature_vec[0:2]  # 2,32,512
        k_ppi_feature = torch.stack(
            [gate_ppi_feature_vec[2], gate_ppi_feature_vec[2]], dim=0
        )  # 2,32,512
        v_ppi_feature = k_ppi_feature
        fusion_ppi_feature = self.ppi_feature_fuison(
            q_ppi_feature, k_ppi_feature, v_ppi_feature
        )[
            0
        ]  # 2,32,512 交叉注意力

        gate_seq_ori = torch.cat([ppi_feature_src, hs_seq], dim=0)  # 3,32,512
        gate_seq = torch.einsum("LBD->BLD", gate_seq_ori).flatten(1)  # 32,1536
        weight_seq = self.gatenet2(gate_seq)
        ppi_vec2 = weight_seq[:, 0:1] * gate_seq_ori[0]
        feature_vec2 = weight_seq[:, 0:1] * gate_seq_ori[1]
        seq_vec2 = weight_seq[:, 1:2] * gate_seq_ori[2]
        gate_seq_vec = torch.stack(
            [ppi_vec2, feature_vec2, seq_vec2], dim=0
        )  # 3,32,512
        q_seq = torch.stack([gate_seq_vec[2], gate_seq_vec[2]], dim=0)  # 2,32,512
        k_seq = gate_seq_vec[0:2]
        v_seq = k_seq

        fusion_seq = self.seq_fuison(q_seq, k_seq, v_seq)[0]  # 2,32,512

        before_fusion_hs = torch.cat(
            [fusion_ppi_feature, fusion_seq], dim=0
        )  # 4,32,512
        after_fusion_hs = self.fusion(before_fusion_hs)  # 4,32,512

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
