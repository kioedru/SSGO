import torch
import torch.nn as nn
import math

from codespace.model.multihead_attention_transformer import (
    _get_activation_fn,
    build_transformerEncoder,
)
from codespace.utils.MultiModal.EnhancedSemanticAttentionModule import (
    EnhancedSemanticAttentionModule,
)
import copy

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
        self.fusion_seq = EnhancedSemanticAttentionModule(
            global_dim=dim_feedforward, local_dim=dim_feedforward, num_heads=8
        )
        self.fusion_ppi_feature = EnhancedSemanticAttentionModule(
            global_dim=dim_feedforward, local_dim=dim_feedforward, num_heads=8
        )
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
        self.ppi_feature_fison = nn.MultiheadAttention(dim_feedforward, args.nheads)

        self.fc_seq = nn.Linear(dim_feedforward * 2, dim_feedforward)
        self.act_seq = activation
        self.drop_seq = nn.Dropout(dropout)
        self.seq_fison = nn.MultiheadAttention(dim_feedforward, args.nheads)

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
        seq_src = torch.cat([seq_src, seq_src], dim=0)  # 2,32,512
        _, hs_ppi_feature = self.ppi_feature_pre_model(src)
        _, hs_seq = self.seq_pre_model(src[2].unsqueeze(0))
        hs_seq = torch.cat([hs_seq, hs_seq], dim=0)  # 2,32,512
        hs = torch.cat([hs_ppi_feature, hs_seq], dim=0)  # 4,32,512

        # global_ppi_feature = torch.einsum("LBD->BLD", hs_ppi_feature)  # 32,2,512
        # local_ppi_feature = torch.einsum("LBD->BLD", ppi_feature_src)  # 32,2,512
        # fusion_ppi_feature = self.fusion_ppi_feature(
        #     global_ppi_feature, local_ppi_feature
        # )  # 32,2,1024
        # fusion_ppi_feature = torch.einsum("BLD->LBD", fusion_ppi_feature)  # 2,32,1024

        # fusion_ppi_feature = self.fc_ppi_feature(fusion_ppi_feature)  # 2,32,512
        # fusion_ppi_feature = self.act_ppi_feature(fusion_ppi_feature)  # 2,32,512
        # fusion_ppi_feature = self.drop_ppi_feature(fusion_ppi_feature)  # 2,32,512

        fusion_ppi_feature = self.ppi_feature_fison(hs_ppi_feature, seq_src, seq_src)[
            0
        ]  # 交叉注意力，encoder特征/原始/原始

        # global_seq = torch.einsum("LBD->BLD", hs_seq)  # 32,1,512
        # local_seq = torch.einsum("LBD->BLD", seq_src)  # 32,1,512
        # fusion_seq = self.fusion_seq(global_seq, local_seq)  # 32,1,1024
        # fusion_seq = torch.einsum("BLD->LBD", fusion_seq)  # 1,32,1024

        # fusion_seq = self.fc_seq(fusion_seq)  # 1,32,512
        # fusion_seq = self.act_seq(fusion_seq)  # 1,32,512
        # fusion_seq = self.drop_seq(fusion_seq)  # 1,32,512
        fusion_seq = self.seq_fison(hs_seq, ppi_feature_src, ppi_feature_src)[0]

        before_fusion_hs = torch.cat(
            [fusion_ppi_feature, fusion_seq], dim=0
        )  # 3,32,512
        after_fusion_hs = self.fusion(before_fusion_hs)  # 3,32,512

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