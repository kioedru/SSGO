import torch
import torch.nn as nn
import math

from codespace.model.multihead_attention_transformer import _get_activation_fn

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
        self.fusion = None
        if args.fusion == "transformer":
            from codespace.model.multihead_attention_transformer import (
                build_transformerEncoder,
            )

            self.fusion = build_transformerEncoder(args)
        elif args.fusion == "bimamba":
            from codespace.model.multihead_attention_bimamba_concat import (
                build_transformerEncoder,
            )

            self.fusion = build_transformerEncoder(args)
        elif args.fusion == "mamba":
            from codespace.model.multihead_attention_mamba_concat import (
                build_transformerEncoder,
            )

            self.fusion = build_transformerEncoder(args)

        self.fc_decoder = FC_Decoder(
            num_class=num_class,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            input_num=input_num + 1,
        )

        self.hs_org_fc0 = nn.Linear(dim_feedforward * 3, dim_feedforward * 2)
        self.hs_org_fc1 = nn.Linear(dim_feedforward * 2, dim_feedforward)
        self.hs_org_norm = nn.LayerNorm(dim_feedforward)

    def forward(self, src):
        # -----------------------------------获取对齐后的特征-------------------------------------------
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
        ppi_feature_org = torch.cat([in_x, in_z], dim=0)  # 2,32,512

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

        in_s = in_s.unsqueeze(0)  # 1,32,512
        seq_org = in_s

        # -----------------------------------将org融合 512*3->512-------------------------------------------
        hs_org = torch.cat([ppi_feature_org, seq_org], dim=0)  # 3,32,512
        hs_org = torch.sum(hs_org, dim=0).unsqueeze(0)  # 1,32,512
        hs_org = self.hs_org_norm(hs_org)  # 1,32,512
        # ppi_feature_org = torch.einsum("LBD->BLD", ppi_feature_org)  # 32,2,512
        # ppi_feature_org = ppi_feature_org.flatten(1)  # 32,512*2
        # seq_org = seq_org.squeeze(0)  # 32,512
        # hs_org = torch.cat([ppi_feature_org, seq_org], dim=1)  # 32,512*3
        # hs_org = self.hs_org_fc0(hs_org)  # 32,512*2
        # hs_org = self.hs_org_fc1(hs_org)  # 32,512
        # hs_org = hs_org.unsqueeze(0)  # 1,32,512
        # -----------------------------------融合-------------------------------------------

        ppi_feature_src = src  # 3,32,19385/1389/1024
        seq_src = src[2].unsqueeze(0)  # 1,32,1024
        _, hs_ppi_feature = self.ppi_feature_pre_model(ppi_feature_src)
        _, hs_seq = self.seq_pre_model(seq_src)
        hs = torch.cat([hs_ppi_feature, hs_seq, hs_org], dim=0)  # 4,32,512
        if self.fusion:
            hs = self.fusion(hs)

        out = self.fc_decoder(hs)
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
        input_num=args.input_num,
    )

    return predictor
