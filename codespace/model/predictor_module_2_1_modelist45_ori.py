import torch
import torch.nn as nn
import math

from codespace.model.multihead_attention_transformer import _get_activation_fn
from codespace.model.gate_net import GateNet

# from codespace.model.multihead_attention_transformer import build_transformerEncoder


class Fusion_Model17(nn.Module):
    def __init__(self, args, dim_feedforward):
        super().__init__()

        from codespace.model.multihead_attention_transformer import (
            build_transformerEncoder,
        )

        self.model17fusionlayer = build_transformerEncoder(args)
        self.model17_fusion = nn.MultiheadAttention(dim_feedforward, args.nheads)
        self.decoder_model17 = FC_Decoder(
            args.num_class,
            dim_feedforward,
            _get_activation_fn(args.activation),
            args.dropout,
            input_num=4,
        )

    def forward(self, hs_ppi_feature, hs_seq, hs_model17):
        model17_fusion_seq = self.model17_fusion(
            torch.stack((hs_model17[2], hs_model17[2]), dim=0),
            hs_ppi_feature,
            hs_ppi_feature,
        )[
            0
        ]  # 2,32,512 做model17的交叉注意力
        model17_fusion = torch.cat(
            [
                hs_model17[0].unsqueeze(0),
                hs_model17[1].unsqueeze(0),
                model17_fusion_seq,
            ],
            dim=0,
        )  # 4,32,512

        model17_fusion = self.model17fusionlayer(
            model17_fusion
        )  # 4,32,512 做model17的transformer fusion
        model17_fusion = self.decoder_model17(model17_fusion)  # 32,45
        return model17_fusion


class Fusion_Model39(nn.Module):
    def __init__(self, args, dim_feedforward):
        super().__init__()

        self.gate = GateNet(dim_feedforward * 3, 3, hard=False)
        self.decoder_model39 = FC_Decoder(
            args.num_class,
            dim_feedforward,
            _get_activation_fn(args.activation),
            args.dropout,
            input_num=3,
        )

    def forward(self, hs_ppi_feature, hs_seq, hs_model17):

        fusion_hs = torch.cat([hs_ppi_feature, hs_seq], dim=0)  # 3,32,512
        fusion_hs_flatten = torch.einsum("LBD->BLD", fusion_hs).flatten(1)  # 32,512*3
        fusion_hs_permuted = torch.einsum("LBD->BLD", fusion_hs)  # 32, 3, 512
        weight = self.gate(fusion_hs_flatten).unsqueeze(-1)  # 32,3,1
        weighted_gate_hs = fusion_hs_permuted * weight  # 32, 3, 512
        weighted_gate_hs = torch.einsum("BLD->LBD", weighted_gate_hs)  # 3, 32, 512

        model39_fusion = self.decoder_model39(weighted_gate_hs)  # 32,45
        return model39_fusion


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
        seq_pre_model17,
        ppi_feature_pre_model17,
        num_class,
        dim_feedforward,
        activation,
        dropout,
        input_num=3,
    ):
        super().__init__()
        self.attention_layers_num = args.attention_layers
        self.seq_pre_model = seq_pre_model
        self.ppi_feature_pre_model = ppi_feature_pre_model
        self.seq_pre_model17 = seq_pre_model17
        self.ppi_feature_pre_model17 = ppi_feature_pre_model17

        self.fusion_model17 = Fusion_Model17(args, dim_feedforward)
        self.fusion_model39 = Fusion_Model39(args, dim_feedforward)
        self.alpha = nn.Parameter(torch.tensor(0.5))

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
        ppi_feature_fc = torch.cat([in_x, in_z], dim=0)  # 2,32,512

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
        seq_fc = in_s  # 1,32,512

        hs_fc = torch.cat([ppi_feature_fc, seq_fc], dim=0)  # 3,32,512

        ppi_feature_src = src  # 3,32,19385\1389\1024
        seq_src = src[2].unsqueeze(0)  # 1,32,1024

        # ppi_feature的预训练模型的Encoder部分
        ppi_feature_encoder_output = ppi_feature_fc  # 2,32,512
        seq_encoder_output = seq_fc  # 1,32,512
        for num in range(self.attention_layers_num):
            ppi_feature_output = (
                self.ppi_feature_pre_model.transformerEncoder.encoder.layers[num](
                    ppi_feature_encoder_output, seq_encoder_output
                )
            )

            seq_output = self.seq_pre_model.transformerEncoder.encoder.layers[num](
                seq_encoder_output, ppi_feature_encoder_output
            )
            ppi_feature_encoder_output = ppi_feature_output
            seq_encoder_output = seq_output

        hs = torch.cat(
            [ppi_feature_encoder_output, seq_encoder_output], dim=0
        )  # 3,32,512

        # 获取model17的encoder结果
        _, hs_model17_ppi_feature = self.ppi_feature_pre_model17(src)
        _, hs_model17_seq = self.seq_pre_model17(src[2].unsqueeze(0))
        hs_model17 = torch.cat(
            [hs_model17_ppi_feature, hs_model17_seq], dim=0
        )  # 3,32,512

        fusion_model17 = self.fusion_model17(
            ppi_feature_encoder_output, seq_encoder_output, hs_model17
        )
        fusion_model39 = self.fusion_model39(
            ppi_feature_encoder_output, seq_encoder_output, hs_model17
        )
        fusion_hs = self.alpha * fusion_model17 + (1 - self.alpha) * fusion_model39

        return hs, fusion_hs


def build_predictor(
    seq_pre_model, ppi_feature_pre_model, seq_pre_model17, ppi_feature_pre_model17, args
):
    predictor = Predictor(
        args,
        seq_pre_model=seq_pre_model,
        ppi_feature_pre_model=ppi_feature_pre_model,
        seq_pre_model17=seq_pre_model17,
        ppi_feature_pre_model17=ppi_feature_pre_model17,
        num_class=args.num_class,
        dim_feedforward=args.dim_feedforward,
        activation=_get_activation_fn(args.activation),
        dropout=args.dropout,
        input_num=args.input_num + 4,
    )

    return predictor
