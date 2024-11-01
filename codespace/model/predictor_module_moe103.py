import torch
import torch.nn as nn
import math

from codespace.model.multihead_attention_transformer import _get_activation_fn
from codespace.model.gate_net import GateNet


class expert0(nn.Module):  # for x1
    def __init__(self, input_num, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * input_num, input_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, input):  # [6,32,512]
        x = input[:3]  # 3,32,512
        x = torch.einsum("LBD->BLD", x).flatten(1)  # 32,512*3
        x = self.fc1(x)  # 32,512
        x = self.relu(x)
        x = self.fc2(x)  # 32,512
        return x


class expert1(nn.Module):  # for x2
    def __init__(self, input_num, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * input_num, input_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, input):  # [6,32,512]
        x = input[3:]  # 3,32,512
        x = torch.einsum("LBD->BLD", x).flatten(1)  # 32,512*3
        x = self.fc1(x)  # 32,512
        x = self.relu(x)
        x = self.fc2(x)  # 32,512
        return x


class expert2(nn.Module):  # for x1 cat x2
    def __init__(self, input_num, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * input_num, input_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, input):  # [6,32,512]
        x = torch.einsum("LBD->BLD", input).flatten(1)  # 32,512*6
        x = self.fc1(x)  # 32,512
        x = self.relu(x)
        x = self.fc2(x)  # 32,512
        return x


class expert3(nn.Module):  # for a*x1 cat b*x2
    def __init__(self, input_num, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * input_num, input_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, input):  # [6,32,512]
        x = torch.cat([input[:3] * self.a, input[3:] * self.b], dim=0)  # 6,32,512
        x = torch.einsum("LBD->BLD", x).flatten(1)  # 32,512*6
        x = self.fc1(x)  # 32,512
        x = self.relu(x)
        x = self.fc2(x)  # 32,512
        return x


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

    def forward(self, hs_model39):

        fusion_hs = hs_model39  # 3,32,512
        fusion_hs_flatten = torch.einsum("LBD->BLD", fusion_hs).flatten(1)  # 32,512*3
        fusion_hs_permuted = torch.einsum("LBD->BLD", fusion_hs)  # 32, 3, 512
        weight = self.gate(fusion_hs_flatten).unsqueeze(-1)  # 32,3,1
        weighted_gate_hs = fusion_hs_permuted * weight  # 32, 3, 512
        weighted_gate_hs = torch.einsum("BLD->LBD", weighted_gate_hs)  # 3, 32, 512

        model39_fusion = self.decoder_model39(weighted_gate_hs)  # 32,45
        return model39_fusion


class Fusion_Model17(nn.Module):
    def __init__(self, args, dim_feedforward):
        super().__init__()

        from codespace.model.multihead_attention_transformer import (
            build_transformerEncoder,
        )

        args_tmp = args
        args_tmp.attention_layers = 2
        self.model17fusionlayer = build_transformerEncoder(args_tmp)

    def forward(self, hs_model17):

        model17_fusion = self.model17fusionlayer(
            hs_model17
        )  # 3,32,512 做model17的transformer fusion
        return model17_fusion


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

    def forward(self, hs):  # hs[32, 512]
        hs = self.output_layer1(hs)  # 32,512
        # sigmoid
        hs = self.activation1(hs)
        hs = self.dropout1(hs)
        out = self.output_layer3(hs)  # 32,45
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
        # self.fusion_model39 = Fusion_Model39(args, dim_feedforward)
        self.gate = GateNet(dim_feedforward * 6, 4, hard=True)
        self.fc_decoder = FC_Decoder(
            num_class, dim_feedforward, activation, dropout, input_num=1
        )
        self.expert0 = expert0(3, dim_feedforward)
        self.expert1 = expert1(3, dim_feedforward)
        self.expert2 = expert2(6, dim_feedforward)
        self.expert3 = expert3(6, dim_feedforward)
        self.expert_list = nn.ModuleList(
            [self.expert0, self.expert1, self.expert2, self.expert3]
        )

    def forward_for_model17(self, src):
        # src[0]是PPI信息的矩阵
        # batch_size=32因此共32行，19385列
        in_x = src[0]  # 32,19385
        in_x = self.ppi_feature_pre_model17.input_proj_x1(
            in_x
        )  # 线性层，输入X的列数19385，输出512*2
        in_x = self.ppi_feature_pre_model17.norm_x1(in_x)  # 标准化层，512*2
        in_x = self.ppi_feature_pre_model17.activation_x1(in_x)  # 激活层 gelu
        in_x = self.ppi_feature_pre_model17.dropout_x1(in_x)  # dropout

        # src[1]是蛋白质结构域和亚细胞位置信息的矩阵
        # batch_size=32因此共32行，1389列
        in_z = src[1]  # 32,1389
        in_z = self.ppi_feature_pre_model17.input_proj_z1(
            in_z
        )  # 线性层，输入Z的列数1389，输出512*2
        in_z = self.ppi_feature_pre_model17.norm_z1(in_z)
        in_z = self.ppi_feature_pre_model17.activation_z1(in_z)
        in_z = self.ppi_feature_pre_model17.dropout_z1(in_z)

        in_x = self.ppi_feature_pre_model17.input_proj_x2(
            in_x
        )  # 线性层，输入512*2，输出512
        in_x = self.ppi_feature_pre_model17.norm_x2(in_x)  # 标准化层，512
        in_x = self.ppi_feature_pre_model17.activation_x2(in_x)  # 激活层 gelu
        in_x = self.ppi_feature_pre_model17.dropout_x2(in_x)  # dropout

        in_z = self.ppi_feature_pre_model17.input_proj_z2(in_z)
        in_z = self.ppi_feature_pre_model17.norm_z2(in_z)
        in_z = self.ppi_feature_pre_model17.activation_z2(in_z)
        in_z = self.ppi_feature_pre_model17.dropout_z2(in_z)

        in_x = in_x.unsqueeze(0)  # 1,32,512
        in_z = in_z.unsqueeze(0)  # 1,32,512
        ppi_feature_fc = torch.cat([in_x, in_z], dim=0)  # 2,32,512

        # src[0]是PPI信息的矩阵
        # batch_size=32因此共32行，19385列
        in_s = src[2]  # 1,32,19385
        in_s = self.seq_pre_model17.input_proj_x1(
            in_s
        )  # 线性层，输入X的列数19385，输出512*2
        in_s = self.seq_pre_model17.norm_x1(in_s)  # 标准化层，512*2
        in_s = self.seq_pre_model17.activation_x1(in_s)  # 激活层 gelu
        in_s = self.seq_pre_model17.dropout_x1(in_s)  # dropout

        in_s = self.seq_pre_model17.input_proj_x2(in_s)  # 线性层，输入512*2，输出512
        in_s = self.seq_pre_model17.norm_x2(in_s)  # 标准化层，512
        in_s = self.seq_pre_model17.activation_x2(in_s)  # 激活层 gelu
        in_s = self.seq_pre_model17.dropout_x2(in_s)  # dropout

        # ----------------------------多头注意力层---------------------------------------
        in_s = in_s.unsqueeze(0)  # 1,32,512
        seq_fc = in_s  # 1,32,512

        hs_fc = torch.cat([ppi_feature_fc, seq_fc], dim=0)  # 3,32,512

        ppi_feature_src = src  # 3,32,19385\1389\1024
        seq_src = src[2].unsqueeze(0)  # 1,32,1024

        # 获取model39的encoder结果
        # ppi_feature的预训练模型的Encoder部分
        ppi_feature_encoder_output = ppi_feature_fc  # 2,32,512
        seq_encoder_output = seq_fc  # 1,32,512
        return ppi_feature_encoder_output, seq_encoder_output

    def forward_for_model39(self, src):
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

        # 获取model39的encoder结果
        # ppi_feature的预训练模型的Encoder部分
        ppi_feature_encoder_output = ppi_feature_fc  # 2,32,512
        seq_encoder_output = seq_fc  # 1,32,512
        return ppi_feature_encoder_output, seq_encoder_output

    def forward(self, src):
        # 获取model39的encoder结果
        ppi_feature_encoder_output, seq_encoder_output = self.forward_for_model39(src)
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

        hs_model39 = torch.cat(
            [ppi_feature_encoder_output, seq_encoder_output], dim=0
        )  # 3,32,512
        # fusion_model39 = self.fusion_model39(hs_fc, hs_model39)  # 32,45

        # 获取model17的encoder结果
        ppi_feature_encoder_output, seq_encoder_output = self.forward_for_model17(src)

        for num in range(self.attention_layers_num):

            ppi_feature_output = (
                self.ppi_feature_pre_model17.transformerEncoder.encoder.layers[num](
                    ppi_feature_encoder_output
                )
            )
            seq_output = self.seq_pre_model17.transformerEncoder.encoder.layers[num](
                seq_encoder_output
            )
            ppi_feature_encoder_output = ppi_feature_output
            seq_encoder_output = seq_output

        hs_model17 = torch.cat(
            [ppi_feature_encoder_output, seq_encoder_output], dim=0
        )  # 3,32,512

        fusion_model17 = self.fusion_model17(hs_model17)  # 3,32,512

        fusion_hs = torch.cat([fusion_model17, hs_model39], dim=0)  # 6,32,512
        fusion_hs_flatten = torch.einsum("LBD->BLD", fusion_hs).flatten(1)  # 32,512*6
        weight = self.gate(fusion_hs_flatten)  # 32,4
        # weight的形状为 [32, 4, 1]，对应的是批次大小为32，每个输入的专家权重

        # 计算每个专家的输出
        expert_outputs = []
        for i, expert in enumerate(self.expert_list):
            expert_output = expert(fusion_hs)  # expert_output 形状为 [32, 512]
            expert_outputs.append(expert_output.unsqueeze(0))  # 1, 32, 512

        # 将所有专家输出堆叠在一起，形状变为 [4, 32, 512]
        expert_outputs = torch.cat(expert_outputs, dim=0)  # 4, 32, 512

        # 使用 torch.einsum 正确地计算加权专家输出
        # weight的形状是 [32, 4]，expert_outputs的形状是 [4, 32, 512]，希望输出为 [32, 512]
        # einsum: "bi,ibd->bd" 表示对 `i` 维度（专家数量）进行求和，剩下的是批次 `b` 和输出维度 `d`
        selected_expert_output = torch.einsum(
            "bi,ibd->bd", weight, expert_outputs
        )  # 32, 512

        fc_output = self.fc_decoder(selected_expert_output)  # 32,45
        return (
            fc_output,
            fusion_model17,
            hs_model39,
            selected_expert_output,
        )  # 前馈，分类，上，下，门控后


def build_predictor(
    seq_pre_model, ppi_feature_pre_model, seq_pre_model17, ppi_feature_pre_model17, args
):
    predictor = Predictor(
        args,
        seq_pre_model=seq_pre_model,
        ppi_feature_pre_model=ppi_feature_pre_model,
        seq_pre_model17=seq_pre_model17,
        num_class=args.num_class,
        ppi_feature_pre_model17=ppi_feature_pre_model17,
        dim_feedforward=args.dim_feedforward,
        activation=_get_activation_fn(args.activation),
        dropout=args.dropout,
        input_num=args.input_num,
    )

    return predictor
