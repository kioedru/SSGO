import torch
import torch.nn as nn
import math

from codespace.model.multihead_attention_transformer import _get_activation_fn
from mamba_ssm import Mamba


class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class Qeruy2Label_Decoder(nn.Module):
    def __init__(self, DecoderTransformer, num_class):
        super().__init__()
        self.DecoderTransformer = DecoderTransformer
        self.num_class = num_class

        hidden_dim = DecoderTransformer.d_model
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)

    def forward(self, x):
        query_input = self.query_embed.weight
        pos = None
        hs = self.DecoderTransformer(x, query_input, pos)[0]  # B,K,d
        out = self.fc(hs[-1])

        # import ipdb; ipdb.set_trace()
        return out

    def finetune_paras(self):
        from itertools import chain

        return chain(
            self.DecoderTransformer.parameters(),
            self.fc.parameters(),
            self.query_embed.parameters(),
        )


class FC_Decoder(nn.Module):
    def __init__(
        self, num_class, dim_feedforward, residue_dim, activation, dropout, input_num=3
    ):
        super().__init__()
        self.num_class = num_class

        self.output_layer1 = nn.Linear(
            dim_feedforward * input_num, dim_feedforward // input_num
        )
        self.activation1 = torch.nn.Sigmoid()
        self.dropout1 = torch.nn.Dropout(dropout)

        self.output_layer3 = nn.Linear(2 * (dim_feedforward // input_num), num_class)
        self.residue_attn = Mamba(d_model=480, d_state=16, d_conv=4, expand=2)
        self.residue_linear1 = nn.Linear(residue_dim, dim_feedforward // input_num)
        self.output_layer2 = nn.Linear(2 * (dim_feedforward // input_num), num_class)

    def forward(self, hs, residue):  # hs[3, 32, 512] residue[32,2000,480]
        residue = self.residue_attn(residue)  # residue[32,2000,480]
        residue = nn.functional.adaptive_avg_pool1d(residue, output_size=1).squeeze(
            -1
        )  # residue[32,2000]
        residue = self.residue_linear1(residue)  # residue[32,512//3]

        # 维度转换 第0维和第1维互换
        hs = hs.permute(1, 0, 2)  # [32, 3, 512]
        # 按第一维度展开
        hs = hs.flatten(1)  # [32,512*3]

        hs = self.output_layer1(hs)  # [32,512//3]

        conca_hs_residue = torch.cat((hs, residue), dim=1)
        # sigmoid
        conca_hs_residue = self.activation1(conca_hs_residue)
        conca_hs_residue = self.dropout1(conca_hs_residue)
        # (512//3,GO标签数)
        out = self.output_layer3(conca_hs_residue)
        return out


class Predictor(nn.Module):
    def __init__(
        self,
        pre_model,
        num_class,
        dim_feedforward,
        residue_dim,
        activation,
        dropout,
        input_num=3,
    ):
        super().__init__()
        self.pre_model = pre_model
        self.fc_decoder = FC_Decoder(
            num_class=num_class,
            dim_feedforward=dim_feedforward,
            residue_dim=residue_dim,
            activation=activation,
            dropout=dropout,
            input_num=input_num,
        )

    def forward(self, src, residue):
        rec, hs = self.pre_model(src)
        out = self.fc_decoder(hs, residue)
        return rec, out


def build_predictor(pre_model, args):
    predictor = Predictor(
        pre_model=pre_model,
        num_class=args.num_class,
        dim_feedforward=args.dim_feedforward,
        residue_dim=args.modesfeature_len[3],
        activation=_get_activation_fn(args.activation),
        dropout=args.dropout,
        input_num=args.input_num,
    )

    return predictor
