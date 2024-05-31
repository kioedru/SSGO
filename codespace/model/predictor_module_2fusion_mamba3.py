import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba
from codespace.model.multihead_attention_transformer import _get_activation_fn


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
    def __init__(self, num_class, dim_feedforward, activation, dropout, input_num=3):
        super().__init__()
        self.num_class = num_class

        self.output_layer1 = nn.Linear(
            dim_feedforward * input_num, dim_feedforward * int(input_num / 2)
        )
        self.activation1 = activation
        self.dropout1 = torch.nn.Dropout(dropout)

        self.output_layer2 = nn.Linear(
            dim_feedforward * int(input_num / 2),
            dim_feedforward // int((input_num / 2)),
        )
        self.activation2 = torch.nn.Sigmoid()
        self.dropout2 = torch.nn.Dropout(dropout)

        self.output_layer3 = nn.Linear(dim_feedforward // int(input_num / 2), num_class)
        self.fusion_atten = Mamba(
            d_model=dim_feedforward, d_state=16, d_conv=4, expand=2
        )

    def forward(self, hs):  # hs[6, 32, 512]
        hs = self.fusion_atten(hs)
        # 维度转换 第0维和第1维互换
        hs = hs.permute(1, 0, 2)  # [32, 6, 512]
        # 按第一维度展开
        hs = hs.flatten(1)  # [32,512*6]
        # 默认(512*2,512//2)，//表示下取整
        hs = self.output_layer1(hs)  # [32,512*(6/2)]
        hs = self.activation1(hs)
        hs = self.dropout1(hs)

        hs = self.output_layer2(hs)  # [32,512//(6/2)]
        hs = self.activation2(hs)
        hs = self.dropout2(hs)

        # (512//2,GO标签数)
        out = self.output_layer3(hs)
        # 后面还需要一个sigmoid，在测试输出后面直接加了
        return out


class Predictor(nn.Module):
    def __init__(
        self,
        pre_model_transformer,
        pre_model_mamba,
        num_class,
        dim_feedforward,
        activation,
        dropout,
        input_num=3,
    ):
        super().__init__()
        self.pre_model_transformer = pre_model_transformer
        self.pre_model_mamba = pre_model_mamba
        self.fc_decoder = FC_Decoder(
            num_class=num_class,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            input_num=input_num,
        )

    def forward(self, src):
        rec_transformer, hs_transformer = self.pre_model_transformer(src)
        rec_mamba, hs_mamba = self.pre_model_mamba(src)
        # hs[3,32,512]
        # 直接拼接
        rec = list(rec_transformer) + list(rec_mamba)
        hs = torch.cat((hs_transformer, hs_mamba), dim=0)  # [6,32,512]
        out = self.fc_decoder(hs)
        return rec, out


def build_predictor(pre_model_transformer, pre_model_mamba, args):
    predictor = Predictor(
        pre_model_transformer=pre_model_transformer,
        pre_model_mamba=pre_model_mamba,
        num_class=args.num_class,
        dim_feedforward=args.dim_feedforward,
        activation=_get_activation_fn(args.activation),
        dropout=args.dropout,
        input_num=6,
    )

    return predictor
