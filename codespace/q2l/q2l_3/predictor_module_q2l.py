import torch
import torch.nn as nn
import math

from codespace.q2l.q2l_3.q2l_transformer import q2l_transformer, _get_activation_fn


class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
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


class FC_Decoder(nn.Module):
    def __init__(self, num_class, dim_feedforward, activation, dropout, input_num=3):
        super().__init__()
        self.num_class = num_class

        self.output_layer1 = nn.Linear(dim_feedforward, dim_feedforward // 2)
        self.activation1 = torch.nn.Sigmoid()
        self.dropout1 = torch.nn.Dropout(dropout)

        self.output_layer3 = nn.Linear(dim_feedforward // 2, num_class)

    def forward(self, hs):  # hs[32, 512]
        # # 维度转换 第0维和第1维互换
        # hs = hs.permute(1, 0, 2)  # [32, 45, 512]
        # # 按第一维度展开
        # hs = hs.flatten(1)  # [32,1536]
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
        activation,
        dropout,
        seq_pre_model=None,
        ppi_feature_pre_model=None,
        num_class=45,
        dim_feedforward=512,
        input_num=3,
    ):
        super().__init__()
        self.seq_pre_model = seq_pre_model
        self.ppi_feature_pre_model = ppi_feature_pre_model
        self.transformer = q2l_transformer(
            dim_feedforward=dim_feedforward,
            nhead=args.nheads,
            num_encoder_layers=args.attention_layers,
            dropout=dropout,
            activation=activation,
        )
        # self.query_embed = nn.Embedding(num_class, dim_feedforward)
        self.query_embed = nn.Parameter(torch.Tensor(num_class, dim_feedforward))
        nn.init.xavier_uniform_(self.query_embed)
        self.fc_decoder = GroupWiseLinear(num_class, dim_feedforward, bias=True)
        # self.fc_decoder = FC_Decoder(
        #     num_class=num_class,
        #     dim_feedforward=dim_feedforward,
        #     activation=activation,
        #     dropout=dropout,
        #     input_num=input_num,
        # )

    def forward(self, src):
        ppi_feature_src = src  # 2,32,512
        seq_src = src[2].unsqueeze(0)  # 1,32,512
        _, hs_ppi_feature = self.ppi_feature_pre_model(ppi_feature_src)  # 2,32,512
        _, hs_seq = self.seq_pre_model(seq_src)
        # hs = torch.cat([hs_ppi_feature, hs_seq], dim=0)  # 4,32,512
        query_input = self.query_embed
        hs = self.transformer(hs_ppi_feature, hs_seq, query_input)  # B,K,d 45,32,512
        out = self.fc_decoder(torch.einsum("KBD->BKD", hs))  # 32,45
        # out = self.fc_decoder(hs[-1])
        return hs, out


def build_predictor(args, seq_pre_model=None, ppi_feature_pre_model=None):
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
