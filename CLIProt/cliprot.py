import torch
from torch import nn
import torch.nn.functional as F

from CLIProt.utils.losses import AsymmetricLossOptimized


class FC_Decoder(nn.Module):
    def __init__(self, num_class=45, dim_feedforward=512, dropout=0.3, input_num=3):
        super().__init__()
        self.num_class = num_class

        self.output_layer1 = nn.Linear(
            dim_feedforward * input_num, dim_feedforward // input_num
        )
        self.activation1 = torch.nn.Sigmoid()
        self.dropout1 = torch.nn.Dropout(dropout)

        self.output_layer3 = nn.Linear(dim_feedforward // input_num, num_class)

    def forward(self, hs):  # hs[32,3, 512]
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


class GO_Adapter(nn.Module):
    """
    input: (num_classes,go_dim)
    output: (num_classes, latent_dim)
    """

    def __init__(self, in_dim, out_dim=512):
        super(GO_Adapter, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


class Protein_Adapter(nn.Module):
    """
    input: (feature_num,batch_size, protein_dim)
    output: (batch_size, latent_dim)
    """

    def __init__(self, in_dim, out_dim=512):
        super(Protein_Adapter, self).__init__()
        self.linear = nn.Linear(in_dim * 3, out_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CLIProt(nn.Module):
    def __init__(self, go_feature, protein_dim=512, go_dim=256, latent_dim=512):
        super().__init__()
        self.go_feature = go_feature
        self.Protein_Adapter = Protein_Adapter(protein_dim, latent_dim)
        self.GO_Adapter = GO_Adapter(go_dim, latent_dim)
        self.Seq_Encoder = torch.load(
            "/home/Kioedru/code/SSGO/codespace/pretrain/one_feature_only/9606/transformer_seq1024_only.pkl",
            map_location="cuda:0",
        )
        self.PPI_Feature_Encoder = torch.load(
            "/home/Kioedru/code/SSGO/codespace/pretrain/transformer/9606/transformer.pkl",
            map_location="cuda:0",
        )
        self.decoder = FC_Decoder()

    def Encoder(self, protein_features):
        ppi_feature_src = protein_features  # 2,32,512
        seq_src = protein_features[2].unsqueeze(0)  # 1,32,512
        _, hs_ppi_feature = self.PPI_Feature_Encoder(ppi_feature_src)  # 2,32,512
        _, hs_seq = self.Seq_Encoder(seq_src)
        hs = torch.cat([hs_ppi_feature, hs_seq], dim=0)
        hs = torch.einsum("LBD->BLD", hs)  # 32,3,512
        return hs

    def forward(self, protein_features):
        align_features = self.Encoder(protein_features)
        return self.decoder(align_features)
        p_embed = self.Protein_Adapter(protein_features)
        g_embed = self.GO_Adapter(self.go_feature)
        # normalize
        p_embed = F.normalize(p_embed, p=2, dim=-1)
        g_embed = F.normalize(g_embed, p=2, dim=-1)

        # 计算相似度矩阵 (点积)
        similarity = torch.matmul(p_embed, g_embed.T)
        return similarity

    def compute_loss(self, logits, labels):
        """
        logits: 模型的输出相似度矩阵 (batch_size, num_go_terms)
        labels: 真实标签 (batch_size, num_go_terms)
        """
        loss_fn = AsymmetricLossOptimized()
        loss = loss_fn(logits, labels)
        return loss


if __name__ == "__main__":
    go_embedding = torch.rand((45, 256))  # 45个标签，每个标签200维
    protein_embedding = torch.rand((3, 32, 512))  # 32个蛋白质，每个蛋白质3,512
    label = torch.randint(
        0, 2, (32, 45)
    ).float()  # 32个蛋白质，每个蛋白质45个标签，随机0或1
    model = CLIProt(
        protein_dim=512,
        go_dim=256,
        latent_dim=512,
    )
    outs = model(protein_embedding, go_embedding)
    print(outs.shape)
    compute_loss = model.compute_loss(outs, label)
    print(compute_loss)
