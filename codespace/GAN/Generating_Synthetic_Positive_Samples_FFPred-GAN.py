import os
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import sys
import pandas as pd

sys.path.append("/home/Kioedru/code/SSGO")
from codespace.utils.read_finetune_data import (
    read_seq_embed_avgpool_esm2_480_by_index,
    read_labels,
)


torch.manual_seed(1)

FEATUREDIM = 480
GDIM = 512
DDIM = int(FEATUREDIM / 3)
FIXED_GENERATOR = False
LAMBDA = 0.1
CRITIC_ITERS = 5
ITERS = 100000


# 生成器
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(FEATUREDIM, GDIM),
            nn.ReLU(True),
            nn.Linear(GDIM, GDIM),
            nn.ReLU(True),
            nn.Linear(GDIM, GDIM),
            nn.Tanh(),
            nn.Linear(GDIM, FEATUREDIM),
        )
        self.main = main

    def forward(self, noise, real_data):
        if FIXED_GENERATOR:
            return noise + real_data
        else:
            output = self.main(noise)
            return output


# 判别器
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(FEATUREDIM, DDIM)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(DDIM, DDIM)
        self.relu = nn.LeakyReLU()
        self.fc3 = nn.Linear(DDIM, DDIM)
        self.relu = nn.LeakyReLU()
        self.fc4 = nn.Linear(DDIM, 1)

    def forward(self, inputs):

        out = self.fc1(inputs)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)

        hidden1 = self.relu(self.fc1(inputs))
        hidden2 = self.relu(self.fc2(self.relu(self.fc1(inputs))))
        hidden3 = self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(inputs))))))

        return out.view(-1), hidden1, hidden2, hidden3


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# def inf_train_gen(GOTermID):

#     with open(
#         "./data/" + GOTermID + "/" + GOTermID + "_Real_Training_Positive.txt"
#     ) as f:
#         MatrixFeaturesPositive = [list(x.split(",")) for x in f]
#     proteinList = [line[0:1] for line in MatrixFeaturesPositive[:]]
#     FeaturesPositive = [line[1:259] for line in MatrixFeaturesPositive[:]]
#     dataset2 = np.array(FeaturesPositive, dtype="float32")

#     return dataset2


def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE, device):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, hidden_output_1, hidden_output_2, hidden_output_3 = netD(
        interpolates
    )

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=(torch.ones(disc_interpolates.size()).to(device)),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


import argparse


def parser_args():
    parser = argparse.ArgumentParser(description="CFAGO main")
    parser.add_argument("--aspect", type=str, choices=["P", "F", "C"], help="GO aspect")
    parser.add_argument("--organism_num", type=str)
    parser.add_argument("--GOTermID", type=str)
    parser.add_argument("--FeatureName", type=str)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parser_args()

    GOTermID = args.GOTermID
    device = args.device
    print(GOTermID, f"trainning on {device}")
    # -----Change this GOTermsID if this script is used for generating synthetic proteins for other GO terms.
    # /home/Kioedru/code/SSGO/data/synthetic/esm2-480/9606/P/GO:0000122/GO:0000122_Real_Training_Negative.pkl
    GOTerm_path = os.path.join(
        "/home/Kioedru/code/SSGO/data/synthetic",
        args.FeatureName,
        args.organism_num,
        args.aspect,
        GOTermID,
    )

    FeaturesPositiveSamples = pd.read_pickle(
        os.path.join(GOTerm_path, GOTermID + "_Real_Training_Positive.pkl")
    )
    BATCH_SIZE = FeaturesPositiveSamples.shape[0]

    netG = Generator()
    netD = Discriminator()
    netD.apply(weights_init)
    netG.apply(weights_init)

    netD = netD.to(device)
    netG = netG.to(device)

    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

    one = torch.FloatTensor([1])
    mone = one * -1
    one = one.to(device)
    mone = mone.to(device)

    data = FeaturesPositiveSamples

    # 整个训练过程的外层循环
    for iteration in range(ITERS):
        # 启用判别器参数梯度
        for p in netD.parameters():
            p.requires_grad = True
        # 获取真实数据并准备
        data = FeaturesPositiveSamples
        real_data = torch.FloatTensor(data)
        real_data = real_data.to(device)
        real_data_v = autograd.Variable(real_data)
        # 生成噪声数据并生成假数据
        noise = torch.randn(BATCH_SIZE, FEATUREDIM)
        noise = noise.to(device)

        with torch.no_grad():
            noisev = autograd.Variable(noise)
            fake = netG(noisev, real_data_v)

        fake_output = fake.data.cpu().numpy()

        # 训练判别器
        for iter_d in range(CRITIC_ITERS):
            # 计算真实数据的判别器输出和损失
            netD.zero_grad()

            D_real, hidden_output_real_1, hidden_output_real_2, hidden_output_real_3 = (
                netD(real_data_v)
            )
            D_real = D_real.mean().unsqueeze(0)
            D_real.backward(mone)
            # 计算假数据的判别器输出和损失
            noise = torch.randn(BATCH_SIZE, FEATUREDIM)
            noise = noise.to(device)
            with torch.no_grad():
                noisev = autograd.Variable(noise)
                fake = netG(noisev, real_data_v)

            inputv = fake
            D_fake, hidden_output_fake_1, hidden_output_fake_2, hidden_output_fake_3 = (
                netD(inputv)
            )
            D_fake = D_fake.mean().unsqueeze(0)
            D_fake.backward(one)
            # 计算梯度惩罚并进行反向传播
            gradient_penalty = calc_gradient_penalty(
                netD, real_data_v.data, fake.data, BATCH_SIZE, device
            )
            gradient_penalty.backward()
            # 计算判别器损失和Wasserstein距离
            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()

        # 每200次迭代保存生成器生成的假数据到文件
        if iteration % 200 == 0:
            pd.to_pickle(
                fake_output,  # np数组
                os.path.join(
                    GOTerm_path,
                    GOTermID
                    + "_Iteration_"
                    + str(iteration)
                    + "_Synthetic_Training_Positive.pkl",
                ),
            )

        # 训练生成器
        # 禁用判别器参数梯度
        if not FIXED_GENERATOR:

            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            # 准备真实数据和噪声数据
            real_data = torch.Tensor(data)
            real_data = real_data.to(device)
            real_data_v = autograd.Variable(real_data)

            # 生成假数据并计算生成器损失
            noise = torch.randn(BATCH_SIZE, FEATUREDIM)
            noise = noise.to(device)
            noisev = autograd.Variable(noise)
            fake = netG(noisev, real_data_v)
            (
                G,
                hidden_output_ignore_1,
                hidden_output_ignore_2,
                hidden_output_ignore_3,
            ) = netD(fake)
            G = G.mean().unsqueeze(0)
            G.backward(mone)
            G_cost = -G
            optimizerG.step()


if __name__ == "__main__":
    main()
