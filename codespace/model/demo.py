from mamba_ssm import Mamba
import torch
from codespace.mamba.mamba_ssm import BiMamba

mamba_atten = BiMamba(d_model=1).to("cuda:0")
src = torch.randn(32, 3, 512).to("cuda:0")
src2 = src.reshape(32, 1536).unsqueeze(2)
print(src2.shape)
src2 = mamba_atten(src2).squeeze(2)  # bimamba [32,1536]
src2 = src2.reshape(3, 32, 512)  # reshape成为输入的形式
print(src2.shape)
