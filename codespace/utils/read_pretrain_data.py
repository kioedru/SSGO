import os
import pandas as pd

dataset_path_in_kioedru = "/home/kioedru/code/SSGO/data"
dataset_path_in_Kioedru = "/home/Kioedru/code/SSGO/data"

if os.path.exists(dataset_path_in_kioedru):
    dataset_path = dataset_path_in_kioedru
else:
    dataset_path = dataset_path_in_Kioedru

pretrain_data_path = os.path.join(dataset_path, "pretrain")


# feature特征：[19385,1389]：亚细胞位置(442)+结构域特征(947)
def read_feature(organism_num):
    file_name = f"{organism_num}_feature.pkl"
    file_path = os.path.join(pretrain_data_path, file_name)
    feature = pd.read_pickle(file_path)
    return feature


# ppi特征：[19385,19385]
def read_ppi(organism_num):
    file_name = f"{organism_num}_ppi.pkl"
    file_path = os.path.join(pretrain_data_path, file_name)
    ppi_data = pd.read_pickle(file_path)
    # 读取ppi稀疏矩阵，转换为稠密矩阵
    ppi_matrix = ppi_data["matrix"].toarray()
    # print(ppi_matrix.shape)
    ppi_id = ppi_data["ppi_id"]
    # print(len(ppi_id), ppi_id[0:20])
    return ppi_matrix, ppi_id


# esm2:[19385,2000,480]->[19385, 2000]
def read_seq_embed_avgpool_esm2_480(organism_num):
    file_name = f"{organism_num}_seq_embed_avgpool_esm2_480.pkl"
    file_path = os.path.join(pretrain_data_path, file_name)
    seq = pd.read_pickle(file_path)
    return seq


# esm2:[19385,seq_len,480]->[19385, 480]
def read_seq_embed_avgpool_esm2_2000(organism_num):
    file_name = f"{organism_num}_seq_embed_avgpool_esm2_2000.pkl"
    file_path = os.path.join(pretrain_data_path, file_name)
    seq = pd.read_pickle(file_path)
    return seq


# prott5:[19385,seq_len,1024]->[19385, 1024]
def read_seq_embed_avgpool_prott5_1024(organism_num):
    file_name = f"{organism_num}_seq_embed_avgpool_prott5_1024.pkl"
    file_path = os.path.join(pretrain_data_path, file_name)
    seq = pd.read_pickle(file_path)
    return seq


# prott5:[19385,seq_len,1024]->[19385, 1024]
def read_seq_embed_avgpool_prott5_1024_new(organism_num):
    file_name = f"{organism_num}_seq_embed_avgpool_prott5_1024_new.pkl"
    file_path = os.path.join(pretrain_data_path, file_name)
    seq = pd.read_pickle(file_path)
    return seq


# 该文件目前未实现：one_hot_sum (19385,26)
def read_seq_one_hot_sum(organism_num):
    file_name = f"{organism_num}_seq_one_hot_sum.pkl"
    file_path = os.path.join(pretrain_data_path, file_name)
    seq = pd.read_pickle(file_path)
    return seq
