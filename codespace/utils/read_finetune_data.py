from read_pretrain_data import (
    read_uniprot_ppi,
    read_ppi,
    read_feature,
    read_seq_embedding_avgpooling,
    read_seq_embedding_avgpooling_prott5,
    read_seq_embedding,
    read_seq_embedding_avgpooling_esm2,
    read_seq_one_hot_sum,
)
import pandas as pd
import os
from sklearn.preprocessing import minmax_scale

# train_data_path = "/home/Kioedru/code/CFAGO_seq/train_data_protocol4"
train_data_path = "/home/kioedru/code/CFAGO/CFAGO_seq/train_data"


# 读取数据集
def read_df(usefor, aspect, organism_num):
    df_name = f"{usefor}_data_{aspect}.pkl"
    df_path = os.path.join(train_data_path, organism_num, df_name)
    df = pd.read_pickle(df_path)
    return df


# 获取ppi对应索引
def get_ppi_index(df, organism_num):
    ppi_matrix, ppi_id = read_ppi(organism_num)
    ppi_id = [x + ";" for x in ppi_id]
    index = df["Cross-reference (STRING)"].map(lambda x: ppi_id.index(x))
    return index


# 获取数据集的feature特征
def read_feature_by_index(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)
    feature = read_feature(organism_num)
    feature = minmax_scale(feature)
    selected_rows = feature[index]
    return selected_rows


# 获取数据集的onehot特征
def read_seq_onehot(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)
    seq_emb = read_seq_one_hot_sum(organism_num)
    seq_emb = minmax_scale(seq_emb)
    selected_rows = seq_emb[index]

    return selected_rows


# 获取数据集的seq_emb_avg特征
def read_seq_embedding_avgpooling_by_index(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)
    seq_emb = read_seq_embedding_avgpooling(organism_num)
    seq_emb = minmax_scale(seq_emb)
    selected_rows = seq_emb[index]

    return selected_rows


# 获取数据集的seq_emb_avg:esm2+prott5特征 (2000,1024)
def read_seq_embedding_avgpooling_esm2_prott5_by_index(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)

    seq_esm2_emb = read_seq_embedding_avgpooling(organism_num)
    selected_rows_esm2 = seq_esm2_emb[index]

    seq_prott5_emb = read_seq_embedding_avgpooling_prott5(organism_num).to("cpu")
    selected_rows_prott5 = seq_prott5_emb[index]

    selected_rows = np.concatenate((selected_rows_esm2, selected_rows_prott5), axis=1)
    selected_rows = minmax_scale(selected_rows)
    return selected_rows


# 获取esm2的480维特征
def read_seq_esm2_480_by_index(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)

    seq_esm2_emb = read_seq_embedding_avgpooling_esm2(organism_num).to("cpu")
    selected_rows_esm2 = seq_esm2_emb[index]

    selected_rows = minmax_scale(selected_rows_esm2)
    return selected_rows


# 获取prott5的1024维特征
def read_seq_prott5_1024_by_index(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)

    seq_prott5_emb = read_seq_embedding_avgpooling_prott5(organism_num).to("cpu")
    selected_rows_prott5 = seq_prott5_emb[index]

    selected_rows = minmax_scale(selected_rows_prott5)
    return selected_rows


# 获取esm2的480维特征和prott5的1024维特征（分别）
def read_esm2_480_and_prott5_1024_respectively_by_index(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)

    seq_esm2_emb = read_seq_embedding_avgpooling_esm2(organism_num).to("cpu")
    selected_rows_esm2 = seq_esm2_emb[index]

    seq_prott5_emb = read_seq_embedding_avgpooling_prott5(organism_num).to("cpu")
    selected_rows_prott5 = seq_prott5_emb[index]

    selected_rows_esm2 = minmax_scale(selected_rows_esm2)
    selected_rows_prott5 = minmax_scale(selected_rows_prott5)
    return selected_rows_esm2, selected_rows_prott5


# 获取esm2的480维特征和prott5的1024维特征（合并）
def read_seq_embedding_avgpooling_esm2_480_prott5_1024_by_index(
    usefor, aspect, organism_num
):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)

    seq_esm2_emb = read_seq_embedding_avgpooling_esm2(organism_num).to("cpu")
    selected_rows_esm2 = seq_esm2_emb[index]

    seq_prott5_emb = read_seq_embedding_avgpooling_prott5(organism_num).to("cpu")
    selected_rows_prott5 = seq_prott5_emb[index]

    selected_rows = np.concatenate((selected_rows_esm2, selected_rows_prott5), axis=1)
    selected_rows = minmax_scale(selected_rows)
    return selected_rows


# 获取数据集的seq_emb特征
def read_seq_embedding_by_index(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)
    seq_emb = read_seq_embedding(organism_num)
    seq_emb = minmax_scale(seq_emb)
    selected_rows = seq_emb[index]

    return selected_rows


# 获取数据集的ppi特征
def read_ppi_by_index(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)
    ppi_matrix, ppi_id = read_ppi(organism_num)
    ppi_matrix = minmax_scale(ppi_matrix)
    selected_rows = ppi_matrix[index]

    return selected_rows


from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


def read_terms(aspect, organism_num):
    terms_name = f"terms_{aspect}.pkl"
    file_path = os.path.join(train_data_path, organism_num, terms_name)
    terms = pd.read_pickle(file_path)
    return terms


# 获取数据集的labels
def read_labels(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    terms = read_terms(aspect, organism_num)
    mlb = MultiLabelBinarizer(classes=terms["terms"].tolist())

    encode = mlb.fit_transform(df["annotations"])
    encode = np.array(encode)
    return encode
