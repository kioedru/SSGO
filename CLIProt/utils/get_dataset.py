from codespace.utils.read_finetune_data import (
    read_feature_by_index,
    read_labels,
    read_ppi_by_index,
    read_seq_embed_avgpool_esm2_2000_by_index,
    read_seq_embed_avgpool_esm2_480_by_index,
    read_seq_embed_avgpool_prott5_1024_by_index,
)

import numpy as np
import torch


# prott5:[num,1024]
def get_finetune_data(usefor, aspect, organism_num):
    feature = read_feature_by_index(usefor, aspect, organism_num)
    ppi_matrix = read_ppi_by_index(usefor, aspect, organism_num)
    seq = read_seq_embed_avgpool_prott5_1024_by_index(usefor, aspect, organism_num)
    labels = read_labels(usefor, aspect, organism_num)
    return feature, seq, ppi_matrix, labels


def get_dataset(aspect, organism_num):
    train_feature, train_seq, train_ppi_matrix, train_labels = get_finetune_data(
        "train", aspect, organism_num
    )
    valid_feature, valid_seq, valid_ppi_matrix, valid_labels = get_finetune_data(
        "valid", aspect, organism_num
    )
    test_feature, test_seq, test_ppi_matrix, test_labels = get_finetune_data(
        "test", aspect, organism_num
    )

    combine_feature = np.concatenate((train_feature, valid_feature), axis=0)
    combine_seq = np.concatenate((train_seq, valid_seq), axis=0)
    combine_ppi_matrix = np.concatenate((train_ppi_matrix, valid_ppi_matrix), axis=0)
    combine_labels = np.concatenate((train_labels, valid_labels), axis=0)

    combine_feature = torch.from_numpy(combine_feature).float()
    combine_seq = torch.from_numpy(combine_seq).float()
    combine_ppi_matrix = torch.from_numpy(combine_ppi_matrix).float()
    combine_labels = torch.from_numpy(combine_labels).float()
    test_feature = torch.from_numpy(test_feature).float()
    test_seq = torch.from_numpy(test_seq).float()
    test_ppi_matrix = torch.from_numpy(test_ppi_matrix).float()
    test_labels = torch.from_numpy(test_labels).float()
    train_dataset = multimodesDataset(
        3, [combine_ppi_matrix, combine_feature, combine_seq], combine_labels
    )
    test_dataset = multimodesDataset(
        3, [test_ppi_matrix, test_feature, test_seq], test_labels
    )
    return train_dataset, test_dataset


class multimodesDataset(torch.utils.data.Dataset):
    def __init__(self, num_modes, modes_features, labels):
        self.modes_features = modes_features
        self.labels = labels
        self.num_modes = num_modes

    def __getitem__(self, index):
        modes_features = []
        for i in range(self.num_modes):
            modes_features.append(self.modes_features[i][index])
        return {"protein_features": modes_features, "labels": self.labels[index]}

    def __len__(self):
        return self.modes_features[0].size(0)
