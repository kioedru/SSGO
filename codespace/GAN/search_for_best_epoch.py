__author__ = "cenwan"

# The Python implementation of Classifier Two-Sample Tests (CTST) for selecting the optimal synthetic protein feature samples.
# 这段代码实现了一个评估器，评估器使用“分类器两样本检验（Classifier Two-Sample Tests, CTST）”方法，使用knn对真实数据和生成数据进行分类
# 目标是找到使分类器无法区分真实数据和生成数据（即分类准确率接近0.5）的特定迭代轮数。
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import numpy as np


def search_best_epoch(aspect, organism_num, GOTermID, FeatureName, device="cpu"):
    GOTerm_path = os.path.join(
        "/home/Kioedru/code/SSGO/data/synthetic",
        FeatureName,
        organism_num,
        aspect,
        GOTermID,
    )
    FeaturesPositiveSamples = pd.read_pickle(
        os.path.join(GOTerm_path, GOTermID + "_Real_Training_Positive.pkl")
    )
    realFeatures = FeaturesPositiveSamples
    realDataset = np.array(realFeatures, dtype="float32")

    # 创建一个标签数组，真实数据的标签为1，生成数据的标签为0
    label = []
    for rowIndex in range(len(realDataset)):
        label.append(1)
    for rowIndex in range(len(realDataset)):
        label.append(0)
    labelArray = np.asarray(label)

    # 初始化用于存储最佳结果的变量
    opt_diff_accuracy_05 = 0.5
    opt_Epoch = 0
    opt_accuracy = 0
    # 迭代训练和评估
    # for indexEpoch in range(0, 500):
    for indexEpoch in range(0, 500):
        epoch = indexEpoch * 200

        # /home/Kioedru/code/SSGO/data/synthetic/esm2-480/9606/F/GO:0000287/GO:0000287_Iteration_800_Synthetic_Training_Positive.pkl
        fakeFeatures = pd.read_pickle(
            os.path.join(
                GOTerm_path,
                GOTermID
                + "_Iteration_"
                + str(epoch)
                + "_Synthetic_Training_Positive.pkl",
            )
        )  # [106,480]
        fakedataset = np.array(fakeFeatures, dtype="float32")

        # 合并真实数据和生成数据
        realFakeFeatures = np.vstack((realDataset, fakedataset))

        # 使用留一法交叉验证来评估分类器性能。每次迭代训练一个k-NN分类器，并预测测试样本的标签。
        prediction_list = []
        real_list = []
        loo = LeaveOneOut()
        loo.get_n_splits(realFakeFeatures)
        for train_index, test_index in loo.split(realFakeFeatures):
            X_train, X_test = (
                realFakeFeatures[train_index],
                realFakeFeatures[test_index],
            )
            y_train, y_test = labelArray[train_index], labelArray[test_index]
            knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
            predicted_y = knn.predict(X_test)
            prediction_list.append(predicted_y)
            real_list.append(y_test)

        # 计算分类准确率并更新最佳结果
        accuracy = accuracy_score(real_list, prediction_list)
        diff_accuracy_05 = abs(accuracy - 0.5)
        if diff_accuracy_05 < opt_diff_accuracy_05:
            opt_diff_accuracy_05 = diff_accuracy_05
            opt_Epoch = epoch
            opt_accuracy = accuracy

    # 输出最佳结果
    return opt_Epoch, opt_accuracy


import argparse


def parser_args():
    parser = argparse.ArgumentParser(description="CFAGO main")
    parser.add_argument("--aspect", type=str, choices=["P", "F", "C"], help="GO aspect")
    parser.add_argument("--organism_num", type=str)
    parser.add_argument("--FeatureName", type=str)
    # parser.add_argument("--device", type=str)
    args = parser.parse_args()
    return args


# nohup python -u /home/Kioedru/code/SSGO/codespace/GAN/search_for_best_epoch.py> /home/Kioedru/code/SSGO/data/synthetic/esm2-480/9606/seach_C.log 2>&1 &


def main():
    args = parser_args()
    args.aspect = "C"
    args.organism_num = "9606"
    args.FeatureName = "esm2-480"

    # /home/Kioedru/code/SSGO/data/finetune/9606/terms_P.pkl
    terms_list = pd.read_pickle(
        f"/home/Kioedru/code/SSGO/data/finetune/{args.organism_num}/terms_{args.aspect}.pkl"
    )["terms"].tolist()
    result_path = f"/home/Kioedru/code/SSGO/data/synthetic/{args.FeatureName}/{args.organism_num}/{args.aspect}_best_epoch.pkl"
    if os.path.exists(result_path):
        results = pd.read_pickle(result_path)
    else:
        results = []
    for term in terms_list:
        print(f"searching best epoch for {term}")
        opt_Epoch, opt_accuracy = search_best_epoch(
            args.aspect, args.organism_num, term, args.FeatureName
        )
        print(f"{term}   best epoch: {opt_Epoch}, best accuracy: {opt_accuracy}")
        results.append(
            {"term": term, "best_epoch": opt_Epoch, "best_accuracy": opt_accuracy}
        )
        pd.to_pickle(results, result_path)

    # Convert the results to a DataFrame
    df = pd.DataFrame(results)
    pd.to_pickle(
        df,
        f"/home/Kioedru/code/SSGO/data/synthetic/{args.FeatureName}/{args.organism_num}/{args.aspect}_best_epoch.pkl",
    )


if __name__ == "__main__":
    main()
