import math

import networkx as nx
import pandas as pd
from pandas import concat
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$              zhanbi                       $"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""


def TPFN_v1(collusion_P, pre, epoch, _list):
    worker_pre = _list  # 预测的串谋工人

    worker_ture = []  # 真实串谋工人
    W_path = "../../result/collusion_data/collusion_" + str(collusion_P) + "/experience_" + str(pre) + "/data" + str(
        epoch) + "_collaborate_W.csv"
    worker_ture_pd = pd.read_csv(W_path)
    for i in range(len(worker_ture_pd)):
        worker_ture.append(worker_ture_pd.iloc[i][0])

    # print("预测串谋工人：", worker_pre)
    # print("真实串谋工人：", worker_ture)

    #  真实串谋工人的任务
    D_path = "../../result/collusion_data/collusion_" + str(collusion_P) + "/experience_" + str(pre) + "/data" + str(
        epoch) + "_collaborate_D.csv"
    data = pd.read_csv(D_path)
    works_ture = data.iloc[:1, :]
    for i in range(len(worker_ture)):
        tmp_works = data[(data.worker == worker_ture[i])]
        works_ture = works_ture.append(tmp_works)
    works_ture = works_ture.iloc[1:, :]
    # print("真实串谋任务")
    # print(works_ture)

    #  串谋串谋工人的所有任务
    works_pre = data.iloc[:1, :]
    for i in range(len(worker_pre)):
        tmp_works = data[(data.worker == worker_pre[i])]
        works_pre = works_pre.append(tmp_works)
    works_pre = works_pre.iloc[1:, :]
    # print("预测串谋任务")
    # print(works_pre)

    TP = 0  # 真实情况是串谋工人的任务，被识别为串谋工人的任务
    # 求 TP  FN
    for i in range(len(works_ture)):
        # print(i)
        for j in range(len(works_pre)):
            df1 = works_ture.iloc[i:i + 1, :]
            df2 = works_pre.iloc[j:j + 1, :]
            if df1.equals(df2):
                TP = TP + 1

    FN = len(works_ture) - TP  # 真实情况是串谋工人的任务，被识别为正常工人的任务
    FP = len(works_pre) - TP  # 真实情况是正常工人的任务，被识别为串谋工人的任务
    TN = len(data) - TP - FP - FN  # 真实情况是正常工人的任务，被识别为真实工人的任务

    # print("TP：", TP, "/", len(works_ture))
    # print("FN=", FN, "/", len(works_ture))
    # print("FP=", FP, "/", len(data)-len(works_ture))
    # print("TN=", TN, "/", len(data)-len(works_ture))

    Accuracy = (TP + TN) / len(data)

    if TP + FP != 0:
        Precsion = TP / (TP + FP)
    else:
        Precsion = 0

    Recall = TP / (TP + FN)

    if Precsion + Recall != 0:
        f1_score = 2 * ((Precsion * Recall) / (Precsion + Recall))
    else:
        f1_score = 0

    print("Accuracy=", Accuracy)
    print("Precsion=", Precsion)
    print("Recall=", Recall)
    print("f1_score=", f1_score)

    return (Accuracy, Precsion, Recall, f1_score)


def Percentage(collusion_P, pre, alpha, beta, K, epoch):
    print("collusion_P=", collusion_P, "alpha=", alpha, "beta=", beta, "KCDN_percentage=", pre, "dection=", pre,
          "epoch=", epoch)

    # print(path)
    path = "../../result/collusion_data/collusion_" + str(collusion_P) + "/experience_" + str(pre) + "/data" + str(
        epoch) + "_collaborate_worker_ks_sumW_sumWN.csv"
    data = pd.read_csv(path)
    # print("Acc Rec Pre F1")
    # 如果是计算APRF
    if (K == "KK"):
        list_worker = []
        list_KKj = []
        for i in range(len(data)):
            ks = data.iloc[i][1]
            sumW = data.iloc[i][2]
            KKj = (alpha / (alpha + beta)) * math.log(ks) + (beta / (alpha + beta)) * math.log(sumW)

            # 工人
            list_worker.append(data.iloc[i][0])
            # KKj
            list_KKj.append(KKj)

        worker = DataFrame({"worker": list_worker})
        KKj = DataFrame({"KKj": list_KKj})

        data = concat([worker, KKj], axis=1)
        data = data.sort_values('KKj')

        collusion_worker = []  # 取前precentage的工人，认为是串谋工人

        collusion_data = data.iloc[len(data) - pre:len(data), :]

        for i in range(len(collusion_data)):
            collusion_worker.append(int(collusion_data.iloc[i][0]))
        collusion_worker = sorted(list(set(collusion_worker)))

        (acc1, pre1, rec1, f1_score) = TPFN_v1(collusion_P, pre, epoch, collusion_worker)

        return (acc1, pre1, rec1, f1_score)

    # if K == "KKK":
    #     list_worker = []
    #     list_KKKj = []
    #     for i in range(len(data)):
    #         ks = data.iloc[i][1]
    #         sumW = data.iloc[i][2]
    #         sumWN = data.iloc[i][3]
    #
    #         KKKj = (alpha / (alpha + beta + gama)) * math.log(ks) \
    #                + (beta / (alpha + beta + gama)) * math.log(sumW) \
    #                + (gama / (alpha + beta + gama)) * math.log(sumWN)
    #
    #
    #         # 工人
    #         list_worker.append(data.iloc[i][0])
    #         # KKj
    #         list_KKKj .append(KKKj)
    #
    #
    #     worker = DataFrame({"worker": list_worker})
    #     KKKj = DataFrame({"KKKj": list_KKKj })
    #
    #     data = concat([worker, KKKj], axis=1)
    #     data = data.sort_values('KKKj')
    #
    #
    #     collusion_worker = []  # 取前precentage的工人，认为是串谋工人
    #
    #     collusion_data = data.iloc[len(data) - pre:len(data), :]
    #
    #     for i in range(len(collusion_data)):
    #         collusion_worker.append(int(collusion_data.iloc[i][0]))
    #     collusion_worker = sorted(list(set(collusion_worker)))
    #
    #     (acc1, pre1, rec1, f1_score) = TPFN_v1(collusion_P, pre, epoch, collusion_worker)
    #
    #     return (acc1, pre1, rec1, f1_score)


def display_Percentage(collusion_P, percentage, alpha, beta, K, epoch):
    # 对于每一种串谋占比
    for pre in range(len(percentage)):

        Acc1 = []
        Pre1 = []
        Rec1 = []
        F1_score = []

        # 每一次实验（一共10次）
        for epo in range(epoch, epoch + 30):
            (acc1, pre1, rec1, f1_score) = Percentage(collusion_P, percentage[pre], alpha, beta, K, epo)

            Acc1.append(acc1)
            Pre1.append(pre1)
            Rec1.append(rec1)
            F1_score.append(f1_score)

        Acc1 = DataFrame({"Acc": Acc1})
        Pre1 = DataFrame({"Pre": Pre1})
        Rec1 = DataFrame({"Rec": Rec1})
        F1_score = DataFrame({"f1_score": F1_score})

        write1 = concat([Acc1, Pre1, Rec1, F1_score], axis=1)
        save_path = "../../result/collusion_data/collusion_0.8/a_b/a" + str(alpha) + "_b" + str(
            beta) + "KCD_Percentage_" + str(percentage[pre]) + "_" + str(epoch) + ".csv"
        write1.to_csv(save_path, sep=',', index=0)


if __name__ == '__main__':

    collusion_P = 0.8
    percentage = [36]

    alpha = [2, 4, 8, 16]
    beta = [0.03125]
    K = "KKK"  # 求KK的准确率

    for a in range(len(alpha)):
        for b in range(len(beta)):
            display_Percentage(collusion_P, percentage, alpha[a], beta[b], K, 0)
