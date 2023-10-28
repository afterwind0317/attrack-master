import math

import pandas as pd
from pandas import concat
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def TPFN_v1(collusion_P, pre, epoch, _list):
    worker_pre = _list  # 预测的串谋工人

    worker_ture = []  # 真实串谋工人
    W_path = "../../result/collusion_data/collusion_" + str(collusion_P) + "/experience_" + str(pre) + "/data" + str(
        epoch) + "_collaborate_W.csv"
    worker_ture_pd = pd.read_csv(W_path)
    for i in range(len(worker_ture_pd)):
        worker_ture.append(worker_ture_pd.iloc[i][0])

    print("预测串谋工人：", worker_pre)
    print("真实串谋工人：", worker_ture)

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



def Top(collusion_P, pre, top, alpha, beta, epoch):
    print("collusion_P=", collusion_P, "alpha=", alpha, "beta=", beta, "KCD_top=", top, "epoch=", epoch)

    # print(path)
    path = "../../result/collusion_data/collusion_" + str(collusion_P) + "/experience_" + str(pre) + "/data" + str(
        epoch) + "_collaborate_worker_ks_sumW_sumWN.csv"
    data = pd.read_csv(path)
    # print("Acc Rec Pre F1")
    # 如果是计算APRF

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
    KKj = DataFrame({"KKKj": list_KKj})

    data = concat([worker, KKj], axis=1)
    data = data.sort_values('KKKj')

    # print(data)

    collusion_worker = []  # 取前precentage的工人，认为是串谋工人

    collusion_data = data.iloc[len(data) - top:len(data), :]

    for i in range(len(collusion_data)):
        collusion_worker.append(int(collusion_data.iloc[i][0]))
    collusion_worker = sorted(list(set(collusion_worker)))

    (acc1, pre1, rec1, f1_score) = TPFN_v1(collusion_P, pre, epoch, collusion_worker)

    return (acc1, pre1, rec1, f1_score)


def display_Top(collusion_P, percentage, tops, alpha, beta, epoch):
    # 对于每一种串谋占比Top
    for t in range(len(tops)):

        Acc1 = []
        Pre1 = []
        Rec1 = []
        F1_score = []

        tmp_top = int(tops[t] * 109)

        # 每一次实验（一共10次）
        for epo in range(epoch, epoch + 30):
            (acc1, pre1, rec1, f1_score) = Top(collusion_P, percentage[0], tmp_top, alpha, beta, epo)

            Acc1.append(acc1)
            Pre1.append(pre1)
            Rec1.append(rec1)
            F1_score.append(f1_score)

        Acc1 = DataFrame({"Acc": Acc1})
        Pre1 = DataFrame({"Pre": Pre1})
        Rec1 = DataFrame({"Rec": Rec1})
        F1_score = DataFrame({"f1_score": F1_score})

        write1 = concat([Acc1, Pre1, Rec1, F1_score], axis=1)
        save_path = "../../result/collusion_data/collusion_0.8/KCD_top/collusion_detect_"+str(percentage[0])+"_"+str(tmp_top)+"%.csv"
        write1.to_csv(save_path, sep=',', index=0)


if __name__ == '__main__':


    collusion_P = 0.8
    percentage = [32]

    alpha = [1]
    beta = [0.01]

    tops = [0.15]


    display_Top(collusion_P, percentage, tops, alpha[0], beta[0], 0)
