"""
1 input data
2 output C
对每两个串谋工人取交集
对交集的任务进行编码
编码后用余弦夹角判断串谋
判断标准： 1 这两个权谋工人所有任务都是串谋数据
         2 这两个串谋工人的交集任务是串谋数据
"""
import numpy as np
import pandas as pd

import numpy as np


# 求余弦相似度
from pandas import DataFrame, concat


"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$              KCDN_top-K                        $"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

# 评价标准1
def TPFN_v1(collusion_P, pre, epoch, _list):

    worker_pre = _list  # 预测的串谋工人

    worker_ture = []  # 真实串谋工人

    W_path = "../../../experience/result/collusion_data/collusion_" + str(collusion_P) + "/experience_" + str(pre) + "/data" + str(
        epoch) + "_collaborate_W.csv"
    worker_ture_pd = pd.read_csv(W_path)
    for i in range(len(worker_ture_pd)):
        worker_ture.append(worker_ture_pd.iloc[i][0])

    print("预测串谋工人：", worker_pre)
    print("真实串谋工人：", worker_ture)

    #  真实串谋工人的任务
    D_path = "../../../experience/result/collusion_data/collusion_" + str(collusion_P) + "/experience_" + str(pre) + "/data" + str(
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

def Acc_Pre_Rec_F1_v1(exp, epoch, collusion_worker):
    # 串谋工人
    worker_pre = collusion_worker
    worker_ture = []
    worker_ture_pd = pd.read_csv("../../../experience/result/experience_" + str(exp) + "/data" + str(epoch) + "_collaborate_W.csv")
    for i in range(len(worker_ture_pd)):
        worker_ture.append(worker_ture_pd.iloc[i][0])

    print("预测串谋工人：", worker_pre)
    print("真实串谋工人：", worker_ture)

    # 找出串谋工人所有的串谋任务
    # 找出每个工人所有的任务，
    data = pd.read_csv("../../../experience/result/experience_" + str(exp) + "/data" + str(epoch) + "_collaborate_D.csv")

    # 找出真实串谋任务
    works_ture = data.iloc[:1, :]
    for i in range(len(worker_ture)):
        tmp_works = data[(data.worker == worker_ture[i])]
        works_ture = works_ture.append(tmp_works)
    works_ture = works_ture.iloc[1:, :]
    # print("真实串谋任务")
    # print(works_ture)

    # 找出预测串谋任务
    works_pre = data.iloc[:1, :]
    for i in range(len(worker_pre)):
        tmp_works = data[(data.worker == worker_pre[i])]
        works_pre = works_pre.append(tmp_works)
    works_pre = works_pre.iloc[1:, :]
    # print("预测串谋任务")
    # print(works_pre)


    TP = 0

    # 求 TP  FN
    for i in range(len(works_ture)):
        # print(i)
        for j in range(len(works_pre)):
            df1 = works_ture.iloc[i:i+1, :]
            df2 = works_pre.iloc[j:j+1, :]
            if df1.equals(df2):
                TP = TP + 1

    FN = len(works_ture) - TP
    FP = len(works_pre) - TP
    TN = len(data) - TP - FP - FN


    print("TP：", TP, "|", len(works_ture))
    print("FN=", FN, "|", len(works_ture))
    print("FP=", FP, "|", len(data)-len(works_ture))
    print("TN=", TN, "|", len(data)-len(works_ture))


    if TP + FP != 0:
        Precsion = TP / (TP + FP)
    else:
        Precsion = 0

    Recall= TP / (TP + FN)

    if Precsion + Recall != 0:
        f1_score = 2 * ((Precsion * Recall) / (Precsion + Recall))
    else:
        f1_score = 0


    print("Precsion=", Precsion)
    print("Recall=", Recall)
    print("f1_score", f1_score)



    return (Precsion, Recall, f1_score)


def Top(collusion_P, pre, top, epo):

    print("collusion_P=", collusion_P, "KCDN_percentage=", pre, "KCDN_top=", top, "epoch=", epo)

    weight_path = "../collusion_data/collusion_0.8/worker_weight/weight_"+str(pre)+"_"+str(epo)+".csv"

    data = pd.read_csv(weight_path)


    collusion_worker = []

    L = len(data)

    for i in range(len(data)):
        w1 = int(data.iloc[L - i - 1][0])
        w2 = int(data.iloc[L - i - 1][1])
        collusion_worker.append(w1)
        collusion_worker.append(w2)
        collusion_worker = sorted(list(set(collusion_worker)))
        if len(collusion_worker) >= top:
            break

    print(collusion_worker)


    collusion_worker = sorted(list(set(collusion_worker)))
    print("评价窗口1 ： pre recall f1....")
    (acc1, pre1, rec1, f1_score) = TPFN_v1(collusion_P, pre, epo, collusion_worker)

    print("------------------------------------------")
    return (acc1, pre1, rec1, f1_score)



def display_Top(collusion_P, percentage, tops, epoch):

    # 对于每一种串谋占比Top
    for t in range(len(tops)):
        Acc1 = []
        Pre1 = []
        Rec1 = []
        F1_score = []

        # 当前的占比
        tmp_top = int(tops[t] * 120)

        for epo in range(epoch, epoch + 15):
            (acc1, pre1, rec1, f1_score) = Top(collusion_P, percentage[0], tmp_top, epo)

            Acc1.append(acc1)
            Pre1.append(pre1)
            Rec1.append(rec1)
            F1_score.append(f1_score)

        Acc1 = DataFrame({"Acc": Acc1})
        Pre1 = DataFrame({"Pre": Pre1})
        Rec1 = DataFrame({"Rec": Rec1})
        F1_score = DataFrame({"f1_score": F1_score})

        write1 = concat([Acc1, Pre1, Rec1, F1_score], axis=1)
        save_path = "../collusion_data/collusion_0.8/top/collusion_detect_" + str(percentage[0]) + "_" + str(tops[t] * 100) + "%.csv"
        write1.to_csv(save_path, sep=',', index=0)


"""
Top
"""
if __name__ == '__main__':

    collusion_P = 0.8
    # 串谋占比
    percentage = [36]

    # 取串谋占比为30%时 求 pre rec f1-score
    tops = [0.05]
    display_Top(collusion_P, percentage, tops, 0)
