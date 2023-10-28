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
"""$              超参数                    $"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
def TPFN_v1(collusion_P, pre, epoch, _list):


    worker_pre = _list  # 预测的串谋工人

    worker_ture = []  # 真实串谋工人
    W_path = "../../result/collusion_data/collusion_"+str(collusion_P)+"/experience_"+str(pre)+"/data"+str(epoch)+"_collaborate_W.csv"
    worker_ture_pd = pd.read_csv(W_path)
    for i in range(len(worker_ture_pd)):
        worker_ture.append(worker_ture_pd.iloc[i][0])


    # print("预测串谋工人：", worker_pre)
    # print("真实串谋工人：", worker_ture)

    #  真实串谋工人的任务
    D_path = "../../result/collusion_data/collusion_"+str(collusion_P)+"/experience_"+str(pre)+"/data"+str(epoch)+"_collaborate_D.csv"
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
            df1 = works_ture.iloc[i:i+1, :]
            df2 = works_pre.iloc[j:j+1, :]
            if df1.equals(df2):
                TP = TP + 1

    FN = len(works_ture) - TP  # 真实情况是串谋工人的任务，被识别为正常工人的任务
    FP = len(works_pre) - TP  # 真实情况是正常工人的任务，被识别为串谋工人的任务
    TN = len(data) - TP - FP - FN  # 真实情况是正常工人的任务，被识别为真实工人的任务


    # print("TP：", TP, "/", len(works_ture))
    # print("FN=", FN, "/", len(works_ture))
    # print("FP=", FP, "/", len(data)-len(works_ture))
    # print("TN=", TN, "/", len(data)-len(works_ture))



    Accuracy= (TP + TN) / len(data)

    if TP + FP != 0:
        Precsion = TP / (TP + FP)
    else:
        Precsion = 0

    Recall= TP / (TP + FN)

    if Precsion + Recall != 0:
        f1_score = 2 * ((Precsion * Recall) / (Precsion + Recall))
    else:
        f1_score = 0

    print("Accuracy=", Accuracy)
    print("Precsion=", Precsion)
    print("Recall=", Recall)
    print("f1_score=", f1_score)

    return (Accuracy, Precsion, Recall, f1_score)


def Threshold(exp, epoch, threshold):
    """确定 串谋阈值 并找出串谋工人 进行验证 acc prec recall"""

    print("exp=", exp, "epoch=", epoch, "超参数=", threshold)

    path = "../result/Ks_shell/weight_"+str(exp)+"_"+str(epoch)+".csv"

    collusion_pd = pd.read_csv(path)

    collusion_worker = []

    for i in range(len(collusion_pd)):
        if collusion_pd.iloc[i][2] >= threshold:
            collusion_worker.append(int(collusion_pd.iloc[i][0]))

    collusion_worker = sorted(list(set(collusion_worker)))
    print("预测串谋工人：", collusion_worker)

    print("评价窗口1 ： acc recall....")
    (acc1, pre1, rec1, f1_score) = TPFN_v1(exp, epoch, collusion_worker)
    print(acc1, pre1, rec1, f1_score)
    print("------------------------------------------")

    return (acc1, pre1, rec1, f1_score)


def display_Threshold(x, threshold, epoch):

    Acc1 = []
    Pre1 = []
    Rec1 = []
    F1_score = []

    for exp in range(len(x)):
        for epo in range(epoch, epoch+30):
            (acc1, pre1, rec1, f1_score) = Threshold(x[exp], epo, threshold)

            Acc1.append(acc1)
            Pre1.append(pre1)
            Rec1.append(rec1)
            F1_score.append(f1_score)

    Acc1 = DataFrame({"Acc": Acc1})
    Pre1 = DataFrame({"Pre": Pre1})
    Rec1 = DataFrame({"Rec": Rec1})
    F1_score = DataFrame({"f1_score": F1_score})

    write1 = concat([Acc1, Pre1, Rec1, F1_score], axis=1)

    write1.to_csv("../../experience/result/ACC_PRE_REC/Algorithm1/threshold_"+str(threshold)+"/experience1_result_"+str(x[0])+"_"+str(epoch)+".csv", sep=',', index=0)




"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$              KCDN_top-K                        $"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
def Acc_Pre_Rec_F1_v1(exp, epoch, collusion_worker):
    # 串谋工人
    worker_pre = collusion_worker
    worker_ture = []
    worker_ture_pd = pd.read_csv("../../experience/result/experience_" + str(exp) + "/data" + str(epoch) + "_collaborate_W.csv")
    for i in range(len(worker_ture_pd)):
        worker_ture.append(worker_ture_pd.iloc[i][0])

    # print("预测串谋工人：", worker_pre)
    # print("真实串谋工人：", worker_ture)

    # 找出串谋工人所有的串谋任务
    # 找出每个工人所有的任务，
    data = pd.read_csv("../../experience/result/experience_" + str(exp) + "/data" + str(epoch) + "_collaborate_D.csv")

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

def Top(exp, k, epo):

    print(exp, k, epo)

    path = "../result/Ks_shell/weight_"+str(exp)+"_"+str(epo)+".csv"
    data = pd.read_csv(path)

    K = int(120 * k * 0.01)
    print("KCDN_top-", k, "->", K, "workers")

    collusion_worker = []


    collusion_data = data.iloc[len(data)-K:len(data), :]
    for i in range(len(collusion_data)):
        collusion_worker.append(int(collusion_data.iloc[i][0]))


    collusion_worker = sorted(list(set(collusion_worker)))
    print("预测串谋工人：", collusion_worker)

    print("评价窗口1 ： pre recall f1....")
    (pre1, rec1, f1_score) = Acc_Pre_Rec_F1_v1(exp, epo, collusion_worker)

    print("------------------------------------------")

    return (pre1, rec1, f1_score)

def display_Top(experience, k, epoch):

    Pre1 = []
    Rec1 = []
    F1_score = []

    for epo in range(epoch, epoch+20):

        (pre1, rec1, f1_score) = Top(experience, k, epo)

        Pre1.append(pre1)
        Rec1.append(rec1)
        F1_score.append(f1_score)


    Pre1 = DataFrame({"Pre": Pre1})
    Rec1 = DataFrame({"Rec": Rec1})
    F1_score = DataFrame({"f1_score": F1_score})


    write1 = concat([Pre1, Rec1, F1_score], axis=1)
    save_path1 = "../../experience/result/ACC_PRE_REC/Algorithm1/Top/top_"+str(k)+"_"+str(epoch)+".csv"
    write1.to_csv(save_path1, sep=',', index=0)




"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$              zhanbi                       $"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
def Percentage(collusion_P, pre, K, epoch):

    print("collusion_P=", collusion_P, "KCDN_percentage=", pre, "dection=", pre, "epoch=", epoch)

    # print(path)
    data = pd.read_csv("../../result/collusion_data/collusion_"+str(collusion_P)+"/experience_"+str(pre)+"/data"+str(epoch)+"_collaborate_"+str(K)+".csv")

    collusion_worker = []  # 取前precentage的工人，认为是串谋工人

    collusion_data = data.iloc[len(data)-pre:len(data), :]

    for i in range(len(collusion_data)):
        collusion_worker.append(int(collusion_data.iloc[i][0]))
    collusion_worker = sorted(list(set(collusion_worker)))


    (acc1, pre1, rec1, f1_score) = TPFN_v1(collusion_P, pre, epoch, collusion_worker)

    return (acc1, pre1, rec1, f1_score)


def display_Percentage(collusion_P, percentage, K, epoch):


    # 对于每一种串谋占比
    for pre in range(len(percentage)):

        Acc1 = []
        Pre1 = []
        Rec1 = []
        F1_score = []

        # 每一次实验（一共10次）
        for epo in range(epoch, epoch+1):
            (acc1, pre1, rec1, f1_score) = Percentage(collusion_P, percentage[pre], K, epo)

            Acc1.append(acc1)
            Pre1.append(pre1)
            Rec1.append(rec1)
            F1_score.append(f1_score)

        Acc1 = DataFrame({"Acc": Acc1})
        Pre1 = DataFrame({"Pre": Pre1})
        Rec1 = DataFrame({"Rec": Rec1})
        F1_score = DataFrame({"f1_score": F1_score})

        write1 = concat([Acc1, Pre1, Rec1, F1_score], axis=1)
        path = "../../result/collusion_data/collusion_"+str(collusion_P)+"/"+str(K)+"_percentage_"+str(percentage[pre])+"_"+str(epoch)+".csv"
        write1.to_csv(path, sep=',', index=0)





if __name__ == '__main__':


    collusion_P = 0.8
    percentage = [12]
    K = "KK"  # 求KK的准确率

    display_Percentage(collusion_P, percentage, K, 0)