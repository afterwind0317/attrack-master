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
"""$              求相似度 并且排序               $"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# 计算余弦夹角
def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0
    return num / denom
# 矩阵转换
def transform(list):

    # print("原列表：", len(list))

    tran_list = np.zeros(len(list)*4)

    for i in range(len(list)):
        if list[i] == 0:
            tran_list[i * 3] = 0
            tran_list[i * 3 + 1] = 0
            tran_list[i * 3 + 2] = 0
            tran_list[i * 3 + 3] = 0

        elif list[i] == 1:
            tran_list[i * 3] = 0
            tran_list[i * 3 + 1] = 0
            tran_list[i * 3 + 2] = 0
            tran_list[i * 3 + 3] = 1

        elif list[i] == 2:
            tran_list[i * 3] = 0
            tran_list[i * 3 + 1] = 0
            tran_list[i * 3 + 2] = 1
            tran_list[i * 3 + 3] = 0

        elif list[i] == 3:
            tran_list[i * 3] = 0
            tran_list[i * 3 + 1] = 1
            tran_list[i * 3 + 2] = 0
            tran_list[i * 3 + 3] = 1

        elif list[i] == 4:
            tran_list[i * 3] = 1
            tran_list[i * 3 + 1] = 0
            tran_list[i * 3 + 2] = 0
            tran_list[i * 3 + 3] = 0


    return tran_list
# 计算每两个人之间的串谋权重
def compute_cos(collusion_P, percentage, epoch):

    print("collusion_P=", collusion_P, "percentag=", percentage,"epoch=", epoch)

    path = "../../../experience/result/collusion_data/collusion_"+str(collusion_P)+"/experience_"+str(percentage)+"/data"+str(epoch)+"_collaborate_D.csv"

    data = pd.read_csv(path)

    worker = data['worker'].drop_duplicates().sort_values()
    task = data['question'].drop_duplicates().sort_values()

    # 获取 R 回答矩阵
    R = np.zeros((len(worker), len(task)))

    for i in range(len(data)):
        w = data.iloc[i][5]
        t = data.iloc[i][4]
        # print("worker:", int(w), "task:", int(t), "answer:", data.iloc[i][6])
        R[int(w-1)][int(t-1)] = int(data.iloc[i][6]+1)


    # 获取C 串谋矩阵
    worker1 = []
    worker2 = []
    collusion_w = []


    for i in range(len(R)):

        for i2 in range(len(R)):

            if i < i2:

                R1 = transform(R[i])  #转化为向量

                R2 = transform(R[i2])  #转化为向量

                w = cosine_similarity(R1, R2)
                # print(w)
                worker1.append(i+1)
                worker2.append(i2+1)
                collusion_w.append(w)

                # print(i+1, i2+1, w)

    worker1 = DataFrame({"worker1":worker1})
    worker2 = DataFrame({"worker2": worker2})
    w = DataFrame({"w": collusion_w})

    write_w = concat([worker1, worker2, w], axis=1)
    write_w = write_w.sort_values(by=["w"])

    # print(write_w)

    save_path = "../collusion_data/collusion_0.8/worker_weight/weight_"+str(percentage)+"_"+str(epoch)+".csv"
    write_w.to_csv(save_path, sep=',', index=0)
# compute_cos 控制台
def display_Cos(collusion_P, percentage, epoch):

    for per in range(len(percentage)):

        for epo in range(epoch, epoch+30):

            compute_cos(collusion_P, percentage[per], epo)




"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$              根据占比求串谋工人              $"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# 找出团队的任务
def process(data, worker_list):

    K = worker_list
    Data = data

    # 模拟团伙的数据库
    S = pd.DataFrame(dict(day=[0], hour=[0], minute=[0], second=[0], question=[0], worker=[0], answer=[0], gt=[0]))

    """
    process START:
    """
    for i in range(len(Data)):  # 任务


        flag1 = 0  # 标记当前任务，0表示团队不回答该任务

        # 如果该团队回答该问题
        for j in range(len(K)):  # 团队是否选择回答该问题
            # 遍历工人集合 看看谁回答了此问题
            if Data.iloc[i][5] == K[j]:  # 如果有 回答

                # 该团队 有人回答此问题
                flag1 = 1
                # print("该团队", K[j], "打算回答该任务,首先查找数据库")

                flag2 = 0  #  0代表 数据库无记录

                # 便利数据库S 有无该数据记录
                for s in range(len(S)):
                    # 数据库有
                    if S.iloc[s][4] == Data.iloc[i][4]:
                        # print("服务器--->S")
                        # print("找到了!")
                        flag2 = 1  # 数据库有记录

                if flag2 == 0:  # 如果数据库无记录
                    # print("数据库没有找到该任务,由", K[j], "回答问题, 记录在D2,同时记录在S")
                    S = S.append(Data.iloc[i])

        # 该团队不接受这个任务
        if flag1 == 0:
            # print("该团队选择不回答该任务, 该任务由其余的工人回答,并且记录在D2")
            pass

    S = S.iloc[1:, :]

    return S

# 评价标准1
def TPFN_v1(per, epoch, _list):
    # 串谋工人
    worker_pre = _list
    worker_ture = []
    W_path = "../../../experience/result/collusion_data/collusion_"+str(collusion_P)+"/experience_"+str(per)+"/data"+str(epoch)+"_collaborate_W.csv"
    worker_ture_pd = pd.read_csv(W_path)
    for i in range(len(worker_ture_pd)):
        worker_ture.append(worker_ture_pd.iloc[i][0])

    # print("预测串谋工人：", worker_pre)
    # print("真实串谋工人：", worker_ture)

    # 找出串谋工人所有的串谋任务
    # 找出每个工人所有的任务，
    data = pd.read_csv("../../../experience/result/collusion_data/collusion_"+str(collusion_P)+"/experience_"+str(per)+"/data"+str(epoch)+"_collaborate_D.csv")

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


    print("TP：", TP, "/", len(works_ture))
    print("FN=", FN, "/", len(works_ture))
    print("FP=", FP, "/", len(data)-len(works_ture))
    print("TN=", TN, "/", len(data)-len(works_ture))



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

# 根据阈值求串谋人员
def ZB(collusion_P, per, epo):

    print("FC: collusion_P=", collusion_P, "KCDN_percentage=", per, "epoch", epo)


    path = "../collusion_data/collusion_0.8/worker_weight/weight_"+str(per)+"_"+str(epo)+".csv"
    data = pd.read_csv(path)

    L = len(data)
    collusion_worker = []

    for i in range(len(data)):
        w1 = int(data.iloc[L - i - 1][0])
        w2 = int(data.iloc[L - i - 1][1])
        collusion_worker.append(w1)
        collusion_worker.append(w2)

        collusion_worker = sorted(list(set(collusion_worker)))

        if len(collusion_worker) >= per:
            break

    collusion_worker = sorted(list(set(collusion_worker)))
    print("预测串谋工人：", collusion_worker)

    print("评价窗口1 ： acc pre recall....")
    (acc1, pre1, rec1, f1_score) = TPFN_v1(per, epo, collusion_worker)

    print("------------------------------------------")

    return (acc1, pre1, rec1, f1_score)


def display_ZB(collusion_P, percentage, epoch):

    for per in range(len(percentage)):

        Acc1 = []
        Pre1 = []
        Rec1 = []
        F1_score = []

        for epo in range(epoch, epoch+30):
            (acc1, pre1, rec1, f1_score) = ZB(collusion_P, percentage[per], epo)

            Acc1.append(acc1)
            Pre1.append(pre1)
            Rec1.append(rec1)
            F1_score.append(f1_score)

        Acc1 = DataFrame({"Acc": Acc1})
        Pre1 = DataFrame({"Pre": Pre1})
        Rec1 = DataFrame({"Rec": Rec1})
        F1_score = DataFrame({"f1_score": F1_score})

        write1 = concat([Acc1, Pre1, Rec1, F1_score], axis=1)

        save_path = "../collusion_data/collusion_0.8/percentage/collusion_detect_"+str(percentage[per])+".csv"
        write1.to_csv(save_path, sep=',', index=0)




"""
Top
"""
if __name__ == '__main__':

    collusion_P = 0.8
    # 串谋占比
    percentage = [11]

    #余弦夹角
    # display_Cos(collusion_P, percentage, 0)

    display_ZB(collusion_P, percentage, 0)
