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

    tran_list = np.zeros(len(list) * 4)

    for i in range(len(list)):
        if list[i] == 0:
            tran_list[i * 4] = 0
            tran_list[i * 4 + 1] = 0
            tran_list[i * 4 + 2] = 0
            tran_list[i * 4 + 3] = 0

        elif list[i] == 1:
            tran_list[i * 4] = 0
            tran_list[i * 4 + 1] = 0
            tran_list[i * 4 + 2] = 0
            tran_list[i * 4 + 3] = 1

        elif list[i] == 2:
            tran_list[i * 4] = 0
            tran_list[i * 4 + 1] = 0
            tran_list[i * 4 + 2] = 1
            tran_list[i * 4 + 3] = 0

        elif list[i] == 3:
            tran_list[i * 4] = 0
            tran_list[i * 4 + 1] = 1
            tran_list[i * 4 + 2] = 0
            tran_list[i * 4 + 3] = 0

        elif list[i] == 4:
            tran_list[i * 4] = 1
            tran_list[i * 4 + 1] = 0
            tran_list[i * 4 + 2] = 0
            tran_list[i * 4 + 3] = 0

    return tran_list


# 计算每两个人之间的串谋权重
def compute_cos(collusion_P, percentage, epoch):
    print("collusion_P=", collusion_P, "percentag=", percentage, "epoch=", epoch)

    path = "../../../experience/result/collusion_data/collusion_" + str(collusion_P) + "/experience_" + str(
        percentage) + "/data" + str(epoch) + "_collaborate_D.csv"

    data = pd.read_csv(path)

    worker = data['worker'].drop_duplicates().sort_values()
    task = data['question'].drop_duplicates().sort_values()

    # 获取 R 回答矩阵
    R = np.zeros((len(worker), len(task)))

    for i in range(len(data)):
        w = data.iloc[i][5]
        t = data.iloc[i][4]
        # print("worker:", int(w), "task:", int(t), "answer:", data.iloc[i][6])
        R[int(w - 1)][int(t - 1)] = int(data.iloc[i][6] + 1)

    # 获取C 串谋矩阵
    worker1 = []
    worker2 = []
    collusion_w = []

    for i in range(len(R)):

        for i2 in range(len(R)):

            if i < i2:
                R1 = transform(R[i])  # 转化为向量

                R2 = transform(R[i2])  # 转化为向量

                w = cosine_similarity(R1, R2)
                # print(w)
                worker1.append(i + 1)
                worker2.append(i2 + 1)
                collusion_w.append(w)

                # print(i+1, i2+1, w)

    worker1 = DataFrame({"worker1": worker1})
    worker2 = DataFrame({"worker2": worker2})
    w = DataFrame({"w": collusion_w})

    write_w = concat([worker1, worker2, w], axis=1)
    write_w = write_w.sort_values(by=["w"])

    # print(write_w)

    save_path = "../collusion_data/collusion_0.8/worker_weight/weight_" + str(percentage) + "_" + str(epoch) + ".csv"
    write_w.to_csv(save_path, sep=',', index=0)


# compute_cos 控制台
def display_Cos(collusion_P, percentage, epoch):
    for per in range(len(percentage)):

        for epo in range(epoch, epoch + 30):
            compute_cos(collusion_P, percentage[per], epo)


"""
求余弦夹角
"""
if __name__ == '__main__':
    collusion_P = 0.8
    # 串谋占比
    percentage = [11, 21, 32, 43, 55]

    # 余弦夹角
    display_Cos(collusion_P, percentage, 0)
