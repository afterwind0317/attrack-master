
"""
START：

欺诈团队：T
欺诈团队工人数：k
欺诈团队工人：t1，t2，。。tk
欺诈团队数据库:S

众包平台 P
众包任务 ai

众包平台P 发布任务 a1
T选择性的接受一个任务ai：
    查找ai 是否在S中，如果在：
        将ai的问题答案返回P
    如果不在：
        随机挑选一个工人 ti，并回答问题返回P
        将问题记录在S当中5

"""
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
import time




# 模拟工程 （原数据集， k：人数，  exp：实验epoch）
def process(Data, k, exp, collusion_P):

    # number = len(np.array(data["worker"].drop_duplicates()))
    # print("本次实验一共有：", number, "个worker")

    # 初始化团队 K
    K = np.array(Data["worker"].drop_duplicates().sample(n=k))
    K = sorted(K)
    # K = [1,13,23,32,50,76,77,81,86,103,105,113]
    print(collusion_P, "--", exp, "/", 10, "--", k)
    print("******************************************************")
    print(K)
    print("../../result/collusion_data/collusion_"+str(collusion_P)+"/experience_"+str(k)+"/data"+str(exp)+"_collaborate_W.csv")  # 工人
    print("../../result/collusion_data/collusion_"+str(collusion_P)+"/experience_"+str(k)+"/data"+str(exp)+"_collaborate_D.csv")  # 团伙参与之后的数据
    print("../../result/collusion_data/collusion_"+str(collusion_P)+"/experience_"+str(k)+"/data"+str(exp)+"_collaborate_S.csv")  # 服务器数据库
    print("../../result/collusion_data/collusion_"+str(collusion_P)+"/experience_"+str(k)+"/data"+str(exp)+"_collaborate_Si.csv")  # 服务器数据中被重新调用的数据
    print("******************************************************")



    # 模拟团伙的数据库
    S = pd.DataFrame(dict(day=[0], hour=[0], minute=[0], second=[0], question=[0], worker=[0], answer=[0], gt=[0]))

    # 经过服务器直接回答了的问题
    Si = pd.DataFrame(dict(day=[0], hour=[0], minute=[0], second=[0], question=[0], worker=[0], answer=[0], gt=[0]))

    # 模拟新生成的数据集
    D2 = pd.DataFrame(dict(day=[0], hour=[0], minute=[0], second=[0], question=[0], worker=[0], answer=[0], gt=[0]))



    """
    process START:
    """
    # 遍历任务，展开答题
    for i in range(len(Data)):
        # print(i)
        # print("当前任务id:", str(data.iloc[i][4]))
        time.sleep(1)

        flag1 = 0  # 标记当前任务，0表示团队不回答该任务

        # 便利串谋团队
        for j in range(len(K)):

            # # 如果其中有工人回答这个任务
            if Data.iloc[i][5] == K[j]:

                # 该团队 有人回答此问题
                flag1 = 1
                # print("当前串谋工人", K[j])

                # 选择是否抄袭
                is_collusion = np.random.rand(1)
                # print("随机数：", is_collusion)

                if is_collusion > collusion_P:
                    # print("不可抄袭",  K[j], "回答问题", data.iloc[i][5], "回答", data.iloc[i][6], ", 记录在D2,同时记录在S")
                    # print()
                    D2 = D2.append(Data.iloc[i])
                    S = S.append(Data.iloc[i])

                else:
                    # print("允许抄袭")

                    # 便利数据库S
                    answer_set = []  # 保存的答案集合
                    for s in range(len(S)):  # 便利数据库
                        if S.iloc[s][4] == Data.iloc[i][4]:  # 如果数据库的任务id == 当前任务id
                            # print("服务器有记录")
                            answer_set.append(S.iloc[s][6])  # 当前任务answer = 数据库任务answer
                    # 如果数据库有这个任务的答案，便抄袭，如果没有答案记录，自己做
                    if len(answer_set) != 0:

                        # print("数据库查询成功, 答案集合为：", answer_set)
                        mv_answer = max(answer_set, key=lambda v: answer_set.count(v))  # 取答案最多的，第一个
                        tmp_data = Data.iloc[i]
                        tmp_data[6] = mv_answer
                        D2 = D2.append(tmp_data)
                        Si = Si.append(tmp_data)

                    else:
                        # print("数据库没有找到该任务,由", data.iloc[i][5], "回答", data.iloc[i][6], "记录在D2,同时记录在S")
                        # print()
                        D2 = D2.append(Data.iloc[i])
                        S = S.append(Data.iloc[i])



        # 该团队不接受这个任务
        if flag1 == 0:
            tmp_worker = Data.iloc[i][5]
            # print("当前正常工人", tmp_worker, "回答记录在D2")
            # print()
            D2 = D2.append(Data.iloc[i])

        # print("****************************************")




    W = DataFrame({"worker": K})

    W.to_csv("../../result/collusion_data/collusion_"+str(collusion_P)+"/experience_"+str(k)+"/data"+str(exp)+"_collaborate_W.csv", sep=',', index=0)  # 工人

    D2 = D2.iloc[1:, :]
    D2.to_csv("../../result/collusion_data/collusion_"+str(collusion_P)+"/experience_"+str(k)+"/data"+str(exp)+"_collaborate_D.csv", sep=',', index=0)  # 团伙参与之后的数据

    S = S.iloc[1:, :]
    S.to_csv("../../result/collusion_data/collusion_"+str(collusion_P)+"/experience_"+str(k)+"/data"+str(exp)+"_collaborate_S.csv", sep=',', index=0)  # 服务器数据库

    Si = Si.iloc[1:, :]
    Si.to_csv("../../result/collusion_data/collusion_"+str(collusion_P)+"/experience_"+str(k)+"/data"+str(exp)+"_collaborate_Si.csv", sep=',', index=0)  # 服务器数据中被重新调用的数据

    print("save success")
    # print()






if __name__ == '__main__':

    percentage = [60]
    collusion_P = 1  # 串谋概率

    path1 = "../../../data/TimeData.csv"
    # path1 = "test.csv"
    """
    模拟平台发布任务 工人接受任务（团伙参与）后生成 W  D2  S  Si的过程"""


    for pre in range(len(percentage)):
        for epo in range(10, 11):

            sourceData = pd.read_csv(path1)
            # time.sleep(5)

            process(sourceData, percentage[pre], epo, collusion_P)