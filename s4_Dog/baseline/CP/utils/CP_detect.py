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

"""
***********************************************************************************
***********************************************************************************
***********************************************************************************
***********************************************************************************
"""

"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$              求WPCR                       $"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# 去掉之前的索引 重新排序
def sort_index(data, name):
    list_ = []
    for i in range(len(data)):
        list_.append(data.iloc[i])

    data = DataFrame({name: list_})
    return data

# 计算Var
def get_Var(data):

    worker = data['worker'].drop_duplicates().sort_values()
    task = data['question'].drop_duplicates().sort_values()
    worker = sort_index(worker, "worker")
    task = sort_index(task, "task")


    # print(data)
    # print(worker)
    # print(task)

    mv_gt = []

    # 投票方法求出每一个task的mv_label
    for t in range(len(task)):
        answer = []
        current_tasks = data[(data.question == task.iloc[t][0])]
        # print(current_tasks)
        for c in range(len(current_tasks)):
            answer.append(current_tasks.iloc[c][6])

        counts = np.bincount(answer)
        gt = np.argmax(counts)
        mv_gt.append(gt)

    mv_gt = DataFrame({"mv_gt": mv_gt})

    mv_R = concat([task, mv_gt], axis=1)

    # print(mv_R)

    Pi = []

    for i in range(len(worker)):  # 循环求每一个工作者的  Pi
        tmp_worker = worker.iloc[i][0]
        # print("tmp_worker:", tmp_worker)
        tmp_worker_works = data[(data.worker == tmp_worker)]  # 当前工作者的所有任务
        # print(tmp_worker_works)

        fz = 0
        for j in range(len(tmp_worker_works)):  # 循环该工人的每一个任务
            tmp_question = tmp_worker_works.iloc[j][4]
            # print(tmp_question)
            # 找出 当前问题的投票 标签
            mv_r = mv_R[(mv_R.task == tmp_question)].iloc[0][1]
            # 如果自己的回答 与 投票标签一致 fz++
            if mv_r == tmp_worker_works.iloc[j][6]:
                fz = fz + 1

        Pi.append(fz/len(tmp_worker_works))

    EPi = np.mean(Pi)
    Pi = DataFrame({"Pi": Pi})

    # print(Pi)
    #
    # print(EPi)

    # 计算Var
    sum = 0
    for i in range(len(Pi)):
        sum = sum + (Pi.iloc[i][0] - EPi) * (Pi.iloc[i][0] - EPi)

    # print("sum=", sum)

    Var = (1/len(worker)) * sum

    # print("Var=", Var)

    return Var

# 删除某公认的回答集
def del_works(data, worker):

    worker_data = data[(data.worker == worker)]
    # print(worker_data)

    index = worker_data.index.values

    data = data.drop(index, axis=0)
    # print(data)

    return  data

# 生成【worker， WPCR】文件
def compute_cpp(collusion_P, percentage,  epoch):


    print("collusion_P=", collusion_P, "percentag=", percentage,"epoch=", epoch)

    path = "../../../experience/result/collusion_data/collusion_"+str(collusion_P)+"/experience_"+str(percentage)+"/data"+str(epoch)+"_collaborate_D.csv"


    data = pd.read_csv(path)
    # print(data)

    worker = data['worker'].drop_duplicates().sort_values()
    task = data['question'].drop_duplicates().sort_values()
    worker = sort_index(worker, "worker")
    task = sort_index(task, "task")


    # 首先求一下原数据集的Var
    VarP = get_Var(data)

    list_WPCR = []

    # 计算去掉每一个工人后的Var
    for i in range(len(worker)):
        # 去掉该工人的 回答集 后的数据
        L_k = del_works(data, worker.iloc[i][0])
        VarP_k = get_Var(L_k)
        # print("Var_"+str(i)+":", VarP)

        WPCR_k = abs(VarP_k - VarP)
        list_WPCR.append(WPCR_k)

        # print("WPCR_"+str(i)+":", WPCR_k)


    WPCR = DataFrame({"WPCR": list_WPCR})

    # 保存到csv【worker， Var, WPCR】
    worker_WPCR = concat([worker, WPCR], axis=1)

    worker_WPCR = worker_WPCR.sort_values('WPCR')

    save_path= "../collusion_data/collusion_0.8/WPCR/wpcr_"+str(percentage)+"_"+str(epoch)+".csv"
    worker_WPCR.to_csv(save_path, sep=',', index=0)

# compute_cpp 控制台
def display_CPP(collusion_P, percentage,  epoch):

    for exp in range(len(percentage)):
        for epoch in range(epoch, epoch+30):
            compute_cpp(collusion_P, percentage[exp], epoch)



"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$              根据top-k来求串谋任务             $"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""

def Acc_Pre_Rec_F1_v1(exp, epoch, collusion_worker):
    # 串谋工人
    worker_pre = collusion_worker
    worker_ture = []
    worker_ture_pd = pd.read_csv("../../../experience/result/experience_" + str(exp) + "/data" + str(epoch) + "_collaborate_W.csv")
    for i in range(len(worker_ture_pd)):
        worker_ture.append(worker_ture_pd.iloc[i][0])

    # print("预测串谋工人：", worker_pre)
    # print("真实串谋工人：", worker_ture)

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
    f1_score = 2 * ((Precsion * Recall) / (Precsion + Recall))


    print("Precsion=", Precsion)
    print("Recall=", Recall)
    print("f1_score", f1_score)



    return (Precsion, Recall, f1_score)


def Top(exp, k, epo):

    print(exp, k, epo)

    path = "../result/WPCR/wpcr_"+str(exp)+"_"+str(epo)+".csv"
    data = pd.read_csv(path)

    K = int(109 * k * 0.01)
    print("KCDN_top-", k, "->", K, "workers")

    collusion_worker = []


    collusion_data = data.iloc[len(data)-K:len(data), :]
    for i in range(len(collusion_data)):
        collusion_worker.append(int(collusion_data.iloc[i][0]))


    print(collusion_worker)


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

    for epo in range(epoch, epoch+1):

        (pre1, rec1, f1_score) = Top(experience, k, epo)

        Pre1.append(pre1)
        Rec1.append(rec1)
        F1_score.append(f1_score)


    Pre1 = DataFrame({"Pre": Pre1})
    Rec1 = DataFrame({"Rec": Rec1})
    F1_score = DataFrame({"f1_score": F1_score})



    write1 = concat([Pre1, Rec1, F1_score], axis=1)
    save_path1 = "../../CP/result/top_"+str(k)+"/experience1_result_"+str(epoch)+".csv"
    write1.to_csv(save_path1, sep=',', index=0)





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

    W_path = "../../../experience/result/collusion_data/collusion_" + str(collusion_P) + "/experience_" + str(per) + "/data" + str(epoch) + "_collaborate_W.csv"
    worker_ture_pd = pd.read_csv(W_path)
    for i in range(len(worker_ture_pd)):
        worker_ture.append(worker_ture_pd.iloc[i][0])

    # print("预测串谋工人：", worker_pre)
    # print("真实串谋工人：", worker_ture)

    # 找出串谋工人所有的串谋任务
    # 找出每个工人所有的任务，
    D_path = "../../../experience/result/collusion_data/collusion_" + str(collusion_P) + "/experience_" + str(per) + "/data" + str(epoch) + "_collaborate_D.csv"
    data = pd.read_csv(D_path)

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

    path = "../collusion_data/collusion_"+str(collusion_P)+"/WPCR/wpcr_"+str(per)+"_"+str(epo)+".csv"
    data = pd.read_csv(path)


    collusion_worker = []

    collusion_data = data.iloc[len(data)-per:len(data), :]
    for i in range(len(collusion_data)):
        collusion_worker.append(int(collusion_data.iloc[i][0]))
    collusion_worker = sorted(list(set(collusion_worker)))

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

            (acc1, pre1, rec1,f1_score) = ZB(collusion_P, percentage[per], epo)

            Acc1.append(acc1)
            Pre1.append(pre1)
            Rec1.append(rec1)
            F1_score.append(f1_score)


        Acc1 = DataFrame({"Acc": Acc1})
        Pre1 = DataFrame({"Pre": Pre1})
        Rec1 = DataFrame({"Rec": Rec1})
        F1_score = DataFrame({"f1_score": F1_score})

        write1 = concat([Acc1, Pre1, Rec1, F1_score], axis=1)

        save_path = "../../collusion_data/collusion_0.8/KCDN_percentage/collusion_detect_" + str(percentage[per]) + ".csv"
        write1.to_csv(save_path, sep=',', index=0)





if __name__ == '__main__':



    collusion_P = 0.8
    # 串谋占比
    percentage = [11]

    # 生成每一个工人的WPCR
    # display_CPP(collusion_P, percentage, 0)



    # 串谋占比对比试验
    display_ZB(collusion_P, percentage, 0)



    # 依据阈值来生成串谋工人并且计算其ACC PRE REC
    # display_ACC_PRE_REC(x, threshold, 0)

    # k = 5  # KCDN_top-10%
    # exp = 32  # 取串谋占比为30%时
    # display_Top(exp, k, 0)
