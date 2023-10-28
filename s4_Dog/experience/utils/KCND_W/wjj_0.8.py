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
"""$              求串谋矩阵                     $"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
def func(N, n):
    a = b = result = 1
    if N < n:
        print("n不能小于m 且均为整数")
    elif ((type(N) != int) or (type(n) != int)):
        print("n不能小于m 且均为整数")
    else:
        minNI = min(n, N - n)  # 使运算最简便
        for j in range(0, minNI):
            # 使用变量a,b 让所用的分母相乘后除以所有的分子
            a = a * (N - j)
            b = b * (minNI - j)
            result = a // b  # 在此使用“/”和“//”均可，因为a除以b为整数
        return result

# (串谋占比，实验epoch， 抄袭率， 相同答案数)
def E2(collusion_P, percentage, classN, epo):
    print("collusion_P=", collusion_P, "percentage=", percentage, "epo=", epo)
    # print("|")
    # 获取数据集
    path = "../../result/collusion_data/collusion_" + str(collusion_P) + "/experience_" + str(
        percentage) + "/data" + str(epo) + "_collaborate_D.csv"
    save_path = "../../result/collusion_data/collusion_" + str(collusion_P) + "/experience_" + str(
        percentage) + "/data" + str(epo) + "_collaborate_wjj.csv"
    data = pd.read_csv(path)

    # 工人去重
    worker = data['worker'].drop_duplicates().sort_values()
    worker = DataFrame({'worker': worker})

    # 初始化权重矩阵， 能力值， 概率
    W = np.zeros((len(worker), len(worker)))
    ability = 0.5 + (1 - 0.5) / 2
    fjj = ability ** 2 + (classN - 1) * (((1 - ability) / (classN - 1)) ** 2)

    # 计算每 两个工人之间的权重 Ew
    # 计算每 两个工人之间的权重 Ew
    for w in range(len(worker)):  # 每行的工人

        # 找出w的所有任务
        tmp_worker1 = worker.iloc[w][0]  # 当前工人
        tmp_works1 = data[(data.worker == tmp_worker1)]  # 当前工人的所有任务

        # print(tmp_works1)

        for wl in range(len(worker)):  # 每列的工人

            if w < wl:  # 排除自己和自己比

                tmp_worker2 = worker.iloc[wl][0]  # 当前工人
                tmp_works2 = data[(data.worker == tmp_worker2)]  # 当前工人的所有任务
                # print(tmp_works2)

                sim = []
                sim_answer = []
                for ii in range(len(tmp_works1)):
                    for jj in range(len(tmp_works2)):
                        if tmp_works1.iloc[ii][4] == tmp_works2.iloc[jj][4] and ii != jj:
                            sim.append(tmp_works1.iloc[ii][4])  # 任务并集
                            if tmp_works1.iloc[ii][6] == tmp_works2.iloc[jj][6]:
                                sim_answer.append(tmp_works1.iloc[ii][4])  # 相同答案并集

                n = len(sim_answer)  # 上
                N = len(sim)  # 下
                pre_n = N * fjj  # 概率上回答答案一样的个数

                if n > pre_n:  # 如果实际重复答案超过概率重复答案个数

                    Fjj = func(N, n) * fjj ** n * (1 - fjj) ** (N - n)  # Fjj的值可能小到0

                    wjj = 1 / Fjj

                    W[w][wl] = wjj

                # print("worker:", int(tmp_worker1), "做过的题目有：", len(tmp_works1), "个")
                # print("worker", int(tmp_worker2), "做过的题目有", len(tmp_works2), "个")
                # print("相同任务：", N)
                # print("概率上相同答案数：", pre_n)
                # print("相同答案：", n)
                # print("wjj=", wjj)
                # print("|")
                # print("|")

    np.savetxt(save_path, W, delimiter=",")
    print("W saved")
    print()

def access_Wh(collusion_P, percentage, classN, epoch):
    # collusion_P   # 抄袭概率
    # percentage   # 串谋占比
    for i in range(len(percentage)):
        for epo in range(epoch, epoch + 30):
            E2(collusion_P, percentage[i], classN, epo)


"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
"""$              求ks 生成【w1, w2, ks】        $"""
"""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"""
# 获取边的数量
def num_E(E, worker):
    # print(E)
    # print(worker)
    number_E = 0
    for i in range(len(E)):
        if E.iloc[i][0] == worker or E.iloc[i][1] == worker:
            number_E = number_E + 1

    return number_E

# 获取worker所有的边的权重之和
def get_sum_weight(E, worker):
    weights_workers = []

    for i in range(len(E)):
        if E.iloc[i][0] == worker or E.iloc[i][1] == worker:
            weights_workers.append(E.iloc[i][2])

    return sum(weights_workers)  # 返回该节点所有边的和

# 获取worker所有的边的权重平均值
def get_avg_weight(E, worker):
    weights_workers = []

    for i in range(len(E)):
        if E.iloc[i][0] == worker or E.iloc[i][1] == worker:
            weights_workers.append(E.iloc[i][2])

    return sum(weights_workers) / len(weights_workers)  # 返回该节点所有边的平均值

# 获取所有 节点邻居的的边的平均值 之和
def get_neighbor_sum_avg_weight(E, worker):
    # print("worker", worker)
    # 找出所有邻居
    list_neib = []  # 保存其邻居
    for i in range(len(E)):
        if E.iloc[i][0] == worker:
            list_neib.append(E.iloc[i][1])
        elif E.iloc[i][1] == worker:
            list_neib.append(E.iloc[i][0])
    # print("邻居有：", list_neib)

    # 计算每个邻居的边的权重之和
    all_neighbor_avg_weight = []  # 该节点 所有邻居 的边 的和

    for i in range(len(list_neib)):
        tmp_neib = list_neib[i]
        # print("tmp_neib=", tmp_neib)

        tmp_avg_weight = get_avg_weight(E, tmp_neib)  # tmp_beib 的 所有边权重平均值
        # print("tmp_sum_weight=", tmp_sum_weight)
        all_neighbor_avg_weight.append(tmp_avg_weight)

    # 返回所有 该节点 所有邻居节点 的边的平均值 之和
    return sum(all_neighbor_avg_weight)

# 获取所有节点邻居的的边的平均值
def get_neighbor_sum_weight(E, worker):
    pass

# 获取 每个节点的 ks
def get_ks_by_k_sehll(G):
    # 为了不改变原图
    graph = G.copy()

    k_shells = []
    # k从最小度开始
    degrees = (graph.degree[n] for n in graph.nodes())
    k = min(degrees)

    while nx.number_of_nodes(graph):

        node_k_shell = []

        nodes_degree = {n: graph.degree[n] for n in graph.nodes()}

        # 每次删除度值最小的节点而不能删除度为ks的节点否则产生死循环。https://neusncp.com/user/blog?id=242
        k_min = min(nodes_degree.values())

        # 是否还存在度为k_min的节点
        flag = True

        while (flag):
            nodes_degree = {n: graph.degree[n] for n in graph.nodes()}
            for ke, va in nodes_degree.items():

                if (va == k_min):
                    node_k_shell.append(ke)
                    graph.remove_node(ke)

            nodes_degree_check = {n: graph.degree[n] for n in graph.nodes()}

            # 检查图中是否存在度为kmin的节点
            if k_min not in nodes_degree_check.values():
                flag = False

        k_shells.append((k, node_k_shell))
        k += 1
    return k_shells

# 获得图的三元组(w1 ,w2, wh)
def get_worker_Graph(path):
    data = pd.read_csv(path, header=None)

    # data = Normalization(data)
    # print(data)

    list_w1 = []
    list_w2 = []
    list_wh = []
    for i in range(len(data)):
        for j in range(len(data)):
            if i < j and data.iloc[i][j] != 0:
                list_w1.append(i + 1)
                list_w2.append(j + 1)
                log_wh = data.iloc[i][j]
                list_wh.append(log_wh)
                # print(i+1, "-", j+1, ":", data.iloc[i][j])
    w1 = DataFrame({"w1": list_w1})
    w2 = DataFrame({"w2": list_w2})
    wh = DataFrame({"log_wh": list_wh})

    E = concat([w1, w2, wh], axis=1)
    return E




if __name__ == '__main__':
    collusion_P = 0.8  # 抄袭概率
    percentage = [11]  # 串谋占比
    classN = 4  # 四分类

    # 生成权重，生成图（这里决定了两个节点之间是否生成边），
    access_Wh(collusion_P, percentage, classN, 0)
