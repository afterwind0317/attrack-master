import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
# //统计
from pandas import DataFrame, concat


def access(P):
    list = [11, 21, 32, 43, 55]

    for i in range(len(list)):
        save_path = "../s4-dog/collusion_" + str(P) + "/access_" + str(list[i]) + ".csv"
        list_id = []
        list_W = []
        list_S = []
        list_Si = []
        for j in range(30):
            list_id.append(j)
            W_data = pd.read_csv("../s4-dog/collusion_" + str(P) + "/experience_" + str(list[i]) + "/data" + str(
                j) + "_collaborate_W.csv")
            list_W.append(len(W_data))
            S_data = pd.read_csv("../s4-dog/collusion_" + str(P) + "/experience_" + str(list[i]) + "/data" + str(
                j) + "_collaborate_S.csv")
            list_S.append(len(S_data))
            Si_data = pd.read_csv("../s4-dog/collusion_" + str(P) + "/experience_" + str(list[i]) + "/data" + str(
                j) + "_collaborate_Si.csv")
            list_Si.append(len(Si_data))

        print(list_id)
        print(list_W)
        print(list_S)
        print(list_Si)

        id = DataFrame({"id": list_id})
        W = DataFrame({"W": list_W})
        S = DataFrame({"S": list_S})
        Si = DataFrame({"Si": list_Si})

        write = concat([id, W, S, Si], axis=1)
        write.to_csv(save_path, sep=',', index=0)


def salary():
    source_U = [1, 1, 1, 1, 1, 1]

    """collusion_0.6"""
    list = [11, 21, 32, 43, 55]

    Mean_U6 = [1]  # 普通工人平均收益PR
    Var_U6 = [0]  # 普通工人收益方差
    for i in range(len(list)):
        path = "../s4-dog/collusion_0.6/access_" + str(list[i]) + ".csv"
        data = pd.read_csv(path)

        U = []
        for j in range(len(data)):
            tmp_p = data.iloc[j][2] + data.iloc[j][3]
            tmp_e = data.iloc[j][2]
            U.append(tmp_p / tmp_e)
        Mean_U6.append(round(sum(U) / len(data), 4))
        Var_U6.append(round(np.std(U), 4))

    """collusion_0.7"""
    Mean_U7 = [1]  # 普通工人平均收益PR
    Var_U7 = [0]  # 普通工人收益方差
    for i in range(len(list)):
        path = "../s4-dog/collusion_0.7/access_" + str(list[i]) + ".csv"
        data = pd.read_csv(path)

        U = []
        for j in range(len(data)):
            tmp_p = data.iloc[j][2] + data.iloc[j][3]
            tmp_e = data.iloc[j][2]
            U.append(tmp_p / tmp_e)
        Mean_U7.append(round(sum(U) / len(data), 4))
        Var_U7.append(round(np.std(U), 4))

    """collusion_0.8"""
    Mean_U8 = [1]  # 普通工人平均收益PR
    Var_U8 = [0]  # 普通工人收益方差
    for i in range(len(list)):
        path = "../s4-dog/collusion_0.8/access_" + str(list[i]) + ".csv"
        data = pd.read_csv(path)

        U = []
        for j in range(len(data)):
            tmp_p = data.iloc[j][2] + data.iloc[j][3]
            tmp_e = data.iloc[j][2]
            U.append(tmp_p / tmp_e)
        Mean_U8.append(round(sum(U) / len(data), 4))
        Var_U8.append(round(np.std(U), 4))

    """collusion_0.9"""
    Mean_U9 = [1]  # 普通工人平均收益PR
    Var_U9 = [0]  # 普通工人收益方差
    for i in range(len(list)):
        path = "../s4-dog/collusion_0.9/access_" + str(list[i]) + ".csv"
        data = pd.read_csv(path)

        U = []
        for j in range(len(data)):
            tmp_p = data.iloc[j][2] + data.iloc[j][3]
            tmp_e = data.iloc[j][2]
            U.append(tmp_p / tmp_e)
        Mean_U9.append(round(sum(U) / len(data), 4))
        Var_U9.append(round(np.std(U), 4))

    """collusion_1.0.0"""
    Mean_U10 = [1]  # 普通工人平均收益PR
    Var_U10 = [0]  # 普通工人收益方差
    for i in range(len(list)):
        path = "../s4-dog/collusion_1.0/access_" + str(list[i]) + ".csv"
        data = pd.read_csv(path)

        U = []
        for j in range(len(data)):
            tmp_p = data.iloc[j][2] + data.iloc[j][3]
            tmp_e = data.iloc[j][2]
            U.append(tmp_p / tmp_e)
        Mean_U10.append(round(sum(U) / len(data), 4))
        Var_U10.append(round(np.std(U), 4))

    x_ticks = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']
    x = range(len(x_ticks))

    plt.plot(x, source_U, linewidth=2, label=u'normal worker')

    plt.errorbar(x, Mean_U6, Var_U6,
                 capsize=4, capthick=1,
                 linewidth=2, marker='x',
                 ms=7,
                 label=u'P=0.6')

    plt.errorbar(x, Mean_U7, Var_U7,
                 capsize=4, capthick=1,
                 linewidth=2, marker='+',
                 ms=7,
                 label=u'P=0.7')

    plt.errorbar(x, Mean_U8, Var_U8,
                 capsize=4, capthick=1,
                 linewidth=2, marker='x',
                 ms=7,
                 label=u'P=0.8')

    plt.errorbar(x, Mean_U9, Var_U9,
                 capsize=4, capthick=1,
                 linewidth=2, marker='o',
                 ms=7,
                 label=u'P=0.9')

    plt.errorbar(x, Mean_U10, Var_U10,
                 capsize=4, capthick=1,
                 linewidth=2, marker='o',
                 ms=7,
                 label=u'P=0.10')

    # label=r"$\frac{p_{0}}{e_{0}}$"
    # 新的实验.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'+str(label)+''))

    plt.legend()  # 让图例生效
    plt.xticks(x, x_ticks, fontsize=12)
    plt.yticks(fontsize=12)

    plt.tick_params(top=True, bottom=True, left=True, right=True)
    plt.tick_params(direction='in')

    plt.grid(axis='y', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)

    plt.xlim(-0.2, 5.2)
    plt.ylim(0, 6)

    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"Collusion Percentage(s4-dog)", size=14)  # X轴标签
    plt.ylabel("Utility", size=14)  # Y轴标签

    plt.tick_params(direction='in')

    path = 'Utility_s4-dog.jpg'

    plt.gcf().savefig(path, dpi=1400, format='jpg')

    plt.show()


if __name__ == '__main__':
    collusion_P = [0.9]
    #
    # for i in range(len(collusion_P)):
    #     access(collusion_P[i])

    salary()
