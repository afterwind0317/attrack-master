import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, concat
import seaborn as sns


# 读取小部分散列的csv 加到 data_drop_duplicates_1.csv
def sig_controller(alpha, beta):
    list_alpha = []
    list_beta = []
    list_avgAcc = []
    list_avgPre = []
    list_avgRec = []
    list_avgF1 = []

    for a in range(len(alpha)):
        for b in range(len(beta)):
            for g in range(len(beta)):
                path = "../../collusion_P_0.8/KCD(废)/a_b/a" + str(alpha[a]) + "_b" + str(
                    beta[b]) + "KCD_Percentage_36_0.csv"

                Accuracy = list(pd.read_csv(path)["Acc"])
                Precision = list(pd.read_csv(path)["Pre"])
                Recall = list(pd.read_csv(path)["Rec"])
                F1 = list(pd.read_csv(path)["f1_score"])

                list_alpha.append(alpha[a] / alpha[a])
                list_beta.append(beta[b] / alpha[a])
                list_avgAcc.append(sum(Accuracy) / 30)
                list_avgPre.append(sum(Precision) / 30)
                list_avgRec.append(sum(Recall) / 30)
                list_avgF1.append(sum(F1) / 30)

    pd_alpha = DataFrame({"alpha": list_alpha})
    pd_beta = DataFrame({"beta": list_beta})

    avgAcc = DataFrame({"avgAcc": list_avgAcc})
    avgPre = DataFrame({"avgPre": list_avgPre})
    avgRec = DataFrame({"avgRec": list_avgRec})
    avgF1 = DataFrame({"avgF1": list_avgF1})

    write = concat([pd_alpha, pd_beta, avgAcc, avgPre, avgRec, avgF1], axis=1)
    write = write.drop_duplicates()

    write.to_csv("data_drop_duplicates3.csv", float_format="%.5f", sep=',', index=0)


def pltData():
    data = pd.read_csv("data_drop_duplicates-plt.csv")

    beta = list(data["beta"])

    label_Acc = list(data["avgAcc"])
    label_Pre = list(data["avgPre"])
    label_Rec = list(data["avgRec"])
    label_F1 = list(data["avgF1"])


    plt.plot(beta, label_Acc,
             linewidth=2, marker='',
             ms=7,
             label=u'' + "Acc" + '')

    plt.plot(beta, label_Pre,
             linewidth=2, marker='',
             ms=7,
             label=u'' + "Pre" + '')

    plt.plot(beta, label_Rec,
             linewidth=2, marker='',
             ms=7,
             label=u'' + "Rec" + '')

    plt.plot(beta, label_F1,
             linewidth=2, marker='',
             ms=7,
             label=u'' + "F1" + '')

    plt.legend()  # 让图例生效
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(axis='y', alpha=0.2)
    plt.grid(axis='x', alpha=0.2)

    plt.tick_params(top=True, bottom=True, left=True, right=True)
    plt.tick_params(direction='in')


    plt.xlim(0, 0.05)
    plt.ylim(0.3, 1)

    plt.subplots_adjust(bottom=0.15)
    plt.title(r'$\alpha = 1$')
    plt.xlabel(r"$\beta$", size=18)  # X轴标签
    # plt.ylabel("Accuracy", size=14)  # Y轴标签


    path = 'KCD_hyper.jpg'
    plt.gcf().savefig(path, dpi=1400, format='jpg')

    plt.show()


if __name__ == '__main__':
    # 统计数据【alpha, beta, avgAcc, avgPre, avgRec, avgF1】
    # 【alpha, beta, avgAcc, avgPre, avgRec, avgF1】

    # alpha = [1]
    # beta = [0, 0.001, 0.002, 0.003, 0.004, 0.005]
    # sig_controller(alpha, beta)

    # alpha = [1,2,4,8,16,32]
    # beta = [1,2,4,8,16,32]
    # for i in range(len(alpha)):
    #     for i in range(len(beta)):
    #         sig_controller(alpha, beta)

    # 画图
    pltData()
