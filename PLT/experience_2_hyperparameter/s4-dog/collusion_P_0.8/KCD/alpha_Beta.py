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

                list_alpha.append(alpha[a]/ alpha[a])
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

    write.to_csv("data_drop_duplicates.csv", float_format="%.5f", sep=',', index=0)


def pltData(target):

    data = pd.read_csv("data_drop_duplicates2.csv")

    beta = list(data["beta"])

    label = list(data[target])

    print(beta)
    print(target)

    plt.plot(beta, label,
             linewidth=1.5, marker='.',
             ms=7,
             label=u'.' + target + '.')

    plt.legend()  # 让图例生效
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y')

    plt.xlim(0, 0.025)

    if target == "avgAcc":
        plt.ylim(0.6, 0.8)
    elif target == "avgRec":
        plt.ylim(0.90, 1)
    elif target == "avgPre":
        plt.ylim(0.4, 0.8)
    elif target == "avgF1":
        plt.ylim(0.6, 0.8)

    plt.subplots_adjust(bottom=0.15)
    plt.title(u"aplha = 1")
    plt.xlabel(u"beta", size=18)  # X轴标签

    plt.tick_params(direction='in')
    plt.tick_params(top=True, bottom=True, left=True, right=True)

    path = 'experience2_synthetic_Acc.jpg'
    plt.gcf().savefig(path, dpi=1800, format='jpg')

    plt.show()


if __name__ == '__main__':
    # 统计数据【alpha, beta, avgAcc, avgPre, avgRec, avgF1】
    # 【alpha, beta, avgAcc, avgPre, avgRec, avgF1】

    alpha = [2, 4, 8, 16, 32, 64]
    beta = [0.03125]
    sig_controller(alpha, beta)





    # 画图
    pltData("avgAcc")
    pltData("avgPre")
    pltData("avgRec")
    pltData("avgF1")
