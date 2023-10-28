import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pandas import DataFrame, concat


def Mean_Var(Alg):
    index = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

    Mean_acc = []
    Var_acc = []

    Mean_pre = []
    Var_pre = []

    Mean_rec = []
    Var_rec = []

    Mean_f1 = []
    Var_f1 = []

    for i in range(len(index)):
        print(index[i])
        path = Alg + "/collusion_detect_36_" + str(index[i]) + "%.csv"

        Accuracy = list(pd.read_csv(path)["Acc"])
        Precision = list(pd.read_csv(path)["Pre"])
        Recall = list(pd.read_csv(path)["Rec"])
        F1 = list(pd.read_csv(path)["f1_score"])

        Mean_acc.append(round(sum(Accuracy) / 30, 3))
        Var_acc.append(round(np.std(Accuracy), 3))

        Mean_pre.append(round(sum(Precision) / 30, 3))
        Var_pre.append(round(np.std(Precision), 3))

        Mean_rec.append(round(sum(Recall) / 30, 3))
        Var_rec.append(round(np.std(Recall), 3))

        Mean_f1.append(round(sum(F1) / 30, 3))
        Var_f1.append(round(np.std(F1), 3))

    Mean_acc = DataFrame({"Mean_acc": Mean_acc})
    Var_acc = DataFrame({"Var_acc": Var_acc})
    Mean_pre = DataFrame({"Mean_pre": Mean_pre})
    Var_pre = DataFrame({"Var_pre": Var_pre})
    Mean_rec = DataFrame({"Mean_rec": Mean_rec})
    Var_rec = DataFrame({"Var_rec": Var_rec})
    Mean_f1 = DataFrame({"Mean_f1": Mean_f1})
    Var_f1 = DataFrame({"Var_f1": Var_f1})

    write = concat([Mean_acc, Var_acc, Mean_pre, Var_pre, Mean_rec, Var_rec, Mean_f1, Var_f1], axis=1)
    print(write)
    write.to_csv(Alg + "/Mean_Var.csv")


def Plt(Alg):
    path = "KCDN/Mean_Var.csv"
    KCDN_mean = list(pd.read_csv(path)["Mean_" + Alg])

    path = "RE/Mean_Var.csv"
    RE_mean = list(pd.read_csv(path)["Mean_" + Alg])

    "FC"
    path = "FC/Mean_Var.csv"
    FC_mean = list(pd.read_csv(path)["Mean_" + Alg])

    "CP"
    path = "CP/Mean_Var.csv"
    CP_mean = list(pd.read_csv(path)["Mean_" + Alg])

    "KCD(废)"
    path = "KCD/Mean_Var.csv"
    KCD_mean = list(pd.read_csv(path)["Mean_" + Alg])

    x_ticks = ['0.1', '0.2', '0.3', '0.4', '0.5']
    x = range(len(x_ticks))

    plt.plot(x, KCDN_mean,
             linewidth=1.5, marker='v',
             ms=7,
             label=u'KCDN_W')

    plt.plot(x, RE_mean,
             linewidth=1.5, marker='o',
             ms=7,
             label=u'RE')

    plt.plot(x, FC_mean,
             linewidth=1.5, marker='o',
             ms=7,
             label=u'FC')

    plt.plot(x, CP_mean,
             linewidth=1.5, marker='o',
             ms=7,
             label=u'CP')

    plt.plot(x, KCD_mean,
             linewidth=1.5, marker='o',
             ms=7,
             label=u'KCD')

    plt.legend()  # 让图例生效
    plt.xticks(x, x_ticks, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y')

    plt.xlim(-0.2, 4.2)
    plt.ylim(0, 1)

    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"Collusion Percentage", size=18)  # X轴标签
    plt.ylabel(Alg, size=18)  # Y轴标签

    plt.tick_params(direction='in')
    plt.tick_params(top=True, bottom=True, left=True, right=True)

    jpg_path = "../collusion_P_0.8/s4-dog" + Alg + ".jpg"

    plt.gcf().savefig(jpg_path, dpi=1800, format='jpg')

    plt.show()


#

if __name__ == '__main__':
    """求平均值与标准型差"""
    Mean_Var("KCDN_W")
    Mean_Var("FC")
    Mean_Var("CP")
    Mean_Var("KCD")

