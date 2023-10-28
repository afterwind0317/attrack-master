import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['STZhongsong']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False


def del_accuracy(data):
    tmp_data = []
    for j in range(30):
        tmp_data.append(data.iloc[0][j])
    tmp_avg = round(sum(tmp_data) / 30, 4)
    tmp_char = round(np.std(tmp_data, ddof=1), 4)
    return (tmp_avg, tmp_char)


if __name__ == '__main__':
    (avg_LACF_HDS, char_LACF_HDS) = del_accuracy(
        pd.read_csv("../data_result/LACF_HDS_acc_collusion_0.8-36.csv", header=None))
    (avg_CP_HDS, char_CP_HDS) = del_accuracy(pd.read_csv("../data_result/CP_HDS_acc_collusion_0.8-36.csv", header=None))
    (avg_FC_HDS, char_FC_HDS) = del_accuracy(pd.read_csv("../data_result/FC_HDS_acc_collusion_0.8-36.csv", header=None))
    (avg_KCD_HDS, char_KCD_HDS) = del_accuracy(
        pd.read_csv("../data_result/KCD_HDS_acc_collusion_0.8-36.csv", header=None))
    (avg_KCDN_HDS, char_KCDN_HDS) = del_accuracy(
        pd.read_csv("../data_result/KCDN_HDS_acc_collusion_0.8-36.csv", header=None))

    (avg_LACF_GLAD, char_LACF_GLAD) = del_accuracy(
        pd.read_csv("../data_result/LACF_GLAD_acc_collusion_0.8-36.csv", header=None))
    (avg_CP_GLAD, char_CP_GLAD) = del_accuracy(
        pd.read_csv("../data_result/CP_GLAD_acc_collusion_0.8-36.csv", header=None))
    (avg_FC_GLAD, char_FC_GLAD) = del_accuracy(
        pd.read_csv("../data_result/FC_GLAD_acc_collusion_0.8-36.csv", header=None))
    (avg_KCD_GLAD, char_KCD_GLAD) = del_accuracy(
        pd.read_csv("../data_result/KCD_GLAD_acc_collusion_0.8-36.csv", header=None))
    (avg_KCDN_GLAD, char_KCDN_GLAD) = del_accuracy(
        pd.read_csv("../data_result/KCDN_GLAD_acc_collusion_0.8-36.csv", header=None))

    # (avg_LACF_MV, char_LACF_MV) = del_accuracy(pd.read_csv("../data_result/LACF_MV_acc_collusion_0.8-36.csv", header=None))
    # (avg_CP_MV, char_CP_MV) = del_accuracy(pd.read_csv("../data_result/CP_MV_acc_collusion_0.8-36.csv", header=None))
    # (avg_FC_MV, char_FC_MV) = del_accuracy(pd.read_csv("../data_result/FC_MV_acc_collusion_0.8-36.csv", header=None))
    # (avg_KCD_MV, char_KCD_MV) = del_accuracy(pd.read_csv("../data_result/KCD_MV_acc_collusion_0.8-36.csv", header=None))
    # (avg_KCDN_MV, char_KCDN_MV) = del_accuracy(pd.read_csv("../data_result/KCDN_MV_acc_collusion_0.8-36.csv", header=None))

    print("LACF-HDS: ", avg_LACF_HDS, char_LACF_HDS)
    print("CP-HDS: ", avg_CP_HDS, char_CP_HDS)
    print("FC-HDS: ", avg_FC_HDS, char_FC_HDS)
    print("KCD-HDS: ", avg_KCD_HDS, char_KCD_HDS)
    print("KCDN_W-HDS: ", avg_KCDN_HDS, char_KCDN_HDS)

    print()
    print("LACF-GLAD: ", avg_LACF_GLAD, char_LACF_GLAD)
    print("CP-GLAD: ", avg_CP_GLAD, char_CP_GLAD)
    print("FC-GLAD: ", avg_FC_GLAD, char_FC_GLAD)
    print("KCD-GLAD: ", avg_KCD_GLAD, char_KCD_GLAD)
    print("KCDN_W-GLAD: ", avg_KCDN_GLAD, char_KCDN_GLAD)

    plt.subplot(1, 1, 1)

    x = np.array([1, 2])

    plt.ylim(0, 1.5)

    y_LACF = [avg_LACF_HDS, avg_LACF_GLAD]
    y_CP = [avg_CP_HDS, avg_CP_GLAD]
    y_FC = [avg_FC_HDS, avg_FC_GLAD]
    y_KCD = [avg_KCD_HDS, avg_KCD_GLAD]
    y_KCDN = [avg_KCDN_HDS, avg_KCDN_GLAD]


    plt.bar(x, y_LACF, width=0.01, align="center", label="LACF")
    plt.bar(x, y_CP, width=0.01, align="center", label="Filter by CP")
    plt.bar(x, y_FC, width=0.01, align="center", label="Filter by FC")
    plt.bar(x, y_KCD, width=0.01, align="center", label="Filter by KCD")
    plt.bar(x, y_KCDN, width=0.01, align="center", label="Filter by KCDN_W")




    plt.title("Filtered aggregation accuracy")

    for a, b in zip(x, y_LACF):
        plt.text(a, b, b, ha='center', va="bottom", fontsize=8)

    for a, b in zip(x + 0.05, y_CP):
        plt.text(a, b, b, ha='center', va="bottom", fontsize=8)

    for a, b in zip(x + 0.05, y_FC):
        plt.text(a, b, b, ha='center', va="bottom", fontsize=8)

    for a, b in zip(x + 0.05, y_KCD):
        plt.text(a, b, b, ha='center', va="bottom", fontsize=8)

    for a, b in zip(x + 0.05, y_KCDN):
        plt.text(a, b, b, ha='center', va="bottom", fontsize=8)

    plt.xlabel('Aggregation algorithm')

    plt.ylabel('Accuracy')

    plt.xticks(x + 0.15, ["HDS", "GLAD"])

    plt.legend()

    plt.show()
