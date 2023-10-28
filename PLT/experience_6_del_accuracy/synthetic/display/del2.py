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


    plt.figure(figsize=(6, 4.5))

    size = 3  # 2组
    x = np.arange(size)
    # x = ['MV', 'GLAD', 'HDS']

    y_LACF = [avg_LACF_HDS, avg_LACF_GLAD, avg_LACF_GLAD]
    y_CP = [avg_CP_HDS, avg_CP_GLAD, avg_CP_GLAD]
    y_FC = [avg_FC_HDS, avg_FC_GLAD, avg_CP_GLAD]
    y_KCD = [avg_KCD_HDS, avg_KCD_GLAD, avg_CP_GLAD]
    y_KCDN = [avg_KCDN_HDS, avg_KCDN_GLAD, avg_CP_GLAD]

    total_width, n = 0.8, 5  # 每一组5个

    width = total_width/n

    x = x - (total_width - width)/2

    plt.xticks([0, 1, 2], ['MV', 'GLAD', 'HDS'])
    plt.ylim(0, 1)

    plt.bar(x, y_LACF, width=width, label='LACF')
    plt.bar(x+width, y_CP, width=width, label='F-CP')
    plt.bar(x+width*2, y_FC, width=width, label='F-FC')
    plt.bar(x+width*3, y_KCD, width=width, label='F-KCD')
    plt.bar(x+width*4, y_KCDN, width=width, label='F-KCDN_W')

    plt.legend(bbox_to_anchor=(0.5, -0.18),loc=8,ncol=10) # , borderaxespad=0

    plt.show()
