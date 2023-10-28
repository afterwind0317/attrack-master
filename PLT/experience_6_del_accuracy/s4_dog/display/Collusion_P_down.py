import numpy as np
import pandas as pd
from pandas import DataFrame, concat
from pylab import *                                 #支持中文
mpl.rcParams['font.sans-serif'] = ['Arial']


def Synthetic(path, source_acc):
    Proportion = [0, 10, 20, 30, 40, 50]

    # 原数据集 聚合的准去率
    Accuracy = [source_acc]

    cha = [0]

    path = path
    data = pd.read_csv(path, header=None)
    for i in range(len(data)):
        tmp_data = []
        for j in range(len(data.iloc[i])):
            tmp_data.append(data.iloc[i][j])


        tmp_avg = round(sum(tmp_data) / 30, 4)
        tmp_cha = round(np.std(tmp_data, ddof=1), 4)

        print(tmp_avg, tmp_avg)
        Accuracy.append(tmp_avg)
        cha.append(tmp_cha)

    Proportion = DataFrame({"Proprotion": Proportion})
    Accuracy = DataFrame({"experience_3_collusion_accuracy": Accuracy})
    cha = DataFrame({"char": cha})

    write = concat([Proportion, Accuracy, cha], axis=1)
    write.to_csv("transform_"+path, sep=',', index=0)




# 大规模匿名框架
def plt_LAFC():

    names = ['0', '0.6', '0.7', '0.8', '0.9', '1.0']

    """MV"""
    x = range(len(names))
    mean1 = list(pd.read_csv("transform_MV_acc_collusion_all-32.csv")["experience_3_collusion_accuracy"])
    var1 = list(pd.read_csv("transform_MV_acc_collusion_all-32.csv")["char"])

    plt.errorbar(x, mean1, var1, color='red',
                 capsize=4, capthick=1,
                 linewidth=1.5, marker='o',
                 mec='red', mfc='red', ms=7,
                 label=u'MV')

    """GLAD"""
    mean2 = list(pd.read_csv("transform_GLAD_acc_collusion_all-32.csv")["experience_3_collusion_accuracy"])
    var2 = list(pd.read_csv("transform_GLAD_acc_collusion_all-32.csv")["char"])

    plt.errorbar(x, mean2, var2, color='blue',
                 capsize=4, capthick=1,
                 linewidth=1.5, marker='x',
                 mec='blue', mfc='blue', ms=7,
                 label=u'GLAD')

    """HDS"""
    mean2 = list(pd.read_csv("transform_HDS_acc_collusion_all-32.csv")["experience_3_collusion_accuracy"])
    var2 = list(pd.read_csv("transform_HDS_acc_collusion_all-32.csv")["char"])

    plt.errorbar(x, mean2, var2, color='green',
                 capsize=4, capthick=1,
                 linewidth=1.5, marker='s',
                 mec='green', mfc='green', ms=7,
                 label=u'HDS')


    # """K-MV"""
    # mean2 = list(pd.read_csv("MV/resultMV_acc_result_del_t7.csv")["experience_3_collusion_accuracy"])
    # var2 = list(pd.read_csv("MV/resultMV_acc_result_del_t7.csv")["char"])
    # plt.errorbar(x, mean2, var2, color='red',
    #              capsize=4, capthick=1,
    #              linewidth=1.5, marker='o',
    #              mec='red', mfc='red', ms=7,
    #              label=u'K-MV')
    #
    #
    # """K-GLAD"""
    # mean2 = list(pd.read_csv("GLAD/resultGLAD_acc_result_del_t7.csv")["experience_3_collusion_accuracy"])
    # var2 = list(pd.read_csv("GLAD/resultGLAD_acc_result_del_t7.csv")["char"])
    # plt.errorbar(x, mean2, var2, color='blue',
    #              capsize=4, capthick=1,
    #              linewidth=1.5, marker='x',
    #              mec='blue', mfc='blue', ms=7,
    #              label=u'K-GLAD')
    #
    #
    # """K-HDS"""
    # mean2 = list(pd.read_csv("HDS/resultHDS_acc_result_del_t7.csv")["experience_3_collusion_accuracy"])
    # var2 = list(pd.read_csv("HDS/resultHDS_acc_result_del_t7.csv")["char"])
    # plt.errorbar(x, mean2, var2, color='green',
    #              capsize=4, capthick=1,
    #              linewidth=1.5, marker='s',
    #              mec='green', mfc='green', ms=7,
    #              label=u'K-HDS')


    plt.legend()  # 让图例生效
    plt.xticks(x, names)
    # 新的实验.grid(axis='y')
    # encoding=utf-8

    plt.xlim(-0.2, 5.2)
    plt.ylim(0.7, 0.9)

    plt.grid(axis='y', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)

    plt.tick_params(top=True, bottom=True, left=True, right=True)
    plt.tick_params(direction='in')

    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"Collusion Opportunity", size=14)  # X轴标签
    plt.ylabel("Accuracy", size=14)  # Y轴标签
    path = 'result_collusion_30%_s4-dog.jpg'
    plt.gcf().savefig(path, dpi=1400, format='jpg')
    plt.show()




if __name__ == '__main__':
    """
    固定串谋机会为0.8， 讨论串谋占比的影响
    """
    # 串谋后准确率下降
    # 1-转化数据集的格式，求平均值与标准差  GLAD
    # path_GLAD = "GLAD_acc_collusion_all-32.csv"
    # Synthetic(path_GLAD, 0.835192)
    #
    # path_MV = "MV_acc_collusion_all-32.csv"
    # Synthetic(path_MV, 0.821561)
    #
    # path_HDS = "HDS_acc_collusion_all-32.csv"
    # Synthetic(path_HDS, 0.838910)


    # plt
    plt_LAFC()



