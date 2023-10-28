import math

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
"""
统计实验结果
"""
def AVG(window, thresold, x):

    if window == 1:
        acc = []
        pre = []
        rec = []
        for i in range(len(x)):
            data_acc = np.array(pd.read_csv("../result/threshold_"+str(thresold)+"/experience"+str(window)+"_result_"+str(x[i])+".csv", usecols=np.arange(0, 1)))
            acc.append(data_acc.sum() / 30)

            data_pre = np.array(pd.read_csv("../result/threshold_"+str(thresold)+"/experience"+str(window)+"_result_"+str(x[i]) +".csv",usecols=np.arange(1, 2)))
            pre.append(data_pre.sum() / 30)

            data_rec = np.array(pd.read_csv("../result/threshold_"+str(thresold)+"/experience"+str(window)+"_result_"+str(x[i]) +".csv",usecols=np.arange(2, 3)))
            rec.append(data_rec.sum() / 30)

        return (acc, pre, rec)

    if window == 2:
        acc = []
        pre = []
        rec = []
        for i in range(len(x)):
            data_acc = np.array(pd.read_csv(
                "../result/threshold_"+str(thresold)+"/experience"+str(window)+"_result_" + str(x[i]) + ".csv",
                usecols=np.arange(0, 1)))
            acc.append(data_acc.sum() / 30)

            data_pre = np.array(pd.read_csv(
                "../result/threshold_"+str(thresold)+"/experience"+str(window)+"_result_" + str(x[i]) + ".csv",
                usecols=np.arange(1, 2)))
            pre.append(data_pre.sum() / 30)

            data_rec = np.array(pd.read_csv(
                "../result/threshold_"+str(thresold)+"/experience"+str(window)+"_result_" + str(x[i]) + ".csv",
                usecols=np.arange(2, 3)))
            rec.append(data_rec.sum() / 30)

        return (acc, pre, rec)


def display(x, thresold, window):


    thresold = thresold

    window = window


    (Acc, Pre, Rec) = AVG(window, thresold, x)
    print("30 次实验窗口"+str(window)+"均值")
    print("Acc=", Acc)
    print("Pre=", Pre)
    print("Rec=", Rec)
    Acc = DataFrame({"acc": Acc})
    Pre = DataFrame({"pre": Pre})
    Rec = DataFrame({"rec": Rec})
    write = concat([Acc, Pre, Rec], axis=1)
    write.to_csv("../result/threshold_"+str(thresold)+"/AVG_acc_pre_rec_"+str(window)+".csv", sep=',', index=0)





if __name__ == '__main__':

    x = [11, 21, 32, 43, 55]



    """统计实验结果"""
    display(x=x, thresold=0.001, window=2)








