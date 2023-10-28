import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from scipy.stats import beta
import numpy as np

"""
********************************************************
"""


def get_resource_f(list_a, list_b, x):
    N = 109  # 工人个数
    sum = 0
    x = round(float(x), 2)
    for i in range(len(list_a)):
        a = list_a[i]  # 答对个数
        b = list_b[i]  # 答错个数
        a = float(a)
        b = float(b)
        xx = beta.pdf(x, a + 1, b + 1)
        sum = sum + xx
    return sum / N


def get_resource_alpha(data):
    normal_worker = data["worker"].drop_duplicates()
    normal_worker = sorted(normal_worker.values.tolist())
    normal_worker = DataFrame({"worker": normal_worker})
    Right = []
    for w in range(len(normal_worker)):
        right = 0
        data_normal_worker = data[(data.worker == normal_worker.iloc[w][0])]

        for i in range(len(data_normal_worker)):
            if data_normal_worker.iloc[i][6] == data_normal_worker.iloc[i][7]:
                right = right + 1
        Right.append(right)
    return Right


def get_resource_beta(data):
    normal_worker = data["worker"].drop_duplicates()
    normal_worker = sorted(normal_worker.values.tolist())
    normal_worker = DataFrame({"worker": normal_worker})
    Wrong = []
    for w in range(len(normal_worker)):
        wrong = 0
        data_normal_worker = data[(data.worker == normal_worker.iloc[w][0])]

        for i in range(len(data_normal_worker)):
            if data_normal_worker.iloc[i][6] != data_normal_worker.iloc[i][7]:
                wrong = wrong + 1
        Wrong.append(wrong)
    return Wrong


def get_resource_result():
    path = "createData.csv"
    data = pd.read_csv(path)
    Result = []

    list_alpha = get_resource_alpha(data)
    list_beta = get_resource_beta(data)

    x = np.arange(0, 1, 0.01)
    for i in range(len(x)):
        result = get_resource_f(list_alpha, list_beta, x[i])
        Result.append(result)
    return Result


"""
********************************************************
"""

"""
********************************************************
"""


def get_collusion_f(list_a, list_b, x):
    N = 109 * 30
    sum = 0
    x = round(float(x), 2)
    for i in range(len(list_a)):
        a = list_a[i]  # 答对个数
        b = list_b[i]  # 答错个数
        a = float(a)
        b = float(b)
        xx = beta.pdf(x, a + 1, b + 1)
        sum = sum + xx
    return sum / N


def get_collusion_alpha(collusion_P, percentage):
    Right = []
    for i in range(30):
        path = "collusion_" + str(collusion_P) + "/experience_" + str(percentage) + "/data" + str(
            i) + "_collaborate_D.csv"
        # print(path)
        data = pd.read_csv(path)
        worker = data["worker"].drop_duplicates()
        worker = sorted(worker.values.tolist())
        worker = DataFrame({"worker": worker})
        for w in range(len(worker)):
            data_normal_worker = data[(data.worker == worker.iloc[w][0])]
            right = 0
            for i in range(len(data_normal_worker)):
                if data_normal_worker.iloc[i][6] == data_normal_worker.iloc[i][7]:
                    right = right + 1
            Right.append(right)
    return Right


def get_collusion_beta(collusion_P, percentage):
    Wrong = []
    for i in range(30):

        path = "collusion_" + str(collusion_P) + "/experience_" + str(percentage) + "/data" + str(
            i) + "_collaborate_D.csv"
        # print(path)
        data = pd.read_csv(path)
        worker = data["worker"].drop_duplicates()
        worker = sorted(worker.values.tolist())
        worker = DataFrame({"worker": worker})
        for w in range(len(worker)):
            data_normal_worker = data[(data.worker == worker.iloc[w][0])]
            wrong = 0
            for i in range(len(data_normal_worker)):
                if data_normal_worker.iloc[i][6] != data_normal_worker.iloc[i][7]:
                    wrong = wrong + 1
            Wrong.append(wrong)
    return Wrong


def get_collusion_result(collusion_P, percentage):
    Result = []

    list_alpha = get_collusion_alpha(collusion_P, percentage)

    list_beta = get_collusion_beta(collusion_P, percentage)

    x = np.arange(0, 1, 0.01)
    for i in range(len(x)):
        result = get_collusion_f(list_alpha, list_beta, x[i])
        Result.append(result)

    return Result


"""
**************************************************
"""

if __name__ == '__main__':

    # 固定串谋概率为0.8，讨论串谋占比的影响 [11,21,32,43,55]
    collusion_P = 0.8

    # 获取原数据集的分布
    resource_result = []
    resource_result = get_resource_result()
    print(0)

    # 获取串谋数据及的分布情况
    collusion_result_6 = []
    collusion_result_6 = get_collusion_result(collusion_P, 11)
    print(11)

    # 获取串谋数据及的分布情况
    collusion_result_7 = []
    collusion_result_7 = get_collusion_result(collusion_P, 21)
    print(21)

    # 获取串谋数据及的分布情况
    collusion_result_8 = []
    collusion_result_8 = get_collusion_result(collusion_P, 32)
    print(32)

    # 获取串谋数据及的分布情况
    collusion_result_9 = []
    collusion_result_9 = get_collusion_result(collusion_P, 43)
    print(43)

    # 获取串谋数据及的分布情况
    collusion_result_10 = []
    collusion_result_10 = get_collusion_result(collusion_P, 55)
    print(55)

    x = np.arange(0, 1, 0.01)
    plt.plot(x, resource_result, color='red', label='origin', )
    plt.plot(x, collusion_result_6, color='blue', label='percentage=10%', )
    plt.plot(x, collusion_result_7, color='gray', label='percentage=20%', )
    plt.plot(x, collusion_result_8, color='yellow', label='percentage=30%', )
    plt.plot(x, collusion_result_9, color='black', label='percentage=40%', )
    plt.plot(x, collusion_result_10, color='c', label='percentage=50%', )

    plt.legend(loc='upper left')

    plt.tick_params(top=True, bottom=True, left=True, right=True)
    plt.tick_params(direction='in')

    plt.grid(axis='y', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)

    plt.subplots_adjust(bottom=0.15)

    plt.xlabel(u"Ability of worker(s4-dog)", size=14)  # X轴标签
    plt.ylabel("Density", size=14)  # Y轴标签

    path = 'Beta_s4-dog_percentage.jpg'
    plt.gcf().savefig(path, dpi=800, format='jpg')

    plt.show()
