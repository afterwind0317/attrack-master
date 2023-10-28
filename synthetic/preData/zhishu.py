

from cmath import log
from math import e

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, concat, np


def Plt(lamb):


    data = pd.read_csv("createData.csv")  # 真实数据
    workers = sorted(data["worker"].drop_duplicates().values.tolist())  # 工人排名
    # print(workers)
    print(workers)

    numbers = [0] * 109  # 工人做题数
    for i in range(len(data)):
        numbers[data.iloc[i][5] - 1] = numbers[data.iloc[i][5] - 1] + 1


    x = workers
    y = sorted(numbers, reverse=True)  # 工人做题数排名

    print(y)

    plt.plot(x, y)



    number_tasks = []

    for i in range(len(x)):
        num = lamb * pow(e, -(lamb) * x[i]) * (345/0.04259058073558091)
        number_tasks.append(num)

    print(number_tasks)

    plt.plot(workers, number_tasks)



    plt.show()




if __name__ == '__main__':

        # s = [0.00001,0.0001,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01, 0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4]
        #
        # for i in range(len(s)):
        #     print("s=", s[i], "--- std=", Plt(s[i]))

        lamb = 0.04453
        Plt(lamb)

