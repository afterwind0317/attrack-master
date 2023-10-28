import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas import concat
from pandas.core.frame import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

def create_data(path_answer, path_result, save_path):

    data = pd.read_csv(path_answer)
    gt = pd.read_csv(path_result)

    list_gt = []
    list_day = []
    list_hour = []
    list_minute = []
    list_second = []

    for i in range(len(data)):
        print(i)
        for j in range(len(gt)):
            if data.iloc[i][0] == gt.iloc[j][0]:
                list_gt.append(gt.iloc[j][1])
                list_day.append(0)
                list_hour.append(0)
                list_minute.append(0)
                list_second.append(0)

    gt = DataFrame({"gt": list_gt})
    day = DataFrame({"day": list_day})
    hour = DataFrame({"hour": list_hour})
    minute = DataFrame({"minute": list_minute})
    second = DataFrame({"second": list_second})

    print(gt)

    data = pd.concat([day, hour, minute, second, data, gt], axis=1)

    # print(data)
    #
    for i in range(30):
        data = shuffle(data)
        data.to_csv(save_path, sep=',', index=0)





if __name__ == '__main__':

    path_answer = "../data/demo_answer_file.csv"
    path_result = "../data/demo_result_file.csv"
    save_path = "../data/createData.csv"


    """
    加入时间戳，加入gt,（随机打乱即可），打乱30次，生成数据集"""
    create_data(path_answer, path_result, save_path)

