import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle


def Time_data(data_path):

    data = pd.read_csv(data_path)


    list_day = []
    list_hour = []
    list_minute = []
    list_second = []

    for i in range(len(data)):
        print(i)
        list_day.append(0)
        list_hour.append(0)
        list_minute.append(0)
        list_second.append(0)

    day = DataFrame({"day": list_day})
    hour = DataFrame({"hour": list_hour})
    minute = DataFrame({"minute": list_minute})
    second = DataFrame({"second": list_second})


    data = pd.concat([day, hour, minute, second, data], axis=1)


    for i in range(1):
        data = shuffle(data)
        path = "../data/TimeData.csv"
        data.to_csv(path, sep=',', index=0)





if __name__ == '__main__':

    # 加入时间戳 随机打乱
    data_path = "../data/sourceData.csv"

    Time_data(data_path)