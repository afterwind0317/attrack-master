import math

import pandas as pd
from pandas import DataFrame, concat

alpha = 1
beta = 1
gamma = 16

def delete(collusion_P, percentage, epoch):

    path = "../../../baseline/CP/collusion_data/collusion_" + str(collusion_P) + "/WPCR/wpcr_" + str(percentage) + "_" + str(epoch) + ".csv"
    data = pd.read_csv(path)

    collusion_worker = []

    collusion_data = data.iloc[len(data) - int(percentage/3):len(data), :]
    for i in range(len(collusion_data)):
        collusion_worker.append(int(collusion_data.iloc[i][0]))
    collusion_worker = sorted(list(set(collusion_worker)))
    print("预测串谋工人：", collusion_worker)



    """delete workers's tasks """
    data_path = "../../result/collusion_data/collusion_" + str(collusion_P) + "/experience_" + str(percentage) + "/data" + str(
        epoch) + "_collaborate_D.csv"
    data = pd.read_csv(data_path)
    print(data)

    for i in range(len(collusion_worker)):
        tmp_worker = collusion_worker[i]

        tmp_tasks = data[(data.worker == tmp_worker)]

        tmp_index = tmp_tasks.index.values
        data = data.drop(index=tmp_index)
    #
    print(data)
    save_path = "../../result/collusion_data/collusion_" + str(collusion_P) + "/experience_" + str(
        percentage) + "/data" + str(
        epoch) + "_collaborate_D_CP.csv"
    data.to_csv(save_path, sep=',', index=0)





def display_delete(collusion_P, percentage, epoch):

    # 每一种串谋占比
    for per in range(len(percentage)):

        # 30次实验
        for epo in range(epoch, epoch + 30):

            delete(collusion_P, percentage[per], epo)





if __name__ == '__main__':

    collusion_P = 0.8
    percentage = [36]
    # percentage = [12]
    """
    找出的串谋工作者，百分之30%
    """

    display_delete(collusion_P, percentage, 0)
