import random
from math import e, log

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, concat, np
from scipy.stats import norm
from sklearn.preprocessing import Normalizer


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

# 生成工人
def createWorker(workerN):

    workers = []

    for i in range(workerN):
        workers.append(i+1)

    workers = DataFrame({"worker": workers})

    return workers

# 生成任务 与 gt
def createTask(taskN, classN):

    task_list = []
    gt = []
    for i in range(1, taskN+1):
        task_list.append(i)
        gt.append(i%classN+1)
    task = DataFrame({"task": task_list})
    gt = DataFrame({"gt": gt})

    data = concat([task, gt], axis=1)

    return data


def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x


def giveWorkers(worker, s):

    worker = worker["worker"].values.tolist()

    workers_P = []

    for i in range(len(worker)):

        tmp_P = 0.04453 * pow(e, -0.04453 * worker[i])

        # print(tmp_P)

        workers_P.append(tmp_P)


    Sum = sum(workers_P)

    for i in range(len(workers_P)):
        workers_P[i] = workers_P[i] / Sum

    # print(worker)
    # print(workers_P)


    p = np.array(workers_P)

    index = np.random.choice(worker, size=10, replace=False, p=p.ravel())

    # print(index)
    return index



def crowdsourcing(worker, data, rN, s):

    # 给工人能力
    ability = []
    for i in range(len(worker)):
        ability.append(round(random.uniform(0.5, 1), 2))

    ability = DataFrame({"ability": ability})
    # [worker, ability]
    workerInfo = concat([worker, ability], axis=1)

    Task = []
    Worker = []
    Annotation = []
    Gt = []
    Ability = []



    for i in range(len(data)):
        print("任务：", i)

        tmp_task = data.iloc[i][0]

        tmp_gt = data.iloc[i][1]

        # 给定工人
        worker_r = giveWorkers(worker, s)

        print("当前任务分配的工人有：", worker_r)

        for w in range(len(worker_r)):

            tmp_worker = worker_r[w]

            tmp_ability = workerInfo[(workerInfo.worker == tmp_worker)].iloc[0][1]
            tmp_hard = round(random.uniform(0, 1), 2)
            if tmp_ability >= tmp_hard:
                annotation = tmp_gt
            else:
                if tmp_gt == 1:
                    annotation = 2
                elif tmp_gt == 2:
                    annotation = 1


            Task.append(tmp_task)
            Gt.append(tmp_gt)
            Annotation.append(annotation)
            Worker.append(tmp_worker)
            Ability.append(tmp_ability)




    Task = DataFrame({"question": Task})
    Worker = DataFrame({"worker": Worker})
    Annotation = DataFrame({"answer": Annotation})
    Gt = DataFrame({"gt": Gt})
    Ability = DataFrame({"ability": Ability})

    data = concat([Task, Worker, Annotation, Gt],axis=1)
    return data



def createData(workerN, taskN, rN, classN, s):


    # 生成工人
    worker = createWorker(workerN)



    # 生成任务 与 gt
    data = createTask(taskN, classN)


    # 冗余答题
    sourceData = crowdsourcing(worker, data, rN, s)


    print(sourceData)


    save_path = "../data/sourceData_"+str(s)+".csv"
    sourceData.to_csv(save_path, sep=',', index=0)




# 画图
def Plt(s):

    path = "../data/sourceData_"+str(s)+".csv"
    data = pd.read_csv(path)

    print("s=", s)

    workers = sorted(data["worker"].drop_duplicates().values.tolist())

    print("工人id (工人排名)")
    print(workers)

    numbers = [0] * 120

    for i in range(len(data)):
        numbers[data.iloc[i][1] - 1] = numbers[data.iloc[i][1] - 1] + 1

    print(len(workers))
    print(len(numbers))


    print("工人答题数：")
    print(numbers)


    plt.title(str(s))

    plt.plot(workers, numbers)

    plt.show()

# 求c的方差
def display_Var(s, workN):

    path = "../data/sourceData_" + str(s) + ".csv"
    data = pd.read_csv(path)

    print("s=", s)
    workers = sorted(data["worker"].drop_duplicates().values.tolist())
    # print("工人id (工人排名)")
    # print(workers)

    numbers = [0] * workN
    for i in range(len(data)):
        numbers[data.iloc[i][1] - 1] = numbers[data.iloc[i][1] - 1] + 1
    # print("工人答题数：")
    # print(numbers)

    C = []

    for i in range(workN):
        print("工人：", i+1, "做了", numbers[i], "道题")
        c = pow(workers[i], s) * numbers[i]
        print("c=", numbers[i], "*", workers[i], "^", s, "=", c)
        C.append(c)
        print()


    var = np.var(C)

    print("Var=", var)
    print(var)




if __name__ == '__main__':

        workerN = 120
        taskN = 800
        rN = 10
        classN = 2
        s = 0.04453

        createData(workerN, taskN, rN, classN, s)

        Plt(s)
        # #
        # display_Var(s, workerN)




