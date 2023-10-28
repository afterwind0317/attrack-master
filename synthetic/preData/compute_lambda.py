import pandas as pd






data = pd.read_csv("createData.csv")  # 真实数据
workers = sorted(data["worker"].drop_duplicates().values.tolist())  # 工人排名
# print(workers)
numbers = [0] * 109  # 工人做题数
for i in range(len(data)):
    numbers[data.iloc[i][5] - 1] = numbers[data.iloc[i][5] - 1] + 1
K = workers
N = sorted(numbers, reverse=True)  # 工人做题数排名


sum_lambda = 0
for i in range(len(K)):
    sum_lambda = sum_lambda + ( N[i] * K[i] / sum(N))

lamb = 1 / sum_lambda

print(lamb)