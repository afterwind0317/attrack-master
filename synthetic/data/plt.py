import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("sourceData.csv")

print(data)

workers = sorted(data["worker"].drop_duplicates().values.tolist())

print(workers)

numbers = [0]*100

for i in range(len(data)):

    numbers[data.iloc[i][1]-1] = numbers[data.iloc[i][1]-1] + 1


print(numbers)


plt.plot(workers, numbers)
plt.show()
