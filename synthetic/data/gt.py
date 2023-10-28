import numpy as np
import pandas as pd
from pandas import DataFrame, concat

data = pd.read_csv("TimeData.csv")

task = data["question"].drop_duplicates().sort_values()

task = task.values.tolist()
task = DataFrame({"question": task})

gt = []
for i in range(len(task)):
    tmp_tasks = data[(data.question==task.iloc[i][0])]

    gt.append(tmp_tasks.iloc[0][7])

gt = DataFrame({"gt": gt})

write = concat([task, gt], axis=1)
write.to_csv("result_file.csv", sep=',', index=0)
