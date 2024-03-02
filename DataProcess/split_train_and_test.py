# -*- coding: utf-8 -*-
# Time： 2022/4/25 14:50
# Author: LinJie
# Description: 
# @Project : MyCode
# @FileName: split_train_and_test.py
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

data = pd.read_csv("D:/file/Final/DataProcess/2022-01-07/data/FingerData.csv")
df = data.copy()

value_counts = data["TRUE"].value_counts()
value_counts = dict(value_counts)

labels = data["TRUE"].unique()
test = pd.DataFrame(columns=data.columns)

for k,v in value_counts.items():
    n = int(v * 0.2)
    test = test.append(data[data['TRUE'] == k].sample(n=n))

for i in test.index:
    if i in data.index:
        df = df.drop(index=i)

df.to_csv("D:/file/Final/DataProcess/2022-01-07/data/train.csv")
test.to_csv("D:/file/Final/DataProcess/2022-01-07/data/test.csv")