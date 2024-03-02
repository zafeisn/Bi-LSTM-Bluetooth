# -*- coding: utf-8 -*-
# Time： 2022/1/4 11:05
# Author: LinJie
# Description: 
# @Project : MyCode
# @FileName: supervised.py

import time
import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelSpreading

# 指纹数据 杂质数据 全部数据
def predict(data_1, data_2, data_3):

    # 获取长度
    l_data_1 = len(data_1)
    l_data_2 = len(data_2)
    l_data_3 = len(data_3)
    # 读取特征
    X_1 = np.array(data_1.iloc[:, 3:11])
    X_2 = np.array(data_2.iloc[:, 3:11])
    X_3 = np.array(data_3.iloc[:, 3:11])
    # 读取标签
    Y_1 = np.array(data_1['TRUE'], dtype=int)
    Y_2 = np.array(data_2['TRUE'], dtype=int)
    Y_3 = np.array(data_3['TRUE'], dtype=int)
    # Y_3 = Y_3[l_data_1:]

    # 垂直拼接X
    X = np.concatenate((X_1, X_2), axis=0)
    X = np.concatenate((X, X_3), axis=0)
    # 垂直拼接Y
    Y = np.concatenate((Y_1, Y_2), axis=0)
    Y = np.concatenate((Y, Y_3), axis=0)
    Y_train = Y.copy()
    # 进行设置处理
    counts = l_data_1 + l_data_2
    Y_train[counts:] = -1
    Y_train = np.asarray(Y_train, dtype=int)

    cls = LabelSpreading(max_iter=100, kernel='rbf', gamma=0.1)
    cls.fit(X, Y_train)
    return cls.transduction_

# 基本指纹库数据
finger_print_data = pd.read_csv("D:/file/Final/DataProcess/2022-01-07/data/fingerPrint.csv")
finger_print_data = finger_print_data.iloc[:, :32]
# 掺杂数据
impurity_data = pd.read_csv("D:/file/Final/DataProcess/2022-01-07/data/impurityData.csv")
# 全部数据
all_data = pd.read_csv("D:/file/Final/DataProcess/2022-01-07/data/allData.csv")

# 拼接
data = np.concatenate((finger_print_data, impurity_data), axis=0)
data = np.concatenate((data, all_data), axis=0)
print(len(data))

start = time.time()
# 获得半监督结果
pre = predict(finger_print_data, impurity_data, all_data)
end = time.time()
print("程序运行时间为：", end - start)
labels = all_data.columns
data = pd.DataFrame(data,columns=labels)
data['pre'] = pre

print("请稍候，正在开始写入...")
data.to_csv("D:/file/Final/DataProcess/2022-04-25/FinalData.csv")
print("恭喜你，数据已成功保存到FinalData.csv中")

data = data[data['pre'] != 307]
data = data[data['TRUE'] == data['pre']]
print("请稍候，正在开始写入...")
data.to_csv("D:/file/Final/DataProcess/2022-04-25/FingerData.csv")
print("恭喜你，数据已成功保存到FingerData.csv中")