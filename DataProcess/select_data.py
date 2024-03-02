# -*- coding: utf-8 -*-
# Time： 2022/2/22 15:03
# Author: LinJie
# Description: 
# @Project : MyCode
# @FileName: select_data.py

import numpy as np
import pandas as pd
import os
import csv
data = pd.read_csv("D:/file/Final/DataProcess/2022-01-07/BaseFingerPrintDatas.csv")
data_labels = data.columns
l_l = data['TRUE'].unique()
offset = -12

# -----------------------------------  创建文件保存路径（需要根据文件时间来创建文件夹） --------------------------------- #

file_folder = "D:/file/Final/DataProcess/2022-01-07"
if not os.path.exists(file_folder):
    os.makedirs(file_folder)
    print(file_folder + " 创建成功")
file_path = file_folder + "/select.csv"


# 新创建文件
if not os.path.exists(file_path):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = list(data_labels)
        # 写入头
        writer.writerow(headers)

for xy in l_l:
    temp = data[data['TRUE'] == xy]
    N6_min = temp.describe()['N6'].loc['max']
    N6_min_ = N6_min + offset
    temp = temp[temp['N6'] > N6_min_]
    temp.to_csv(file_path, mode='a', header=False, index=False)
print("恭喜你，文件已成功完成写入")

