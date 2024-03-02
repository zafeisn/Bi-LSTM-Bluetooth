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

def select(file, file_folder, file_name, row, offset):
    data = pd.read_csv(file)
    data_labels = data.columns
    labels = data['TRUE'].unique()

# -----------------------------------  创建文件保存路径（需要根据文件时间来创建文件夹） --------------------------------- #

    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
        print(file_folder + " 创建成功")
    file_path = file_folder + file_name


    # 新创建文件
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = list(data_labels)
            # 写入头
            writer.writerow(headers)

    write_to_csv(labels,data,file_path, row, offset)

def write_to_csv(labels, data, file_path, row, offset):
    for xy in labels:
        temp = data[data['TRUE'] == xy]
        row_max = temp.describe()[row].loc['max']
        row_max_ = row_max + offset
        temp = temp[temp[row] > row_max_]
        temp.to_csv(file_path, mode='a', header=False, index=False)
    print("恭喜你，文件已成功完成写入")

