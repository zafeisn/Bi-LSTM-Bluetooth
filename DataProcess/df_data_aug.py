# -*- coding: utf-8 -*-
# Time： 2022/3/20 14:29
# Author: LinJie
# Description: 数据增强--打乱数据
# @Project : MyCode
# @FileName: df_data_aug.py

from sklearn.utils import shuffle
import pandas as pd
import os
import csv

def read_file(file_path, file_folder, file_name):

    data = pd.read_csv(file_path)
    data_labels = data.columns

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

    write_to_csv(data, file_path)

def write_to_csv(data, file_path):
    for i in range(9):
        data.to_csv(file_path, mode='a', header=False, index=False)
        data = shuffle(data)
    print("写入完成")