# -*- coding: utf-8 -*-
# Time： 2021/12/31 14:22
# Author: LinJie
# Description: 1、对小数据训练集进行以K=2的聚类算法得到基本数据库
#              2、完成聚类结果后，进行数据存储
#              3、需要注意的地方：运行的文件夹的时间设置，否则会造成数据的重写
#              4、记录时间。手动建表进行记录
# @Project : FinalCode
# @FileName: basedata.py

import os
import csv
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import cs_sort
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置

# 读取csv
data = pd.read_csv("D:/file/Final/Data/2022-01-07-Train.csv")
# 对坐标标签进行拼接
data['true'] = data['X'].map(str) + data['Y'].map(str)
data['true'] = pd.to_numeric(data['true'])

# 数据全部标签
data_labels = data.columns[1:]
# 坐标属性标签
name_labels = data_labels[:3]
# 特征信息标签
feature_labels = data_labels[3:]
# 信号强度标签
rssi_labels = data_labels[:23]
# 参考点标签
rf_labels = data['true'].unique()

def fs_cluster(rf_labels, rssi_labels, data, file_path):
    # 先新建文件，若不存在进行创建
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = list(data_labels)
            headers.append('t_cluster')
            headers.append('t_sort')
            # 写入头
            writer.writerow(headers)
    # 记录参考点个数
    rf_counts = len(rf_labels)
    # 开始运行时间
    start_time = time.time()
    # 遍历数据集
    for i in range(rf_counts):
        # 获取每个参考点对应的数据（rf_data：用于最后写入使用；c_data：用于聚类）
        rf_data = data[data['true'] == rf_labels[i]][data_labels]
        c_data = rf_data[rssi_labels]
        # 初始化聚类个数，因为第一步只需要取K=2即可，为减少计算复杂度，这里只写到5
        n_cluster = range(1, 5)
        # 得到聚类信息
        kmeans = [KMeans(n_clusters=j).fit(c_data) for j in n_cluster]
        # 获取每个聚类的分数（目前这个分数没什么用）
        scores = [kmeans[k].score(c_data) for k in range(len(kmeans))]
        # 进行预测
        t_cluster = kmeans[2].predict(c_data)
        # 原始聚类结果设置
        rf_data['t_cluster'] = t_cluster
        # 进行聚类排序
        t_sort = cs_sort(t_cluster, kmeans[2])
        # 排序后聚类结果设置
        rf_data['t_sort'] = t_sort
        # 保存到文件中
        rf_data.to_csv(file_path, mode='a', header=False, index=False)
    # 结束运行时间
    end_time = time.time()
    print("恭喜你，文件已成功完成写入")
    print("请稍等，正在计算本次运行时间...")
    times = end_time - start_time
    print("完成本次运行总时间：" + str(times) + "s")


# -----------------------------------  创建文件保存路径（需要根据文件时间来创建文件夹） --------------------------------- #

file_folder = "D:/file/Final/DataProcess/2022-01-07"
if not os.path.exists(file_folder):
    os.makedirs(file_folder)
    print(file_folder + " 创建成功")
file_path = file_folder + "/BaseData.csv"

# 开始聚类
fs_cluster(rf_labels, rssi_labels, data, file_path)

