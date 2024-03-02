# -*- coding: utf-8 -*-
# Time： 2021/12/31 14:27
# Author: LinJie
# Description: 1、完成基本数据集 -> 基本指纹库的建立
#              2、运行前需要注意：不要忘了创建文件夹的时间设置
#              3、因为数据是按照末尾添加的，所以多次运行的话数据量会越来越大并且重复
#              4、不足的地方：
#              4.1、由于使用手肘法，需要根据画图人工经验值判断聚类个数，存在偏差；
#              4.2、代码还未优化，由于是根据参考点个数去决定运行的次数，同时一旦运行了就需要完成所需聚类次数，完成的时间可能过长；
#              5、数据保存。手肘法画图进行本地存储；同时最后完成聚类结果，进行数据保存
#              6、记录时间。手动建表进行记录
# @Project : FinalCode
# @FileName: basefinger.py

import os
import csv
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.preprocessing import StandardScaler
from util import cs_sort, sum_time
from sklearn.cluster import KMeans
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置

data = pd.read_csv("D:/file/Final/DataProcess/2022-01-07/BaseData.csv")
data = data[data['t_sort'] != 2]

# 数据全部标签
data_labels = data.columns
# 坐标属性标签
name_labels = data_labels[:3]
# 特征信息标签
feature_labels = data_labels[3:]
# 信号强度标签
rssi_labels = data_labels[3:23]
# 参考点标签
rf_labels = data['true'].unique()
# 记录数据中最小个数
min_count = min(data['true'].value_counts())


def fs_cluster(rf_lables, rssi_labels, data, file_path, images_folder):
    # 新创建文件
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = list(data_labels)
            headers.append('f_cluster')
            #             headers.append('f_sort')
            # 写入头
            writer.writerow(headers)

    # 记录参考点个数
    rf_counts = len(rf_labels)
    # 记录总运行时间
    run_time = []
    # 遍历数据集
    for i in range(rf_counts):
        # 开始运行时间
        start_time = time.time()

        # 获取每个参考点对应的写入数据和特征数据
        rf_data = data[data['true'] == rf_labels[i]][data_labels]
        c_data = rf_data[rssi_labels]
        # 初始化聚类个数
        n_cluster = np.arange(1, 20)
        # 得到聚类信息
        kmeans = [KMeans(n_clusters=j).fit(c_data) for j in n_cluster]
        # 获取聚类分数
        scores = [kmeans[k].score(c_data) for k in range(len(kmeans))]
        # 开始使用手肘法画图
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(n_cluster.astype(np.str), scores, label=str(rf_labels[i]))
        # 设置横轴刻度步长
        # x = MultipleLocator(1)
        # ax.xaxis.set_major_locator(x)
        plt.xlabel("聚类集群数")
        plt.ylabel("聚类分数")
        plt.title("ELow - 曲线")
        plt.legend(loc='lower right')
        plt.plot(color='g')
        # 保存到本地
        plt.savefig(images_folder + "/" + str(rf_lables[i]) + ".png")
        # 结束运行时间
        end_time = time.time()
        times_one = end_time - start_time
        print("初始化运行时间：" + str(times_one) + "s")
        plt.show()

        # 根据图手动设置聚类个数
        cluster = input("请输入不超过 " + str(20) + " 个的聚类数：")
        print("请稍等，正在设置...")
        print("本次聚类已成功设置个数 = " + cluster)

        # 开始聚类，得到指纹库聚类结果
        f_cluster = kmeans[int(cluster)].predict(c_data)

        # 开始运行时间
        start_time = time.time()
        # 将指纹库聚类结果添加到原始数据中
        rf_data['f_cluster'] = f_cluster

        # 结束运行时间
        end_time = time.time()
        times_two = end_time - start_time
        print("完成聚类运行时间：" + str(times_two) + "s")
        times = times_one + times_two
        print("本次总运行时间：" + str(times) + "s")
        run_time.append(times)
        # 进行聚类排序
        f_sort = cs_sort(f_cluster, kmeans[int(cluster)])
        # 排序后聚类结果设置
        rf_data['f_sort'] = f_sort
        # 开始写入到文件中
        rf_data.to_csv(file_path, mode='a', header=False, index=False)
    print("恭喜你，文件已成功完成写入")
    print("请稍等，正在计算总运行时长...")
    runtime = sum_time(run_time)
    print("完成" + str(rf_counts) + "次的总运行时长：" + str(runtime) + "s")

# -----------------------------------  创建文件保存路径（需要根据文件时间来创建文件夹） --------------------------------- #

file_folder = "D:/file/Final/DataProcess/2022-03-22"
images_folder = file_folder + "/img"
if not os.path.exists(file_folder):
    os.makedirs(file_folder)
    print(file_folder + " 创建成功")
if not os.path.exists(images_folder):
    os.makedirs(images_folder)
    print(images_folder + " 创建成功")
file_path = file_folder + "/BaseFingerPrintData.csv"

# 开始聚类
fs_cluster(rf_labels, rssi_labels, data, file_path, images_folder)
