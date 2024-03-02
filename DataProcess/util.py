# -*- coding: utf-8 -*-
# Time： 2022/1/4 10:09
# Author: LinJie
# Description: 完成总和相加、大小排序
# @Project : MyCode
# @FileName: util.py

# 计算总和
def cs_sum(KMeans):
    # 记录每个聚类中心的和
    c_sums = []
    # 获取聚类中心
    c_centers = KMeans.cluster_centers_
    counts = len(c_centers)
    for i in range(counts):
        temp = 0
        for num in c_centers[i]:
            temp += abs(num)
        c_sums.append(temp)
    return c_sums, counts

# 排序
def cs_sort(cluster, KMeans):
    c_sums, counts = cs_sum(KMeans)
    temp = c_sums.copy()
    # 完成排序
    temp.sort()
    # 记录原来的索引位置
    o_labels = []
    for i in range(counts):
        o_labels.append(c_sums.index(temp[i]))
    # 开始排序
    for j in range(len(cluster)):
        # 交换标签位置
        o_label = cluster[j]
        cluster[j] = str(o_labels.index(o_label))
    return cluster

# 计算总
def sum_time(times):
    time = 0
    for t in range(len(times)):
        time += t
    return time