# -*- coding: utf-8 -*-
# Time： 2023/2/22 12:20
# Author: LinJie
# Description: 
# @Project : MyCode
# @FileName: ElowTest.py


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance

data = pd.read_csv("D:/file/Final/Data/2022-04-22.csv")
X = data.iloc[:, 4:12]
print(X[0:35])
X = X[172:218]
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)

plt.figure(figsize=(6,4))
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,8))
visualizer.ax.set_xlabel('K值')
visualizer.ax.set_ylabel('分数')

visualizer.fit(x_scaled)
visualizer.show()

