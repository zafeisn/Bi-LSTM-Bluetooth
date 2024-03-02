# -*- coding: utf-8 -*-
# Time： 2022/1/4 11:05
# Author: LinJie
# Description: 需要将输入紧凑
# @Project : MyCode
# @FileName: CNN.py

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import lil_matrix

base_path = "D:/file/Final/DataProcess"
time_path = "2022-01-07"
file_name = "BaseFingerPrintDatas.csv"
file_path = base_path + "//" + time_path + "//" + file_name

data = pd.read_csv(file_path)

def d_process(data):
    return (data + 110) / 110

# 构造输入
features = data.iloc[:, 3:11]
labels = data.iloc[:, :2]
features = np.asarray(features)
labels = np.asarray(labels)

img_x = np.zeros(shape=(4230, 7, 7, 1))
beacon_coords = {
    "N1" : (2, 3),
    "N2" : (2, 4),
    "N3" : (2, 5),
    "N4" : (4, 6),
    "N5" : (6, 5),
    "N6" : (6, 4),
    "N7" : (6, 3),
    "N8" : (4, 2)
}
for key,value in beacon_coords.items():
    img_x[:, value[0], value[1], 0] = d_process(data[key])

# ================================================================ 划分数据集
from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(img_x, labels, test_size = .2)

import time
from tensorflow.keras.layers import Conv2D,LeakyReLU,Input,Dropout,Flatten,BatchNormalization,Bidirectional,LSTM,InputLayer
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

start_time = time.time()
def CNN_Model():
    model = tf.keras.Sequential()
    model.add(Input(shape=(7,7,1)))
    model.add(Conv2D(128, kernel_size=(5,5), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(64, kernel_size=(5,5), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(32, kernel_size=(3,3), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(16, kernel_size=(3,3), padding='same'))
    model.add(LeakyReLU())

    # 新加
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(128, kernel_size=(5, 5), padding='same'))
    model.add(LeakyReLU())

    model.add(Dense(32))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(2))
    model.add(LeakyReLU())

    return model

model = CNN_Model()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mean_absolute_error', metrics='mse')

history = model.fit(train_x, train_y, epochs=800, batch_size=128, verbose=1, validation_data=(val_x, val_y))

plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('Model MSE')
plt.ylabel('Mse')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
pred = model.predict(val_x)  # 预测

# ================================================================ 误差评估
from math import sqrt
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

pred = np.asarray(pred)
val_y = np.asarray(val_y)

# 1、平均绝对误差
print("------------------------- MAE ---------------------------")
print("x = ", mean_absolute_error(val_y[:,0],pred[:,0]))
print("y = ", mean_absolute_error(val_y[:,1],pred[:,1]))

# 2、均方误差
print("------------------------- MSE ---------------------------")
print("x = ", mean_squared_error(val_y[:,0],pred[:,0]))
print("y = ", mean_squared_error(val_y[:,1],pred[:,1]))

# 3、均方根误差
print("------------------------- RMSE ---------------------------")
print("x = ", sqrt(mean_squared_error(val_y[:,0],pred[:,0])))
print("y = ", sqrt(mean_squared_error(val_y[:,1],pred[:,1])))

# 4、最大误差
print("------------------------- ME ---------------------------")
print("x = ", max_error(val_y[:,0],pred[:,0]))
print("y = ", max_error(val_y[:,1],pred[:,1]))