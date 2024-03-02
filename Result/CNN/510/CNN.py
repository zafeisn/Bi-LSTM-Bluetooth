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

# ================================================================ 读取数据集
train = pd.read_csv("D:/file/Final/DataProcess/2022-04-25/train.csv")
test = pd.read_csv("D:/file/Final/DataProcess/2022-04-25/test.csv")
# 验证集
val = pd.read_csv("D:/file/Final/DataProcess/2022-04-25/val.csv")

# ================================================================ 数值处理

def data_process(data):
    return (data + 110) / 110

# ================================================================ 数据处理

img_x_train = np.zeros(shape=(6195, 7, 7, 1))
img_x_test = np.zeros(shape=(1515, 7, 7, 1))
img_x_val = np.zeros(shape=(3567, 7, 7, 1))
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
    img_x_train[:, value[0], value[1], 0] = data_process(train[key])
    img_x_test[:, value[0], value[1], 0] = data_process(test[key])
    img_x_val[:, value[0], value[1], 0] = data_process(val[key])

y_train = np.asarray(train.iloc[:, :2])
y_test = np.asarray(test.iloc[:, :2])
y_val = np.asarray(val.iloc[:, :2])
# ================================================================ 划分数据集

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

# ================================================================ 保存模型结构
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mean_absolute_error', metrics='mse')

history = model.fit(img_x_train, y_train, epochs=200, batch_size=64, verbose=1, validation_data=(img_x_test, y_test))

# ================================================================ 模型保存
tf.saved_model.save(model, "xy_model" + "\\" + "1")

plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('Model MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
pred = model.predict(img_x_val)  # 预测

# ================================================================ 误差评估
from math import sqrt
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

pred = np.asarray(pred)
val_y = np.asarray(y_val)

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