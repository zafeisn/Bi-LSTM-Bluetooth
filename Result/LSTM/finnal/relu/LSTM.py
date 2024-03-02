# -*- coding: utf-8 -*-
# Time： 2022/1/4 11:05
# Author: LinJie
# Description: 
# @Project : MyCode
# @FileName: LSTM.py

import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os

# ================================================================ 读取数据集
train = pd.read_csv("D:/file/Final/DataProcess/2022-04-25/train.csv")
test = pd.read_csv("D:/file/Final/DataProcess/2022-04-25/test.csv")
# 验证集
val = pd.read_csv("D:/file/Final/DataProcess/2022-04-25/val.csv")

# ================================================================ 数值处理

def data_process(data):

    data.iloc[:, 3:11] = np.where(data.iloc[:, 3:11] <= 0,
                            (data.iloc[:, 3:11] + 110) / 110,
                            -(data.iloc[:, 3:11] - 110)/ 110)
data_process(train)
data_process(test)
data_process(val)
# ================================================================ 数据处理
from scipy.sparse import lil_matrix
from sklearn.preprocessing import StandardScaler

def lable_to_rssi(data):
    # 输入RSSI
    in_data = data.copy()
    in_data = in_data.iloc[:, 3:11]
    in_data = lil_matrix(in_data).toarray()
    # XYZ坐标
    out_y = data.copy()
    out_y = out_y.iloc[:, 0:2]
    out_y = lil_matrix(out_y).toarray()
    return in_data,out_y

x_train, y_train = lable_to_rssi(train)
x_test, y_test = lable_to_rssi(test)

x_val, y_val = lable_to_rssi(val)
# ================================================================ 划分数据集
# RSSI扩展维度（268,10）→（268,1,10）
x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)
x_val = np.expand_dims(x_val, axis=1)
# ================================================================ 回归模型
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization,InputLayer,LSTM,Bidirectional,Dropout
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras import regularizers

seed = 7
np.random.seed(seed)
model = Sequential()
model.add(InputLayer(input_shape=(1, 8)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True)))
# model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=True)))
# model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True)))
# model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=True)))
model.add(Bidirectional(LSTM(100, activation='relu')))
# model.add(Bidirectional(LSTM(100, activation='tanh')))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(2))

# ================================================================ 保存模型结构
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

from keras.callbacks import EarlyStopping
monitor = EarlyStopping(monitor='val_mse', min_delta=1e-3, patience=50, verbose=1, mode='auto')

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=800, batch_size=64, verbose=1, validation_data=(x_test, y_test), shuffle=True)

score = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
print(score)  # 输出预测的loss和mse
pred_y = model.predict(x_val)  # 预测

# ================================================================ 模型保存
tf.saved_model.save(model, "xy_model" + "\\" + "1")

# ================================================================ 拟合曲线
import matplotlib.pyplot as plt
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'],'--')
# plt.title('Model MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig("拟合曲线.png")
plt.show()

# ================================================================ 误差评估
from math import sqrt
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# 1、平均绝对误差
print("------------------------- MAE ---------------------------")
print("x = ", mean_absolute_error(y_val[:,0],pred_y[:,0]))
print("y = ", mean_absolute_error(y_val[:,1],pred_y[:,1]))

# 2、均方误差
print("------------------------- MSE ---------------------------")
print("x = ", mean_squared_error(y_val[:,0],pred_y[:,0]))
print("y = ", mean_squared_error(y_val[:,1],pred_y[:,1]))

# 3、均方根误差
print("------------------------- RMSE ---------------------------")
print("x = ", sqrt(mean_squared_error(y_val[:,0],pred_y[:,0])))
print("y = ", sqrt(mean_squared_error(y_val[:,1],pred_y[:,1])))

# 4、最大误差
print("------------------------- ME ---------------------------")
print("x = ", max_error(y_val[:,0],pred_y[:,0]))
print("y = ", max_error(y_val[:,1],pred_y[:,1]))