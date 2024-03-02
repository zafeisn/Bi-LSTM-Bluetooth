# -*- coding: utf-8 -*-
# Time： 2022/2/23 12:27
# Author: LinJie
# Description: 
# @Project : MyCode
# @FileName: cal_error.py

from math import sqrt
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# 误差评估
def cal_error(y_test, pred_y):
    # 1、平均绝对误差
    print("------------------------- MAE ---------------------------")
    print("x = ", mean_absolute_error(y_test[:, 0], pred_y[:, 0]))
    print("y = ", mean_absolute_error(y_test[:, 1], pred_y[:, 1]))

    # 2、均方误差
    print("------------------------- MSE ---------------------------")
    print("x = ", mean_squared_error(y_test[:, 0], pred_y[:, 0]))
    print("y = ", mean_squared_error(y_test[:, 1], pred_y[:, 1]))

    # 3、均方根误差
    print("------------------------- RMSE ---------------------------")
    print("x = ", sqrt(mean_squared_error(y_test[:, 0], pred_y[:, 0])))
    print("y = ", sqrt(mean_squared_error(y_test[:, 1], pred_y[:, 1])))

    # 4、最大误差
    print("------------------------- ME ---------------------------")
    print("x = ", max_error(y_test[:, 0], pred_y[:, 0]))
    print("y = ", max_error(y_test[:, 1], pred_y[:, 1]))