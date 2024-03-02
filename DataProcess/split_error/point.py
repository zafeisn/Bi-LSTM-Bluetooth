# -*- coding: utf-8 -*-
# Time： 2022/5/15 17:23
# Author: LinJie
# Description: 
# @Project : MyCode
# @FileName: point.py

import pandas as pd
import numpy as np

# proportion = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 50, 100, 200, 300, 400, 500, 600]

proportion = [30,50,70,90,110,130,150,170,190,210,230,250,270,290,310,330,350,370,390,500,600]

def cp(error,p):
    ans = []
    num = []
    for item in p:
        n = ((error <= item) == True)
        num.append(n.sum())
        ans.append(n.sum()/len(error))
    return ans,num

x = pd.read_csv("D:/file/Final/DataProcess/2022-05-16/error/SVM/x.csv")

y = pd.read_csv("D:/file/Final/DataProcess/2022-05-16/error/SVM/y.csv")

point = pd.read_csv("D:/file/Final/DataProcess/2022-05-16/error/SVM/point.csv")

ans_x,num_x = cp(x,proportion)
ans_y,num_y = cp(y,proportion)
ans_point,num_point = cp(point,proportion)

data = pd.DataFrame(proportion, columns=['proportion'])
data['num_x'] = np.array(num_x)
data['ans_x'] = np.array(ans_x)
data['num_y'] = np.array(num_y)
data['ans_y'] = np.array(ans_y)
data['num_point'] = np.array(num_point)
data['ans_point'] = np.array(ans_point)

data.to_csv("D:/file/Final/DataProcess/2022-05-16/error/SVM/2/cp.csv",index=False)

"30","50","70","90","110","130","150","170","190","210","230","250","270","290","310","330","350","370","390","500","600"