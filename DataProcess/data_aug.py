# -*- coding: utf-8 -*-
# Time： 2022/3/20 14:38
# Author: LinJie
# Description: 
# @Project : MyCode
# @FileName: data_aug.py

from df_data_aug import read_file, write_to_csv

# file_path = "D:/file/Final/DataProcess/2022-01-07/test1.csv"
# file_folder = "D:/file/Final/DataProcess/2022-01-07"
# file_name = "/data_aug2.csv"

# file_path = "D:/file/Final/DataProcess/2022-01-07/test/test.csv"
# file_folder = "D:/file/Final/DataProcess/2022-01-07/test"
# file_name = "/test_aug.csv"

file_path = "D:/file/Final/DataProcess/2022-01-07/train/train.csv"
file_folder = "D:/file/Final/DataProcess/2022-01-07/train"
file_name = "/train_aug.csv"
read_file(file_path, file_folder, file_name)
