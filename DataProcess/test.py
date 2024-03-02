# -*- coding: utf-8 -*-
# Time： 2022/3/20 14:02
# Author: LinJie
# Description: 
# @Project : MyCode
# @FileName: test.py

from df_select_data import select, write_to_csv

file_path = "D:/file/Final/DataProcess/2022-01-07/test.csv"
file_folder = "D:/file/Final/DataProcess/2022-01-07"
file_name = "/test1.csv"
row = 'N5'
offset = -12

select(file_path, file_folder, file_name, row, offset)

