import xlrd
import csv
import os
import xlsxwriter
from xlutils.copy import copy
import openpyxl
from openpyxl.utils import get_column_letter,column_index_from_string
from openpyxl import Workbook

workbook = xlsxwriter.Workbook("/home/huangjiehui/Project/PDAnalysis/Tools/Data.csv")
worksheet = workbook.add_worksheet()

imgs_dir_0 = "/data1/hjh/ProjectData/PDData/0result_ALL/"
imgs_dir_1 = "/data1/hjh/ProjectData/PDData/1result_ALL/"
imgs_dir_2 = "/data1/hjh/ProjectData/PDData/2result_ALL/"
imgs_dir_3 = "/data1/hjh/ProjectData/PDData/3result_ALL/"
imgs_dir_4 = "/data1/hjh/ProjectData/PDData/4result_ALL/"

worksheet.write(0, 0,"data_path")
worksheet.write(0, 1,"label_list")
PD_count=0 # 总行数
for path, subdirs, files in os.walk(imgs_dir_0):
    for i in range(len(files)):
        worksheet.write(i+1, 0, path+files[i])
        worksheet.write(i+1, 1 ,0)
        PD_count=PD_count+1
    print("Successed 00 Files") 

for path, subdirs, files in os.walk(imgs_dir_1):
    for j in range(0,len(files)): 
        worksheet.write(PD_count+j, 0, path+files[j])
        worksheet.write(PD_count+j, 1, 1)
    PD_count=PD_count+len(files)
    print("Successed 01 Files")
    
for path, subdirs, files in os.walk(imgs_dir_2):
    for j in range(0,len(files)): 
        worksheet.write(PD_count+j, 0, path+files[j])
        worksheet.write(PD_count+j, 1, 2)
    PD_count=PD_count+len(files)
    print("Successed 02 Files")
    
for path, subdirs, files in os.walk(imgs_dir_3):
    for j in range(0,len(files)): 
        worksheet.write(PD_count+j, 0, path+files[j])
        worksheet.write(PD_count+j, 1, 3)
    PD_count=PD_count+len(files)
    print("Successed 03 Files")
    
for path, subdirs, files in os.walk(imgs_dir_4):
    for j in range(0,len(files)): 
        worksheet.write(PD_count+j, 0, path+files[j])
        worksheet.write(PD_count+j, 1, 4)
    print("Successed 04 Files")

workbook.close()

import pandas as pd
print(pd.__version__)
df = pd.read_excel("/home/huangjiehui/Project/PDAnalysis/Tools/Data.csv",index_col=0) # xlrd 2.0版本以上不支持
# df = data.dropna(how="all")
df.to_csv("/home/huangjiehui/Project/PDAnalysis/Tools/Data.csv", encoding='utf-8')

print("Successful!")