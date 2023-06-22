import xlrd
import csv
import os
import xlsxwriter
from xlutils.copy import copy
import openpyxl
from openpyxl.utils import get_column_letter,column_index_from_string
from openpyxl import Workbook
import pandas as pd


df = pd.read_csv('/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data-ForTest/test-data.csv')


for i in range(0,11):
    dir_path = f"/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data-ForTest/Data/{i}/"
    for path, subdirs, files in os.walk(dir_path):
        for i in range(len(files)):
            cur_path = path+files[i]
            if cur_path[-3:] != "jpg":
                continue
            new_row = {'path':cur_path,'label': -1}
            df = df.append(new_row, ignore_index=True)
    
df.to_csv('/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data-ForTest/test-data.csv',index=False)