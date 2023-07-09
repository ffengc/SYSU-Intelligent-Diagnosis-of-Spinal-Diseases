import xlrd
import csv
import os
import xlsxwriter
from xlutils.copy import copy
import openpyxl
from openpyxl.utils import get_column_letter,column_index_from_string
from openpyxl import Workbook
import pandas as pd


df_disc = pd.read_csv('/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/disc-data.csv')
df_vertebra = pd.read_csv('/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/vertebra-data.csv')

imgs_dir_0 = "/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/disc/v1/"
imgs_dir_1 = "/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/disc/v2/"
imgs_dir_2 = "/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/disc/v3/"
imgs_dir_3 = "/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/disc/v4/"
imgs_dir_4 = "/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/disc/v5/"

for path, subdirs, files in os.walk(imgs_dir_0):
    for i in range(len(files)):
        cur_path = path+files[i]
        cur_label = 0
        new_row = {'path':cur_path,'label':cur_label}
        df_disc = df_disc.append(new_row, ignore_index=True)

for path, subdirs, files in os.walk(imgs_dir_1):
    for i in range(len(files)):
        cur_path = path+files[i]
        cur_label = 1
        new_row = {'path':cur_path,'label':cur_label}
        df_disc = df_disc.append(new_row, ignore_index=True)

for path, subdirs, files in os.walk(imgs_dir_2):
    for i in range(len(files)):
        cur_path = path+files[i]
        cur_label = 2
        new_row = {'path':cur_path,'label':cur_label}
        df_disc = df_disc.append(new_row, ignore_index=True)
        
for path, subdirs, files in os.walk(imgs_dir_3):
    for i in range(len(files)):
        cur_path = path+files[i]
        cur_label = 3
        new_row = {'path':cur_path,'label':cur_label}
        df_disc = df_disc.append(new_row, ignore_index=True)
        
for path, subdirs, files in os.walk(imgs_dir_4):
    for i in range(len(files)):
        cur_path = path+files[i]
        cur_label = 4
        new_row = {'path':cur_path,'label':cur_label}
        df_disc = df_disc.append(new_row, ignore_index=True)
        
        
        
df_disc.to_csv('/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/disc-data.csv',index=False)

imgs_dir_0 = "/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/vertebra/v1/"
imgs_dir_1 = "/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/vertebra/v2/"

for path, subdirs, files in os.walk(imgs_dir_0):
    for i in range(len(files)):
        cur_path = path+files[i]
        cur_label = 0
        new_row = {'path':cur_path,'label':cur_label}
        df_vertebra = df_vertebra.append(new_row, ignore_index=True)

for path, subdirs, files in os.walk(imgs_dir_1):
    for i in range(len(files)):
        cur_path = path+files[i]
        cur_label = 1
        new_row = {'path':cur_path,'label':cur_label}
        df_vertebra = df_vertebra.append(new_row, ignore_index=True)
        
df_vertebra.to_csv('/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/vertebra-data.csv',index=False)
print('successful!')