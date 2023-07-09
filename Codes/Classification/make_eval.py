


import pandas as pd
import numpy as np
import os
import cv2

df = pd.read_csv('/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/TestingInfo/final_test.log')
arr = df.to_numpy()
arr_list = arr.tolist()
print('load data sucess')

def screenArray(target, arr):
    ret = []
    for i in range(0,len(arr)):
        if arr[i][0] == target:
            ret.append(arr[i])
    return ret

labels_path = '/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data-ForTest/Labels'
def get_spc_path(idx):
    for root, dirs, files in os.walk(labels_path):
        if idx < 10:
            txt_path = f'00{idx}-000.txt'
        else:
            txt_path = f'0{idx}-000.txt'
        if txt_path in files:
            file_path = os.path.join(root, txt_path)
            return file_path

def deduplicationForLst(lst):
    seen = set()
    result = [row for row in lst if not (row[0] in seen or seen.add(row[0]))]
    return result

def find_from_screened_array(arr, target):
    # arr 是一个 3列 的二维矩阵
    for i in range(0,len(arr)):
        if arr[i][1] == target:
            return arr[i][2]
    return -1

def processing_array(arr, idx): # 把相对的左边变成真实的坐标，然后只留下x和y就行了
    # 先去掉后两列
    arr = [row[:-2] for row in arr]
    # 找到原图，得到原图的长和宽
    img_path = '/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data-ForTest/images'
    for root, dirs, files in os.walk(img_path):
        if idx < 10:
            file_path = f'00{idx}-000.jpg'
        else:
            file_path = f'0{idx}-000.jpg'
        if file_path in files:
            file_path = os.path.join(root, file_path)
            break
    image = cv2.imread(file_path, 0)
    row, col = image.shape
    # 相对坐标转真实坐标
    for i in range(0,len(arr)):
        arr[i][1] *= col
        arr[i][2] *= row
    return arr
    
output_root_path = '/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/MakeEvalOutput/'
for i in range(0,50):
    # 先把arr里面151开头的筛选出来
    screened_array = screenArray(i, arr_list)
    # 现在要找到那个txt文件
    txt_file_path = get_spc_path(i)
    txt_array = pd.read_csv(txt_file_path, sep=' ').to_numpy().tolist()
    txt_array = processing_array(txt_array, i) # 去掉w和h，转化成真实坐标
    txt_array = deduplicationForLst(txt_array)
    assert(len(txt_array) <= 11) # 检查一下是否是找到了11块骨头
    for j in range(0,len(txt_array)):
        cur_idx = txt_array[j][0]
        cur_pred = find_from_screened_array(screened_array, cur_idx)
        txt_array[j].append(cur_pred + 1) # 因为我的标签和真实标签是差1的
    column = ['bone_idx', 'x', 'y', 'pred']
    cur_df = pd.DataFrame(txt_array, columns=column)
    output_path = output_root_path + f'{i}-000.txt'
    cur_df.to_csv(output_path, sep=',', index=False)
    
print('done!')