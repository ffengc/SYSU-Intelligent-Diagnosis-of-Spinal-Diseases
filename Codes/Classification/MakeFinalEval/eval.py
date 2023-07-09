

import os
import numpy as np
import pandas as pd

eval_root_path = '/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/MakeFinalEval/'

def getTxtDataframe(idx) -> pd.DataFrame:  # 这个函数应该返回两个DataFrame类型
    first_path = eval_root_path + f"./MakeEvalOutput/{idx}-000.txt"
    if idx < 10:
        second_path = eval_root_path + f"./result/00{idx}.txt"
    else:
        second_path = eval_root_path + f"./result/0{idx}.txt"
    df_pred = pd.read_csv(first_path)
    df_true = pd.read_csv(second_path, sep = ' ')
    return df_true, df_pred

def get_pred_targets(key, arr): # 得到预测的指标，分离出来
    for i in range(0,len(arr)):
        if arr[i][0] == key:
            return arr[i][1], arr[i][2], arr[i][3]
    return None, None, None # 如果没找到 -- 返回空

def eval_dist(x1, y1, x2, y2): # 计算两个点的欧几里得距离
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

# 一共要处理50次txt文件，一次处理两个
TP = 0 # 真正例
FP = 0 # 假正例
FN = 0 # 假负例
TN = 0 # 真负例
CORRECT_DETECTION = 0 # 检测正确
CORRECT_CLASSIFICATION = 0 # 分类正确
SAMPLES = 0 # 计算一共有多少张图片


for i in range(0,50):
    true_df, pred_df = getTxtDataframe(i)
    # 现在需要处理这两个df文件
    true_arr = true_df.to_numpy().tolist()
    pred_arr = pred_df.to_numpy().tolist()
    # 现在是列表类型
    # assert(len(true_arr) >= len(pred_arr)) # 真实的只会比预测的长，因为预测的做了去重 #BUG
    # 最好不要独立成接口了，因为独立之后要用global，很麻烦
    for i in range(0,len(true_arr)):
        SAMPLES += 1
        key = true_arr[i][0]
        true_x = true_arr[i][1]
        true_y = true_arr[i][2]
        true_label = true_arr[i][3]
        pred_x, pred_y, pred_label = get_pred_targets(key, pred_arr)
        # 先检查返回是不是空，如果是，表示没找到
        if pred_x == None:
            # 没找到
            FN += 1 # 判为假负例
            continue
        # 走到这里表示表示找到了
        cur_dist = eval_dist(true_x, true_y, pred_x, pred_y)
        if cur_dist < 8 and pred_label == true_label:
            # 预测正确 
            TP += 1 # 真正例
        else:
            # 其他情况都是预测错误
            FP += 1 # 假正例
        if cur_dist < 8:
            CORRECT_DETECTION += 1
        if pred_label == true_label:
            CORRECT_CLASSIFICATION += 1

# 循环结束之后四个指标已经处理好
ACC = TP / (TP + FP)
RECALL = TP / (TP + FN)
AP = TP / (TP + FP + FN)
CORRECT_CLASSIFICATION_RATE = CORRECT_CLASSIFICATION / SAMPLES
CORRECT_DETECTION_RATE = CORRECT_DETECTION / SAMPLES
msg1 = f'acc: {ACC}, recall: {RECALL}, ap: {AP}'
msg2 = f'CORRECT_CLASSIFICATION_RATE: {CORRECT_CLASSIFICATION_RATE}, CORRECT_DETECTION_RATE: {CORRECT_DETECTION_RATE}'
print(msg1)
print(msg2)
f = open('/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/MakeFinalEval/eval.log','w')
f.write('6月17日测试:\n')
f.write(msg1 + '\n')
f.write(msg2 + '\n')
print('done!')