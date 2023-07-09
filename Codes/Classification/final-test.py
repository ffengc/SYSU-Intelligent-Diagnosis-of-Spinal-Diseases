# nvidia-smi
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataset import BasicDataset
import pandas as pd # 对panda做处理
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn # 做深度学习！
from email import parser
import os
import torch
from model import DenseNet121
from tools import get_acc
import numpy as np


# 代码路径
root = "/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/"

# 选设备
device = torch.device("cuda")

# load data
df = pd.read_csv(f'/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data-ForTest/test-data.csv')
# df_shuffle = df.sample(frac=1, random_state=100)  # 最终测试不用打乱
df_test = df

test_data_path = list(df_test['path'])
test_labels = list(df_test['label'])

test_transform = None 
test_set = BasicDataset(test_data_path, test_labels, test_transform)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1) # 最终测试的时候设置 batch_size = 1
print("Data Load success!")

# load model
model_v2_path = root + f'./Data-ForTest/Model/vertebra-model.pth'
model_v5_path = root + f'./Data-ForTest/Model/disc-model.pth' # disc是五分类 vertebra是二分类
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 选择机器
model_v2 = torch.load(model_v2_path).to(device)
model_v5 = torch.load(model_v5_path).to(device)
model_v2.eval()
model_v5.eval()
criterion = nn.CrossEntropyLoss()


# 预测结果输出的文件
output_f = open('/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/TestingInfo/final_test.log','w')
output_f.write('image_index,bone_index,predict\n')

inner_epoch = 0
for data in test_loader:
    inner_epoch += 1
    print(f'testing... inner_epoch: {inner_epoch}/{len(test_loader)}',end='\r')
    image, cur_img_path = data
    # cur_img_path.tolist()[0][0]是第一个数字 -- 表示图片序号
    # cur_img_path.tolist()[0][1]是第二个数字 -- 表示是第几块骨头
    img_index = cur_img_path.tolist()[0][0]
    bone_index = cur_img_path.tolist()[0][1]
    
    image = image.cuda().float()

    if bone_index % 2 == 0: # 偶数 -- 五分类
        logit = model_v5(image)
        pred = torch.argmax(logit, dim=-1)  # 硬分类
    elif bone_index % 2 == 1: # 奇数 -- 二分类
        logit = model_v2(image)
        pred = torch.argmax(logit, dim=-1)  # 硬分类
    # 现在已经知道了图片的序号+骨头的序号+对应的预测结果
    msg = f'{img_index},{bone_index},{pred.item()}' + '\n'
    output_f.write(msg)