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

option = "vertebra" # disc / vertebra

# 代码路径
root = "/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/"

# 选设备
device = torch.device("cuda")

# load data
df = pd.read_csv(f'/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/{option}-data.csv')
df_shuffle = df.sample(frac=1, random_state=100)  # 洗牌
split_idx = int(len(df_shuffle) * 0.8)
df_test = df_shuffle.iloc[split_idx:]

test_data_path = list(df_test['path'])
test_labels = list(df_test['label'])

test_transform = None 
test_set = BasicDataset(test_data_path, test_labels, test_transform)
test_loader = DataLoader(test_set, batch_size=8, shuffle=True, num_workers=1)
print("Data Load success!")


# load model
f = open(f'/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/TestingInfo/acc_{option}.log','w')
f.write('model,acc,loss')
for i in range(0,99,2):
    model_path = root + f'./models/{option}/model_{i}.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 选择机器
    model = torch.load(model_path).to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    testing_acc = []
    testing_loss = []
    inner_epoch = 0
    for data in test_loader:
        inner_epoch += 1
        print(f'testing... inner_epoch: {inner_epoch}/{len(test_loader)}',end='\r')
        image, label = data
        
        image = image.cuda().float()
        label = label.cuda()
        logit = model(image)
        loss = criterion(logit, label)
        pred = torch.argmax(logit, dim=-1)  # 硬分类
        
        cur_acc = get_acc(pred, label)
        cur_loss = loss.cpu().item()
        testing_acc.append(cur_acc)
        testing_loss.append(cur_loss)

    testing_acc = np.array(testing_acc)
    testing_loss = np.array(testing_loss)
    final_acc = np.mean(testing_acc)
    final_loss = np.mean(testing_loss)
    # 输出当前测试的信息
    msg = "model: {:d}, acc: {:.4f}, loss: {:.4f}".format(i, final_acc, final_loss)
    print(msg)
    f.write(msg+'\n')