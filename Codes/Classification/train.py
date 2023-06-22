
# 偶数是五分类的，奇数是二分类

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
# split_idx = int(len(df_shuffle) * 0.8)
df_train = df_shuffle # .iloc[:split_idx]

train_data_path = list(df_train['path'])
train_labels = list(df_train['label'])

train_transform = None 
train_set = BasicDataset(train_data_path, train_labels, train_transform)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=1)
print("Data Load success!")


# 初始化模型并定义损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 选择机器
num_class = 2
if option == "disc":
    num_class = 5
model = DenseNet121(num_class).to(device) # 五分类或而分类
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=1e-4)  # Adam这个优化器比较重要

save_model = 2
train_epoch_max = 100

f_training_acc = open(f'/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/TrainingInfo/training_{option}_acc.log','w')
f_training_acc.write('epoch,train_acc,train_loss')
for epoch in range(train_epoch_max):
    training_acc = []
    training_loss = []
    inner_epoch = 0
    for data in train_loader:
        inner_epoch += 1
        print(f'training... inner_epoch: {inner_epoch}/{len(train_loader)}',end='\r')
        image, label = data
        
        image = image.cuda().float()
        label = label.cuda()
        logit = model(image)
        loss = criterion(logit, label)
        pred = torch.argmax(logit, dim=-1)  # 硬分类
        
        cur_acc = get_acc(pred, label)
        cur_loss = loss.cpu().item()
        training_acc.append(cur_acc)
        training_loss.append(cur_loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('\n',end='') # 换行
    training_acc = np.array(training_acc)
    training_loss = np.array(training_loss)
    epoch_acc = np.mean(training_acc)
    epoch_loss = np.mean(training_loss)
    # 输出当前训练的信息
    msg = "epoch: {:d}, acc: {:.4f}, loss: {:.4f}".format(epoch, epoch_acc, epoch_loss)
    print(msg)
    f_training_acc.write(msg+'\n')
    # 保存模型
    if epoch % save_model == 0:
        model_save_path = root + f"./models/{option}/model_{epoch}.pth"
        torch.save(model,model_save_path)