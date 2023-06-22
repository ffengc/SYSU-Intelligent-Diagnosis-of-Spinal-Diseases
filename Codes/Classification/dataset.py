# %%
import numpy as np
from torch.utils.data import Dataset
import random
from scipy import ndimage
import torch
from PIL import Image
def scipy_rotate(volume):
    # define some rotation angles
    angles = [-20, -10, -5, 5, 10, 20]
    # pick angles at random
    angle = random.choice(angles)
    # rotate volume
    volume = ndimage.rotate(volume, angle, axes=(1, 2), reshape=False)
    volume[volume < 0] = 0 # Jack BUG 0618
    volume[volume > 1] = 1
    return volume

class BasicDataset(Dataset):
    def __init__(self, image_path, label, transform=None):
        self.image_path = image_path
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.image_path)
    def __path2array(self, path):
        first_string = ''
        second_string = ''
        flag = 0 # 表示添加到第一个字符串，1表示添加到第二个字符串
        for i in range(0,len(path)):
            if path[i].isdigit() and flag == 0:
                first_string += path[i]
            if path[i] == '-':
                flag = 1
                continue
            if path[i].isdigit() and flag == 1:
                second_string += path[i]
        first_num = int(first_string)
        second_num = int(second_string)
        arr = np.array([first_num,second_num])
        return arr
            
    def __getitem__(self, idx): # 给这个包重写一个__getitem__
        img = Image.open(self.image_path[idx])
        path_array = self.__path2array(self.image_path[idx][-10:-4]) # 这里传进去有两种可能，第一种是"/151-0", 第二种是"151-10"
        assert(img != None)
        image = np.array(img)
        image = np.resize(image, (3, 224, 224))
        lbl = self.label[idx]
        if lbl != -1: # 如果是-1表示是最终测试
            return torch.from_numpy(image), torch.tensor(lbl, dtype=torch.long)
        else:
            return torch.from_numpy(image), torch.tensor(path_array)