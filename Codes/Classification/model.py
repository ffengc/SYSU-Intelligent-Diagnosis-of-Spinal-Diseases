import torch
import torch.nn as nn
import torchvision
# noinspection PyUnresolvedReferences
from torchsummary import summary
 
 
'''-------------一、构造初始卷积层-----------------------------'''
def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
 
'''-------------二、定义Dense Block模块-----------------------------'''
 
'''---(1)构造Dense Block内部结构---'''
#BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv
class _DenseLayer(nn.Module):
    def __init__(self, inplace, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
           # growth_rate：增长率。一层产生多少个特征图
            nn.Conv2d(in_channels=inplace, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.dropout = nn.Dropout(p=self.drop_rate)
 
    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        return torch.cat([x, y], 1)
 
'''---(2)构造Dense Block模块---'''
class DenseBlock(nn.Module):
    def __init__(self, num_layers, inplances, growth_rate, bn_size , drop_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        # 随着layer层数的增加，每增加一层，输入的特征图就增加一倍growth_rate
        for i in range(num_layers):
            layers.append(_DenseLayer(inplances + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.layers(x)
 
 
'''-------------三、构造Transition层-----------------------------'''
#BN+1×1Conv+2×2AveragePooling
class _TransitionLayer(nn.Module):
    def __init__(self, inplace, plance):
        super(_TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplace,out_channels=plance,kernel_size=1,stride=1,padding=0,bias=False),
            nn.AvgPool2d(kernel_size=2,stride=2),
        )
 
    def forward(self, x):
        return self.transition_layer(x)
 
 
'''-------------四、搭建DenseNet网络-----------------------------'''
class DenseNet(nn.Module):
    def __init__(self, init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16],num_classes=10):
        super(DenseNet, self).__init__()
        bn_size = 4
        drop_rate = 0
        self.conv1 = Conv1(in_planes=3, places=init_channels)
 
        blocks*4
 
        #第一次执行特征的维度来自于前面的特征提取
        num_features = init_channels
 
        # 第1个DenseBlock有6个DenseLayer, 执行DenseBlock（6,64,32,4）
        self.layer1 = DenseBlock(num_layers=blocks[0], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        # 第1个transition 执行 _TransitionLayer（256,128）
        self.transition1 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        #num_features减少为原来的一半，执行第1回合之后，第2个DenseBlock的输入的feature应该是：num_features = 128
        num_features = num_features // 2
 
        # 第2个DenseBlock有12个DenseLayer, 执行DenseBlock（12,128,32,4）
        self.layer2 = DenseBlock(num_layers=blocks[1], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        # 第2个transition 执行 _TransitionLayer（512,256）
        self.transition2 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        # num_features减少为原来的一半，执行第2回合之后，第3个DenseBlock的输入的feature应该是：num_features = 256
        num_features = num_features // 2
        # 第3个DenseBlock有24个DenseLayer, 执行DenseBlock（24,256,32,4）
        self.layer3 = DenseBlock(num_layers=blocks[2], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        # 第3个transition 执行 _TransitionLayer（1024,512）
        self.transition3 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        # num_features减少为原来的一半，执行第3回合之后，第4个DenseBlock的输入的feature应该是：num_features = 512
        num_features = num_features // 2
 
        # 第4个DenseBlock有16个DenseLayer, 执行DenseBlock（16,512,32,4）
        self.layer4 = DenseBlock(num_layers=blocks[3], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[3] * growth_rate
 
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(num_features, num_classes)
 
    def forward(self, x):
        x = self.conv1(x)
 
        x = self.layer1(x)
        x = self.transition1(x)
        x = self.layer2(x)
        x = self.transition2(x)
        x = self.layer3(x)
        x = self.transition3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
 
def DenseNet121(num_class = 10):
    return DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16], num_classes = num_class)
'''
if __name__=='__main__':
    # model = torchvision.models.densenet121()
    model = DenseNet121()
    print(model)
    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
    '''
if __name__ == '__main__':
 
     net = DenseNet(num_classes=10).cuda()
     summary(net, (3, 224, 224))