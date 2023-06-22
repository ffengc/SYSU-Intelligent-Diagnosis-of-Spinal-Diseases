# SYSU-Intelligent-Diagnosis-of-Spinal-Diseases
Intelligent Diagnosis Project for Spinal Diseases, Final Course Design of the Second Semester Deep Learning Course in sophomore year

该项目所有代码存置于`Codes/`目录下。

**进入目录：`cd Codes/`**

### 环境准备

我们在 **python=3.8**，**Pytorch>=1.7.0**，**GPU 显存>=4G**的环境进行实验。

所需外部包见：`./requirements.txt`

**执行：`pip install -r requirements.txt` 安装所有依赖。**

### 检测网络

**进入检测网络目录：`cd tph-yolov5/`**

#### 目录结构

目录名称为`tph-yolov5/`，其中：

- data/文件夹，用来存放训练集、验证集及测试集。

- run/文件夹，用来保存训练结果和推理结果。

- models/文件夹，用来存放tph-yolov5模型文件。

- train.py，用以模型的训练

- detect.py，用以推理。

接下来将会解释详细功能及运行指令。

#### 数据准备

通过前面的数据处理，我们得到了前150张图片以下数据：

- jpg文件：每张图片的原图，命名为`xxx-000.jpg`

- txt文件：每张图片的标签，命名为`xxx-000.txt`

在该txt中，每一行代表每个小部位，而每一列从左到右依次是：编号、x坐标、y坐标、宽、高。

我们将150张照片中的前140张作为训练集，后10张作为验证集。

- 训练集的jpg文件存放在`tph-yolov5/data/data1/images/train`中；

- 训练集的txt文件存放在`tph-yolov5/data/data1/labels/tarin`中，验证集同理。

最后，在`./data/train.yaml`文件中指定好数据存放的路径，即可开始训练环节。

#### 模型训练

我们采用以下指令进行模型训练：
 `python train.py --batch 4 --epochs 80 --data tph-yolov5/data/train.yaml --weights tph-yolov5/yolov5l-xs-1.pt`

其中，首先指定训练代码train.py，然后指定batch和epoch，再通过train.yaml 文件获取数据集的存放位置，最后指定模型的权重文件yolov5l-xs-1.pt，即可开始训练。

**训练完成后，新的权重文件将会保存在`tph-yolov5/runs/train/xxx`中。**

我们经过多次的调参，最后得到了效果最好的权重文件，`exp7/weights/best.pt`。我们将在接下来的测试阶段使用这个文件。

#### 检测模型测试

执行如下指令即可测试：

`python detect.py --source tph-yolov5/data/test --weights tph-yolov5/runs/train/exp7/weights/best.pt --save-txt`

​		其中，首先指定测试的代码文件detect.py，然后指定需要测试的图片，存放在文件夹test中，再指定训练部分我们训练出来的最佳模型权重best.pt，即可开始训练。我们多了一个指令--save-txt，是为了将模型的分类参数保存下来，用以下一步的分类模型，以及用来量化我们检测任务的完成情况，这一部分在评价指标部分会有体现。

**测试结果会保存在`tph-yolov5/runs/detect`文件夹中。**

### 分类网络

**分类网络代码包括分类网络代码+最终模型评价代码**

#### 目录结构

目录名称为`Classification/`，该目录内部内容整体结构可见`tree.output`

先进入`Classification/`文件夹，`cd Classification/`

如果目录下没有`tree.output`，可以`touch tree.output; tree . > tree.output`让OS打印目录结构并重定向到`tree.output`文件中。(没有`tree`可以执行`sudo yum install tree`进行安装)

**.py文件功能简介：**

`train.py`训练

`test.py`分类模型的单独测试

`model.py` 模型的定义

`tools.py`一些工具接口

`dataset.py`重写torch的dataset

`final-test.py`最终的整合测试

`make_eval.py` 生成供最终指标计算的txt文件

详细功能我会在后面内容中介绍。

#### 数据准备

​		首先，通过手册的json文件，和原图，**我们将200张图片的前150张（作为分类网络的训练集）**，按照json文件的信息，把原图切割成若干小图，`./Data/disc/`和`./Data/vertebra/`分别存放椎间盘和椎体的若干图片(.jpg)。得到切割好的结果之后，再移动到`./Classification/`目录下进行数据的准备和整理。

​		进入当前目录下`Data/`目录，该目录用来存放分类模型所有的训练数据。

`./Data/disc/`和`./Data/vertebra/`分别存放椎间盘和椎体的所有图片数据(.jpg)、

**创建数据索引的csv文件：**`touch disc-data.csv` 和 `touch vertebra-data.csv`

**给csv文件输入列名：**分别在两个csv文件开头输入`path,label`+回车

**运行：`LabelMake.py`** `python LabelMake.py`

运行后所有的图片及其标签值会被存储在csv文件当中，形式如下所示：

**vertebra-data.csv**

```bash
path,label
/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/vertebra/v1/091.jpg,0
/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/vertebra/v1/353.jpg,0
/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/vertebra/v1/170.jpg,0
/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/vertebra/v1/139.jpg,0
/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/vertebra/v1/349.jpg,0
/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/vertebra/v1/531.jpg,0
/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data/vertebra/v1/232.jpg,0
```

#### 模型训练

**模型训练有两种模式：`disc`模式和`vertebra`模式**

在`./train.py`开头的代码中选择option参数，可以调整训练模式

```python
option = "vertebra" # disc / vertebra
```

**注意：`train.py`里的所有路径采用绝对路径的方式，因此执行代码时需要自行调整代码中所有的路径参数。**

- `train.py`运行结束之后，所有的模型`.pth`文件会被保存到`./models/vertebra/`和`./models/disc/`下，供测试使用。

- 所有训练信息会被保存到`./TrainingInfo`目录下

#### 分类模型测试

**这里的测试不是项目最终的性能测试，仅仅只是分类模型性能的测试，作用是通过测试挑选出最优的分类模型，供后续最终测试使用。**

**代码：`test.py`**

执行方式和`train.py`一样，需要选择训练模式（disc/vertebra）和更改所有路径。

- 测试代码会在`./models/vertebra/`和`./models/disc/`中读取`.pth`文件，并对测试集进行测试。
- 测试信息会被保存到`./TestingInfo`目录下

### 模型最终测试和评价指标的计算

**此部分代码也在`./Classification/`目录下。**

经过上一步骤分类模型测试之后，我们可以通过查看`./TestingInfo`里面的测试信息，挑选出两个拥有最优性能的模型（disc的最优模型和vertebra的最优模型），并对其进行重命名，放置到特定目录下：`./Data-ForTest/Model/disc-model.pth`和`./Data-ForTest/Model/vertebra-model.pth`。

#### 最终测试的数据准备

无论是检测网络的训练还是分类网络的训练，我们都是取了200张图片的前150张进行训练，后50张进行测试。

上层检测网络，对后50张图片进行测试之后，得到一系列小图片和label的txt文件，把他们和原图（后50张）一起，放到以下特定目录下：

`./Data-ForTest/Data/` 存放检测网络分割后的小图片**，每一类用数字0～10进行编号，（0表示T12-L1, 1表示L1, 2表示L1-L2以此类推供11块骨头）**

`./Data-ForTest/images/` 存放后50张原图片（大图）

`./Data-ForTest/Labels/` 存放检测网络对50张图片处理后的输出结果，里面的形式如下所示。

```bash
2 0.527344 0.326172 0.132812 0.0898438
4 0.515625 0.416016 0.132812 0.0976562
5 0.503906 0.464844 0.140625 0.101562
7 0.494141 0.5625 0.144531 0.101562
3 0.519531 0.371094 0.125 0.09375
8 0.494141 0.615234 0.144531 0.113281
0 0.537109 0.242188 0.121094 0.0859375
1 0.533203 0.283203 0.113281 0.0820312
6 0.5 0.513672 0.140625 0.0976562
9 0.496094 0.667969 0.148438 0.09375
10 0.505859 0.71875 0.128906 0.101562
```

**第一列表示一张图片中骨头的序号，第二列到第五列分别表示yolo画框的中心点x，中心点y，长和宽。**

**（0表示T12-L1, 1表示L1, 2表示L1-L2以此类推共11块骨头，偶数表示是椎间盘，作五分类，奇数表示是椎体，作二分类）**

**执行`./Data-ForTest/Data/rename.sh`对每张图片进行重命名，在它原有名字上，加上它的骨头的编号，为什么要这样做我后续解释。**

**创建`./Data-ForTest/test-data.csv`**，图片路径索引的csv文件，在这个csv文件里面，我设置所有的label为-1，因为最终测试中，label已经不重要了，我们有特定的评价指标。`touch test-data.csv`

**给csv文件输入列名：**分别在两个csv文件开头输入`path,label`+回车

**执行`./Data-ForTest/LabelMake.py`向`test-data.csv`里面填内容。**

格式如下：

```bash
path,label
/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data-ForTest/Data/0/032-0.jpg,-1
/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data-ForTest/Data/0/006-0.jpg,-1
/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data-ForTest/Data/0/013-0.jpg,-1
/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data-ForTest/Data/0/021-0.jpg,-1
/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data-ForTest/Data/0/037-0.jpg,-1
/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data-ForTest/Data/0/023-0.jpg,-1
/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data-ForTest/Data/0/002-0.jpg,-1
/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data-ForTest/Data/0/017-0.jpg,-1
/home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data-ForTest/Data/0/007-0.jpg,-1
```

#### 模型最终测试

**执行：`final-test.py`**

在`dataset.py`里面，我通过判断`label`的值，判断是否代码是在运行`final-test.py`还是`test.py`，如果label是-1，表示我们正在进行最终测试，此时，**`dataset.py`将会返回一个图片的名字后的编号。**可以看到，通过前面`rename.sh`的作用，我们在dataset内部，可以得知该小图片，是属于哪一个骨头的，这样，我们就能在`final-test.py`里面决定，是用五分类模型进行预测，还是二分类模型进行预测。

`final-test.py`执行后，最终测试的结果会被存储在`./TestingInfo/final_test.log`文件里，格式如下：

```bash
image_index,bone_index,predict
19,3,1
43,7,1
7,8,1
42,7,1
47,1,1
```

第一列表示它属于哪一张大图，第二列表示它被yolo预测成了哪一块骨头，第三列表示这块骨头被预测成了哪一类。

#### 最终性能指标的计算

为了计算最后的性能指标，我们需要四部分文件进行计算

- `./TestingInfo/final_test.log`
- `./Data-ForTest/Labels/` 检测网络给我们提供的50个txt文件，每个里面包括了检测网络的分割结果（5.4.1节已经介绍过了）
- `./Data-ForTest/images/` 50张原始图片
- `./Data-ForTest/result/` 50个txt文件，但是这个有别于`./Data-ForTest/Labels/` ，这个是准确的分割结果，是通过手册提供的json所得

执行`make_eval.py`，把`./TestingInfo/final_test.log`和`./Data-ForTest/Labels/` 的结果对应起来，得到50个标准格式的txt文件**（`./Data-ForTest/result/`的格式与之相同，因此通过50组，每组两个txt文件的相互比对，可以计算出中心点的差值，也可以比对pred和真实的label，从而计算性能指标）**，格式如下：

```bash
bone_idx,x,y,pred
4.0,132.0,106.500096,1
5.0,128.999936,119.000064,1
7.0,126.500096,144.0,2
3.0,132.999936,95.000064,2
8.0,126.500096,157.499904,1
0.0,137.499904,62.000128,1
1.0,136.499968,72.499968,2
6.0,128.0,131.500032,2
9.0,127.000064,171.000064,2
10.0,129.499904,184.0,1

```

第一列表示骨头的序号，第二列表示真实中心点的x坐标，第三列表示y坐标，第四列表示预测值。

最后将真实的和预测的，50个txt文件，分别整理到`./MakeFinalEval/MakeEvalOutput/`目录和`./MakeFinalEval/result/`目录下。

执行`./MakeFinalEval/eval.py`，即可得到最后的结果，存储在`./MakeFinalEval/eval.log`下。
