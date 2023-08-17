import torch
from torch import nn
import time
import numpy as np
#加载datetime模块
import datetime
#打印当前时间，年月日，时分秒
now = datetime.datetime.now()


x = torch.tensor([1, 2, 3])
x.device
x = torch.ones(2, 3, device='cuda')
x.device

#我们可以使用 arange 创建一个行向量 x。这个行向量包含以0开始的前12个整数，它们默认创建为整数。
x=torch.arange(12)
x
#可以通过张量的shape属性来访问张量（沿每个轴的长度）的形状 。
x.shape
#张量中元素的总数,一个标量
x.numel()
#改变一个张量的形状而不改变元素数量和元素值，可以调用reshape函数
X = x.reshape(3, 4)
X
X.shape
#有时，我们希望使用全0、全1、其他常量，或者从特定分布中随机采样的数字来初始化矩阵。 我们可以创建一个形状为（2,3,4）的张量，其中所有元素都设置为0。
torch.zeros((2, 3, 4))
torch.ones((2, 3, 4))
torch.randn(3, 4)
#我们还可以通过提供包含数值的Python列表（或嵌套列表），来为所需张量中的每个元素赋予确定值。 在这里，最外层的列表对应于轴0，内层的列表对应于轴1。
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.tensor([[[[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]]]).shape
#运算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
# “按元素”方式可以应用更多的计算，包括像求幂这样的一元运算符
torch.exp(x)
# 连结（concatenate）   0:连接行--行的维度堆叠 1：连接列--列的维度堆叠
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
# 通过逻辑运算符构建二元张量
X == Y
# 对张量中的所有元素进行求和，会产生一个单元素张量。
X.sum()
# 即使形状不同，我们仍然可以通过调用 广播机制（broadcasting mechanism）来执行按元素操作。
# 所以可能会出现问题，它不会立即报错，但是会改变我们想要的张量
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
a + b
# 索引和切片
X,X[-1], X[1:3]
# 指定索引来将元素写入矩阵
X[1, 2] = 9
X
# 为多个元素赋值相同的值
X[0:2, :] = 12
X
# 会显示False,不可取，不必要地分配内存，某些代码可能会无意中引用旧的参数
before = id(Y)
Y = Y + X
id(Y) == before
# 原地操作
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y   #使用切片表示法将操作的结果分配给先前分配的数组
print('id(Z):', id(Z))
# 如果在后续计算中没有重复使用X， 我们也可以使用X[:] = X + Y或X += Y来减少操作的内存开销。
before = id(X)
X += Y
id(X) == before
# 转换为NumPy张量
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
# 要将大小为1的张量转换为Python标量，我们可以调用item函数或Python的内置函数
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)

# 数据预处理
# 创建一个人工数据集，并存储在CSV（逗号分隔值）文件
import os
# os.makedirs(os.path.join('..', 'test_data'), exist_ok=True)  #创建文件夹 退回上一层
os.makedirs(os.path.join('test_data'), exist_ok=True)  #创建文件夹
data_file = os.path.join('test_data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 如果没有安装pandas，只需取消对以下行的注释来安装pandas
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)


# 处理缺失值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

# 对于inputs中的类别值或离散值，我们将“NaN”视为一个类别
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 现在inputs和outputs中的所有条目都是数值类型，它们可以转换为张量格式
import torch
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y

# 线性代数
# 标量由只有一个元素的张量表示。 下面的代码将实例化两个标量，并执行一些熟悉的算术运算，即加法、乘法、除法和指数。
import torch
x = torch.tensor(3.0)
y = torch.tensor(2.0)
x + y, x * y, x / y, x**y
 
#  向量
x = torch.arange(4)
x
len(x) #长度
x.shape   #形状

A = torch.arange(20).reshape(5, 4)
A
# 矩阵的转置
A.T

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
# 转置进行比较
B == B.T

# 就像向量是标量的推广，矩阵是向量的推广一样，我们可以构建具有更多轴的数据结构
X = torch.arange(24).reshape(2, 3, 4)
X

# 给定具有相同形状的任意两个张量，任何按元素二元运算的结果都将是相同形状的张量
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
A, A + B

# 两个矩阵的按元素乘法称为Hadamard积（Hadamard product）
A * B

# 将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘。
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape

#  计算其元素的和
x = torch.arange(4, dtype=torch.float32)
x = torch.arange(4)
x, x.sum()

# 任意形状张量的元素和 
A=torch.arange(20, dtype=torch.float32).reshape(5,4) 
A=torch.arange(20).reshape(5,4)  #可以修改为 A=torch.arange(20*2).reshape(2,5,4) ,看一些多维度的变化
A.shape, A.sum()

# 输入矩阵沿0轴降维以生成输出向量，因此输入轴0的维数在输出形状中消失 ，压缩0维，0维消失
A_sum_axis0 = A.sum(axis=0)   
A_sum_axis0, A_sum_axis0.shape
# 汇总所有列的元素降维（轴1）。因此，输入轴1的维数在输出形状中消失，压缩1维，1维消失
A_sum_axis1 = A.sum(axis=1)  
A_sum_axis1, A_sum_axis1.shape
# 沿着行和列对矩阵求和，等价于对矩阵的所有元素进行求和，压缩0和1维，0，1维消失，但如果是多维的，还会保留其他维度
A.sum(axis=[0, 1])  # 结果和A.sum()相同

# 我们可以调用函数来计算任意形状张量的平均值
A.mean(), A.sum() / A.numel()
# 计算平均值的函数也可以沿指定轴降低张量的维度。
A.mean(axis=0), A.sum(axis=0) / A.shape[0]

# 非降维求和
# 有时在调用函数来计算总和或均值时保持轴数不变会很有用。  对呀的维度没有消失，但会变成1
sum_A = A.sum(axis=1, keepdims=True)
sum_A,sum_A.shape,A.shape
# 由于sum_A在对每行进行求和后仍保持两个轴，我们可以通过广播将A除以sum_A。
A / sum_A

# 某个轴计算A元素的累积总和
A.cumsum(axis=0)


# 点积
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
# 我们可以通过执行按元素乘法，然后进行求和来表示两个向量的点积
torch.sum(x * y)

# 在代码中使用张量表示矩阵-向量积，我们使用mv函数。 当我们为矩阵A和向量x调用torch.mv(A, x)时，
# 会执行矩阵-向量积。 注意，A的列维数（沿轴1的长度）必须与x的维数（其长度）相同。
A.shape, x.shape, torch.mv(A, x)    #其中元素类型要一样，int和float32会报错 RuntimeError: expected scalar type Long but found Float

# 矩阵-矩阵乘法
B = torch.ones(4, 3)
torch.mm(A, B)

# L2范数  向量元素平方和的平方根
u = torch.tensor([3.0, -4.0])
torch.norm(u)
# L1范数   向量元素的绝对值之和
torch.abs(u).sum()
# 矩阵的 Frobenius范数满足向量范数的所有性质   矩阵元素平方和的平方根
torch.norm(torch.ones((4, 9)))


# 自动求导
import torch
x = torch.arange(4.0)
x

# 需要一个地方来存储梯度
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad  # 默认值是None

# 计算y
y = 2 * torch.dot(x, x)
y

y.backward()
x.grad
x.grad == 4 * x

# 现在计算x的另一个函数。
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()  #梯度清零
y = x.sum()  #另外一个函数
y.backward()   #算梯度
x.grad

# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad


# 将某些计算移动到记录的计算图之外
x.grad.zero_()
y = x * x
u = y.detach()   #做成了一个常数，u不是关于x的函数
z = u * x
z.sum().backward()
x.grad == u
# 由于记录了y的计算结果，我们可以随后在y上调用反向传播， 得到y=x*x关于的x的导数，即2*x。
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x


# 即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

# 计算梯度
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

a.grad == d / a

# 线性回归是对n维输入的加权，外加偏差
# 使用平方损失来衡量预测值和真实值的差异
# 线性回归有显示解
# 线性回归可以看作是单层神经网络

# 学习率：沿着梯度方向靠近最优解，不能太大也不能太小
# 批量：整个训练集算梯度太贵，随机采样b个样本来近似损失，b是批量大小，重要的超参数。
# b很大是近似得很精确，但如果有很多样本相同，b大b小时没有区别的，就会浪费计算。
# b很小的时候计算复杂度变小， 但不适合并行最大利用计算资源 。

# 梯度下降通过不断沿着反梯度方向更新参数求解
# 小批量随机梯度下降时深度学习默认的求解算法
# 两个重要的超参数时批量大小和学习率

# 线性回归从零开始实现

import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 生成数据集
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))    #均值0方差1的随机数，长度num_examples，列数len(w)
    y = torch.matmul(X, w) + b                        #  y=X*w+b
    y += torch.normal(0, 0.01, y.shape)               #加入随机噪音，和y的形状一样  
    return X, y.reshape((-1, 1))                      #y做成一个列向量返回，-1表示自适应，1表示一列

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)  #生成特征和标注，标注labels是有噪声的


print('features:', features[0],'\nlabel:', labels[0])

# d2l.set_figsize()
# plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)


# 读取数据集
def data_iter(batch_size, features, labels):   #批量大小，特征，标号 
    num_examples = len(features)                #有多少样本
    indices = list(range(num_examples))         # indices  所有样本的编号
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)                      #打乱 indices 顺序
    for i in range(0, num_examples, batch_size):    #每次 拿 batch_size个样本 
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])  #indices是打乱顺序后的编号，batch_indices拿batch_size个打乱顺序后的编号
        yield features[batch_indices], labels[batch_indices]   #迭代器，会从上次返回的地方继续执行


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break


#  初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)   #需要拟合出来的w,看和true_w的差异
b = torch.zeros(1, requires_grad=True)                        #需要拟合出来的b，看和true_b的差异
# 定义模型
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b
# 定义损失函数
def squared_loss(y_hat, y):  #@save    #y_hat是预测值,y是真实值
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
# 定义优化算法
def sgd(params, lr, batch_size):  #@save   #params是参数,lr是学习率,batch_size是批量大小
    """小批量随机梯度下降"""
    with torch.no_grad():             #更新的时候不要求梯度
        for param in params:          #循环所有参数
            param -= lr * param.grad / batch_size   
            param.grad.zero_()         #梯度初始化


# 训练
lr = 0.01   #学习率太大，loss可能nan
num_epochs = 100
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):  #拿出批量大小的X和y
        l = loss(net(X, w, b), y)  # X和y的小批量损失    net里面做预测
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()         #求和之后算梯度
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)   #整个features输入，做一下损失
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


# 比较真实参数和通过训练学到的参数来评估训练的成功程度
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')


# 线性回归的简洁实现

# 生成数据集
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)   #synthetic_data人工数据合成函数


# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)   # shuffle是否随机打乱

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 使用iter构造Python迭代器，并使用next从迭代器中获取第一项
next(iter(data_iter))   

# nn是神经网络的缩写
from torch import nn
# 第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1。
net = nn.Sequential(nn.Linear(2, 1))    #一个全连接层，输入特征为2，输出特征为1

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()
# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()   #优化器，梯度清理
        l.backward()           #pytorch 做了sum,不用再做了  ，计算梯度 
        trainer.step()        #调用step函数进行梯度更新
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)

# softmax回归

# 回归:单连续数值输出，自然区间R，和真实值的区别作为损失
# 分类:通常多个输出，输出个数是类别个数，输出i是预测为第i类的置信度

# softmax得到每个类的预测置信度，使用交叉熵来衡量预测和标号的区别

# L1 loss  l(y,y') =  |y-y'|
# L2 loss  l(y,y') =  1/2 * |y-y'|^2
# huber's robust loss    l(y,y') = |y-y'|- 1/2   if|y-y'|>1
#                        l(y,y') =  1/2*(y-y')^2   otherwise   


# 图像分类数据集
# Fashion-MNIST数据集

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()


# 读取数据集
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=False)  # train=True下载训练数据集,transform=trans得到tensor而不是图片
mnist_test = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=False)

# 查看数据长度
len(mnist_train), len(mnist_test)
# 每个输入图像的高度和宽度均为28像素。 数据集由灰度图像组成，其通道数为1
mnist_train[0][0].shape

# Fashion-MNIST中包含的10个类别，分别为t-shirt（T恤）、trouser（裤子）、pullover（套衫）、
# dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）
# 和ankle boot（短靴）。 以下函数用于在数字标签索引及其文本名称之间进行转换。
def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 创建一个函数来可视化这些样本。
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# 以下是训练数据集中前几个样本的图像及其相应的标签。
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));

# 读取小批量
batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers()) #num_workers进程数


# 看一下读取训练数据所需的时间。
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'



# 整合所有组件  定义load_data_fashion_mnist函数
def load_data_fashion_mnist(batch_size, resize=None):  #@save 
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))   #resize可以把图片变得更大一点
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="data", train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))




# softmax回归的从零开始实现
import torch
from IPython import display
from d2l import torch as d2l
from torchvision import transforms
import torchvision
from torch.utils import data



batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)  #d2l中是../data，且download=True，所以使用上方修改过的

# 将展平每个图像，把它们看作长度为28*28=784的向量
# 因为我们的数据集有10个类别，所以网络输出维度为10
num_inputs = 784
num_outputs = 10
#初始化权重和偏置
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) 
b = torch.zeros(num_outputs, requires_grad=True)

# 定义softmax操作
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
X.sum(0, keepdim=True), X.sum(1, keepdim=True)

# 实现softmax
def softmax(X):
    X_exp = torch.exp(X)    #每个元素做指数运算
    partition = X_exp.sum(1, keepdim=True)    #1维度求和--压缩1维度
    return X_exp / partition  # 这里应用了广播机制，每一行除以此行的和
# 正如上述代码，对于任何随机输入，我们将每个元素变成一个非负数。 此外，依据概率原理，每行总和为1、
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(1)

# 实现softmax回归模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 创建一个数据样本y_hat，其中包含2个样本在3个类别的预测概率， 以及它们对应的标签y。
# 有了y，我们知道在第一个样本中，第一类是正确的预测； 而在第二个样本中，第三类是正确的预测。 
# 然后使用y作为y_hat中概率的索引， 我们选择第一个样本中第一个类的概率和第二个样本中第三个类的概率。
y = torch.tensor([0, 2])  #真实标号
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])   #预测
y_hat[[0, 1], y]

# 只需一行代码就可以实现交叉熵损失函数。
# 用真实的标号去看对应的预测概率，y中的[0,2]相当于y_chat中0.1和0.5两个概率，对应的是真实标号，
# 因为y_hat第一行预测第三个类别为0.6，但真实情况y是第一个类别，但第一个类别预测概率只有0.1，算交叉熵就会很大

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)

# 将预测类别与真实y元素进行比较
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)   #每一行最大值的下标存起来
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

accuracy(y_hat, y) / len(y)
# 对于任意数据迭代器data_iter可访问的数据集， 我们可以评估在任意模型net的精度
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数  是一个累加器
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())  #y.numel()样本总数
    return metric[0] / metric[1]


# 这里定义一个实用程序类Accumulator，用于对多个变量进行累加。 在上面的evaluate_accuracy函数中，
#  我们在Accumulator实例中创建了2个变量， 分别用于存储正确预测的数量和预测的总数量。 
# 当我们遍历数据集时，两者都将随着时间的推移而累加
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 由于我们使用随机权重初始化net模型， 因此该模型的精度应接近于随机猜测。 例如在有10个类别情况下的精度为0.1
evaluate_accuracy(net, test_iter)

#  softmax回归的训练
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

# 在动画中绘制数据的实用程序类Animator， 它能够简化本书其余部分的代码。
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

# 训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

# 小批量随机梯度下降来优化模型的损失函数，设置学习率为0.1。
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
# 我们训练模型10个迭代周期。 请注意，迭代周期（num_epochs）和学习率（lr）都是可调节的超参数。
#  通过更改它们的值，我们可以提高模型的分类精度
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


# 预测
def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)

# softmax回归的简洁实现
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 初始化模型参数
# softmax回归的输出层是一个全连接层


# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)  #每一层跑一下

# 在交叉熵损失函数中传递未规范化的预测，并同时计算softmax及其对数
loss = nn.CrossEntropyLoss(reduction='none')

# 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)