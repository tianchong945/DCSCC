import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms


# 随机数种子
np.random.seed(0)


class GaussianBlur(object):
    """blur a single image on CPU""" # 在CPU上模糊图像
    def __init__(self, kernel_size):
        radias = kernel_size // 2 # // 除后整数部分
        kernel_size = radias * 2 + 1
        # nn.Conv2d参数 输入输出通道=3 卷积核尺寸 卷积步长=1 padding 每一条边补充0的层数 bias=False 不添加偏置
        # groups=3 groups(int, optional) – 输入3组数据，每组分别对应一个通道进行卷积，此时每个输出通道只需要在其中输入通道上卷积
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        # nn.Sequential
        # 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
        # 同时以神经网络模块为元素的有序字典也可以作为传入参数
        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0) # unsqueeze(0)在第0维增加一个维度

        # 从[0.1,2.0)中随机取一个值
        sigma = np.random.uniform(0.1, 2.0)
        # np.arange() 返回一个序列，左为起点 右为终点，步长为1
        x = np.arange(-self.r, self.r + 1)
        # np.exp()：返回e的幂次方
        # np.power(x, 2)为x^2
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        # sum 将各项相加
        x = x / x.sum()
        # torch.from_numpy（x） 将x转化为张量形式
        # view 输出数组大小为1*x （-1表示自行判断）
        # repeat（3,1）对张量重复扩张（列重复3倍，行不重复）
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        # ？？？？？？什么作用  不确定： 使用预训练的词向量，在此处指定预训练的权重
        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        # 反向传播时不会自动求导了，节约了显存
        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img