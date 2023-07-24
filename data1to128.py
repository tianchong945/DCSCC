import pickle
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_aug.gaussian_blur import GaussianBlur
from get_aug import get_dataug
import random

# 第一部分  数据集来源 地址
"""
数据来源为1 2 3 4 5 6 7 8 9 10 11 12 13 14
具体数据对应，见goodnote 草稿文档
图片进行翻转、裁剪变换
"""
class MyDataset1to128(Dataset):
    def __init__(self,txtpath1,txtpath2,txtpath3,txtpath4,txtpath5,txtpath6,txtpath7,
                 txtpath8,txtpath9,txtpath10,txtpath11,txtpath12,txtpath13,txtpath14):

        # 打开第一步创建的txt文件，按行读取，将结果以元组方式保存在imgs里
        self.imgs = ['','','','','','','','','','','','','','','']
        self.imgs[1] = self.txtool(txtpath1)
        self.imgs[2] = self.txtool(txtpath2)
        self.imgs[3] = self.txtool(txtpath3)
        self.imgs[4] = self.txtool(txtpath4)
        self.imgs[5] = self.txtool(txtpath5)
        self.imgs[6] = self.txtool(txtpath6)
        self.imgs[7] = self.txtool(txtpath7)
        self.imgs[8] = self.txtool(txtpath8)
        self.imgs[9] = self.txtool(txtpath9)
        self.imgs[10] = self.txtool(txtpath10)
        self.imgs[11] = self.txtool(txtpath11)
        self.imgs[12] = self.txtool(txtpath12)
        self.imgs[13] = self.txtool(txtpath13)
        self.imgs[14] = self.txtool(txtpath14)

    # 从txt中读取得到数据集中图片的地址和标号
    def txtool(self, path):
        imgs = []
        datainfo = open(path, 'r')
        for line in datainfo:
            line = line.strip('\n')
            words = line.split()
            imgs.append((words[0]))
        return imgs

    def __len__(self):
        self.length = len(self.imgs[7])
        return 5000

    # 从地址与标号得到经过变换的图片
    def getpic(self,imgs,transform):
        pic = imgs
        pic = Image.open(pic)
        pic = transform(pic)
        return pic

    def get_randaug(self,s):
        a = random.uniform( 1 - 0.8*s, 1 + 0.8*s )
        b = random.uniform( 1 - 0.8*s, 1 + 0.8*s )
        c = random.uniform( 1 - 0.8*s, 1 + 0.8*s )
        d = random.uniform( -0.5*s, 0.5*s )
        return a,b,c,d

    # 对前64图进行改变
    def simclr_transform0(self,size = 128, s = 1):
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size,scale=(0.75,1)),
                                              transforms.RandomRotation(degrees=(-30, 30)),
                                              transforms.RandomHorizontalFlip(p = 0.3),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])

        return data_transforms

    # 只进行形状变换
    def simclr_transform1(self):
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=128),
                                              # transforms.RandomHorizontalFlip() 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5
                                              transforms.RandomHorizontalFlip(),
                                              # transforms.ToTensor() 将给定图像转为Tensor
                                              transforms.ToTensor()])

        return data_transforms

    # 对后64图进行固定颜色增强
    def simclr_transform2(self, a, b, c, d):
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=128),
                                              # transforms.RandomHorizontalFlip() 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter_new1(a, b, c, d),
                                              # transforms.ToTensor() 将给定图像转为Tensor
                                              transforms.ToTensor()])
        return data_transforms

    def simclr_transform4(self, size=256, s=1):
            # 进行一些列图像处理操作，再用compose函数将全部操作串起来
            # RandomResizedCrop将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小
        data_transforms = transforms.Compose([transforms.ToTensor()])

        return data_transforms

    # 得到标号数据集对应的数据数目 共14个数据集
    def gen_rand(self,name):
        data = {
        '1': lambda: random.randint(0, 37415),
        '2': lambda: random.randint(0, 37415),
        '3': lambda: random.randint(0, 37372),
        '4': lambda: random.randint(0, 37372),
        '5': lambda: random.randint(0, 47840),
        '6': lambda: random.randint(0, 47840),
        '7': lambda: random.randint(0, 46220),
        '8': lambda: random.randint(0, 46220),
        '9': lambda: random.randint(0, 55260),
        '10': lambda: random.randint(0, 101120),
        '11': lambda: random.randint(0, 82900),
        '12': lambda: random.randint(0, 222350),
        '13': lambda: random.randint(0, 223900),
        '14': lambda: random.randint(0, 94710),
        }
        num = data[name]
        num = num()
        return(num)

    def __getitem__(self, index):

        transforms0 = self.simclr_transform0()
        transforms1 = self.simclr_transform1()
        transforms2 = self.simclr_transform2
        transforms4 = self.simclr_transform4()

        pic1 = []
        pic2 = []
        pic = []
        # n表示图片为矩阵中总第几张图片 m作为中间变量，每次变为8，则重新归1
        n = 1
        m = 1
        # 在矩阵中加入前64张图片
        while(n <= 56):
            # 数据集排布为1-8
            name1 = int((n - 1)/8) + 1
            name2 = name1 + 1
            # 分别得到对应数据集路径集合
            list = self.imgs
            img_path_list1 = list[name1]
            img_path_list2 = list[name2]

            # num为图片的索引
            # img1 img2 分别为配对图片组中的第一张、第二张片
            num = self.gen_rand(str(name1))
            path1 = img_path_list1[num]
            img1 = self.getpic(path1, transforms0)
            path2 = img_path_list2[num]
            img2 = self.getpic(path2, transforms0)
            pic1.append(img1)
            pic2.append(img2)

            if(m == 8):
                n += 9
                pic = pic + pic1 + pic2
                #pic = pic.append(pic1)
                pic1 = []
                pic2 = []
                m = 1
            else:
                n += 1
                m += 1

        # 上述循环得到的pic list=64 为前64图
        # 此处得到位置为65-70的图片
        n = 65
        name = 10
        while(n <= 70):
            img_path_list = self.imgs[name]
            img1 = self.getpic(img_path_list[self.gen_rand(str(name))],transforms1)
            img2 = self.getpic(img_path_list[self.gen_rand(str(name))],transforms1)
            pic1 = []
            pic2 = []
            pic1.append(img1)
            pic2.append(img2)
            pic = pic + pic1 + pic2
            name += 1
            n += 2

        n = 71
        # 上述循环得到pic list=70 前70图
        # 以下循环得到71-128图
        while(n <= 128):
            name = random.randint(1,14)
            a, b, c, d = self.get_randaug(1)
            img_path_list = self.imgs[name]
            transforms = transforms2(a,b,c,d)
            img1 = self.getpic(img_path_list[self.gen_rand(str(name))],transforms)
            img2 = self.getpic(img_path_list[self.gen_rand(str(name))],transforms)
            pic1 = []
            pic2 = []
            pic1.append(img1)
            pic2.append(img2)
            pic = pic + pic1 + pic2
            n += 2

        pic = torch.stack(pic,dim = 0)

        return pic
