import numpy as np

# 设置随机数种子
np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""
    #  用一张图片的两个随机裁剪作为query和key
    # base_transform是一段函数 将一张图片进行随机数据增强
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views
    # __call__将类作为函数调用
    def __call__(self, x):
        # 返回两次  即对一张图片进行两次base_transform  并返回
        return [self.base_transform(x) for i in range(self.n_views)]

# 输入一张图片返回两张图片______已经证实 返回两张经过变换的图片
# 如 返回img2={list:2} 包含两列矩阵 此处函数，假如transform=ContrastiveLearningViewGenerator（···）
# 输入为img，即：a = transform（img） 得到的为list：2的tensor 大小为{3,长，宽}
# 调整思路1 做两个数据集 sdpc和mrxs 每次调用transform函数，则返回两张图片的tensor
# 调整思路2 构建读取数据的函数————每次返回两个tensor数据

# img1 = ContrastiveLearningViewGenerator(img)