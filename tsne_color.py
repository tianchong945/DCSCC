import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from models.resnet_simclr import ResNetSimCLR_128
from models.resnet_simclr import ResNetSimCLR_color
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import os
import random
import warnings
import openpyxl
import torchvision

# n = input('每张图片经过的数据增强种类数目 = ')
n = 10
# warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# transform
def transform(a, b, c, d):
    data_transforms = transforms.Compose([
        transforms.ColorJitter_new1(a, b, c, d),
        transforms.ToTensor()
    ])
    return data_transforms


class use_parameter():
    def __init__(self, path_model, name):
        self.path = path_model
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=False, num_classes=128),
            "resnet50": models.resnet50(pretrained=False, num_classes=128)
        }
        self.backbone = self.resnet_dict[name]
        dim_mlp = 128  # self.backbone.fc.in_features
        self.linear = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, 128)
        )
        self.color = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, 64)
        )

    def get_model_backbone(self):
        model = self.backbone
        with open(self.path, 'rb') as f:
            state_dict = torch.load(f)['state_dict']
        # model = nn.DataParallel(model)
        model.load_state_dict(state_dict, strict=False)
        model1 = model.cuda()
        return model1

    def get_model1(self):
        model = ResNetSimCLR_128(base_model='resnet50', out_dim=128)
        with open(self.path, 'rb') as f:
            state_dict = torch.load(f)['state_dict']
        # model = nn.DataParallel(model)
        model.load_state_dict(state_dict, strict=False)
        model1 = model.cuda()
        return model1

    def get_model_color(self):
        model = ResNetSimCLR_color(base_model='resnet50', out_dim=128)
        with open(self.path, 'rb') as f:
            state_dict = torch.load(f)['state_dict']
        # model = nn.DataParallel(model)
        model.load_state_dict(state_dict, strict=False)
        model_color = model.cuda()
        return model_color


path_model = '/home/tcz/pycharm/pycharm-2022.1.4/SimCLR-master/SimCLR-12-27-128batch/runs' \
             '/Feb09_19-14-26_ubt-142/checkpoint_0100.pth.tar'
name = "resnet50"
getmodel = use_parameter(path_model, name)
model = getmodel.get_model_color()
model.eval()

# Load images from the folder and generate n sets of data augmentation.
imgs_path = []
relative_path = []
folder_path = '/home/tcz/pycharm/pycharm-2022.1.4/SimCLR-master/SimCLR-12-27-128batch/tsne/3.3test'
for file_name in os.listdir(folder_path):
    if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
        file_path = os.path.join(folder_path, file_name)
        imgs_path.append(file_path)
        relative_path.append(file_name)

n = int(n)
features_list = []

# Create a new Excel workbook
wb = openpyxl.Workbook()
ws = wb.active
ws.append(["x", "y"])

# Read n rows of the abcd array, so that the data augmentation amplitude is the same each time
number_list = []
# open txt
with open("data.txt", "r") as f:
    # 遍历每一行
    for line in f:
        item = line.strip().split(",")
        number_list.append([float(x) for x in item])

for i, img_path in enumerate(imgs_path):
    img = Image.open(img_path).convert('RGB')
    transform_list = []
    for j in range(n):
        a, b, c, d = number_list[j]
        transform_list.append(transform(a, b, c, d))

    feature_list = []
    for transform_func in transform_list:
        img_transformed = transform_func(img)
        img_tensor = torch.unsqueeze(img_transformed, 0)
        img_tensor = img_tensor.to(torch.device('cuda'))
        feature = model(img_tensor)
        feature = feature.detach().cpu()
        feature_list.append(feature.detach().numpy())
    features_list.append(feature_list)

# Concatenate the features into a two-dimensional array
features = np.vstack(features_list)

# t-SNE
tsne = TSNE(n_components=2, perplexity=n, random_state=0,
            early_exaggeration=70, learning_rate=170)
m = features.shape[0]
out_dim = features.shape[2]

features = features.reshape(m, out_dim)
features_tsne = tsne.fit_transform(features)



# Generate a new color array, for each image with the same color enhancement

color_same_aug = ['red', 'blue', 'green', 'purple', 'yellow', 'orange', 'black', 'pink', 'brown', 'gray',
                  'darkgreen', 'skyblue']

plt.figure()

for i in range(50):

    x, y = features_tsne[i, :]

    c_index = i % 10
    s_index = i % 10

    # Draw a scatter plot using the corresponding colors and shapes, and set the edge color to black
    plt.scatter(x, y, c=color_same_aug[c_index])
    plt.annotate(str(i + 1), (x, y))

plt.savefig('/home/tcz/pycharm/pycharm-2022.1.4/SimCLR-master'
            '/SimCLR-12-27-128batch/tsne/tsne-1.jpg', dpi=300)

plt.show()
