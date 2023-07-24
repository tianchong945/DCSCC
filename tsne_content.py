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
from data_aug.gaussian_blur import GaussianBlur


n = 100
path_model = 'path_model'

# warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def transform(a, b, c, d):
    size = 128
    data_transforms = transforms.Compose([
        transforms.ColorJitter_new1(a, b, c, d),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * size)),
        transforms.RandomResizedCrop(size=size, scale=(0.7, 1)),
        transforms.RandomRotation(degrees=(-30, 30)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor()
    ])
    return data_transforms

def get_randaug(s=0.4):
    a = random.uniform(1 - 0.8 * s, 1 + 0.8 * s)
    b = random.uniform(1 - 0.8 * s, 1 + 0.8 * s)
    c = random.uniform(1 - 0.8 * s, 1 + 0.8 * s)
    d = random.uniform(0, 0.5 * s )
    return a, b, c, d


class use_parameter():
    def __init__(self, path_model, name):
        self.path = path_model
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=False, num_classes=128),
            "resnet50": models.resnet50(pretrained=False, num_classes=128)
        }
        self.backbone = self.resnet_dict[name]
        dim_mlp = self.backbone.fc.in_features
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

        model.load_state_dict(state_dict)#, strict=False)
        model1 = model.cuda()
        return model1

    def get_model1(self):
        model = ResNetSimCLR_128(base_model='resnet50', out_dim=128)
        with open(self.path, 'rb') as f:
            state_dict = torch.load(f)['state_dict']

        model.load_state_dict(state_dict, strict=False)
        model1 = model.cuda()
        return model1


    def get_model_color(self):
        model = ResNetSimCLR_color(base_model='resnet50', out_dim=128)
        with open(self.path, 'rb') as f:
            state_dict = torch.load(f)['state_dict']

        model.load_state_dict(state_dict, strict=False)
        model_color = model.cuda()
        return model_color



name = "resnet50"
getmodel = use_parameter(path_model, name)
model = getmodel.get_model1()
model.eval()


imgs_path = []
relative_path = []
folder_path = 'folder_path'
for file_name in os.listdir(folder_path):
    if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
        file_path = os.path.join(folder_path, file_name)
        imgs_path.append(file_path)
        relative_path.append(file_name)

n = int(n)
features_list = []


wb = openpyxl.Workbook()
ws = wb.active
ws.append(["x", "y"])

for i, img_path in enumerate(imgs_path):
    img = Image.open(img_path).convert('RGB')
    transform_list = []
    for j in range(n):
        a, b, c, d = get_randaug(0.4)
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



features = np.vstack(features_list)


tsne = TSNE(n_components=2, perplexity=n, random_state=0)
m = features.shape[0]
features = features.reshape(m,128)
features_tsne = tsne.fit_transform(features)



label_colors = {
    '0.jpg': 'mediumorchid',
    '1.jpg': 'skyblue',
    '2.jpg': 'red',
    '3.jpg': 'orange',
    '4.jpg': 'limegreen',
    '5.jpg': 'fuchsia',
    '6.jpg': 'dodgerblue',
    '7.jpg': 'gold',
    '8.jpg': 'purple',
    '9.jpg': 'darkgreen',
    '10.jpg': 'coral',
    '11.jpg': 'teal',
    '12.jpg': 'deeppink',
    '13.jpg': 'royalblue',
    '14.jpg': 'yellow',
}


fig, ax = plt.subplots()
for i in range(len(imgs_path)):
    for j in range(n):
        label = relative_path[i]
        if label in label_colors:
            color = label_colors[label]
        else:
            color = np.random.rand(3,)
            label_colors[label] = color
        x, y = features_tsne[i*n+j, :]
        ws.append([x,y])
        ax.scatter(x, y, color=color, alpha=0.4, s=40, edgecolor='none')#, label=label if j==0 else "")

ax.axis('off')
wb.save("./my_excel_file.xlsx")
#ax.set_title('1 tsne')
ax.legend()
plt.savefig('./tsne-our.jpg', dpi=300)
plt.show()