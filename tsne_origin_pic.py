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

n = 10
# warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class use_parameter():
    def __init__(self, path_model, name):
        self.path = path_model
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=False, num_classes=128),
            "resnet50": models.resnet50(pretrained=False, num_classes=128)
        }
        self.backbone = self.resnet_dict[name]
        dim_mlp = 128
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
        model.load_state_dict(state_dict, strict=False)
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


path_model = 'path_model'
name = "resnet50"
getmodel = use_parameter(path_model, name)
model = getmodel.get_model1()
model.eval()

imgs_path = []
relative_path = []
folder_path = '/home/tcz/pycharm/pycharm-2022.1.4/SimCLR-master/SimCLR-12-27-128batch/tsne/3.4test/'
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

trans_totensor = transforms.ToTensor()

feature_list = []
for i in range(50):
    img_path = folder_path + str(i) + '.jpg'
    img = Image.open(img_path).convert('RGB')
    img_transformed = trans_totensor(img)
    img_tensor = torch.unsqueeze(img_transformed, 0)
    img_tensor = img_tensor.to(torch.device('cuda'))
    feature = model(img_tensor)
    feature = feature.detach().cpu()
    feature_list.append(feature.detach().numpy())
    features_list.append(feature_list)
    print(img_path)

features = np.vstack(features_list)

tsne = TSNE(n_components=2, perplexity=n, random_state=0,
            early_exaggeration=12, learning_rate=170)
m = features.shape[0]
out_dim = features.shape[2]

features = features.reshape(m, out_dim)
features_tsne = tsne.fit_transform(features)


color_same_aug = ['red', 'blue', 'green', 'purple', 'yellow', 'orange', 'black', 'pink', 'brown', 'gray',
                  'darkgreen', 'skyblue']

plt.figure()

for i in range(50):

    x, y = features_tsne[i, :]
    c_index = int(i / 10)
    s_index = i / 10
    plt.scatter(x, y, c=color_same_aug[c_index])
    plt.annotate(str(i + 1), (x, y))

plt.savefig('pic.jpg', dpi=300)

plt.show()