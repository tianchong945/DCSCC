import torchvision
import torch



# transform = transforms.ToTensor()
transform = None
root = r'E:\BaiduNetdiskDownload\StyleRobust Contrastive\duibi_dataset'
# torchvision.datasets.ImageFolder
train_data = torchvision.datasets.ImageFolder(root, transform=transform)
train_iter = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0)

test_data = torchvision.datasets.ImageFolder(root, transform=transform)
test_iter = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=True, num_workers=0)
