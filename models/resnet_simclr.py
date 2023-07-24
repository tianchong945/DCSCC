import torch.nn as nn
import torchvision.models as models

#from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=True, num_classes=1000)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp))
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(dim_mlp, 128))
        self.color = nn.Sequential(nn.ReLU(), nn.Linear(dim_mlp, 64))

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        out = self.backbone(x)
        out_1 = out.view(x.size(0), -1)
        out1 = self.linear(out_1)
        out_ = out.view(x.size(0), -1)
        classsify = self.color(out_)
        return  out1,classsify

class ResNetSimCLR_128(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR_128, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}
        # ResNetSimCLR("resnet18",128)
        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp))
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(dim_mlp, 128))
        #self.linear = nn.Linear(128,128)
        #self.color = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 64))

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        out = self.backbone(x)
        out_1 = out.view(x.size(0), -1)
        out1 = self.linear(out_1)
        return  out1

class ResNetSimCLR_color(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR_color, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}
        # ResNetSimCLR("resnet18",128)
        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp))
        #self.linear = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 128))
        #self.linear = nn.Linear(128,128)
        self.color = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 64))

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        out = self.backbone(x)
        out_ = out.view(x.size(0), -1)
        classsify = self.color(out_)
        return  classsify