from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from dataloader_plus import MyDataset_new
from data1to64 import MyDataset1to64
from data64to128 import MyDataset64to128
from data1to128 import MyDataset1to128

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder


    def get_dataset(self, name, n_views):
        path = '/home/tcz/pycharm/pycharm-2022.1.4/SimCLR-master/WSI_datafile_tools/datatxt_whole/'

        txtpath1 = str(path + 's1.txt'),
        txtpath2 = str(path + 's2.txt'),
        txtpath3 = str(path + 's3.txt'),
        txtpath4a = str(path + 's4a.txt'),
        txtpath5b = str(path + 's5b.txt'),
        txtpath4b = str(path + 's4b.txt'),
        txtpath10b = str(path + 's10b.txt'),
        txtpath9b = str(path + 's9b.txt'),

        txtpath5a = str(path + 's5a.txt'),
        txtpath6 = str(path + 's6.txt'),
        txtpath7 = str(path + 's7.txt'),
        txtpath8 = str(path + 's8.txt'),
        txtpath9a = str(path + 's9a.txt'),
        txtpath10a = str(path + 's10a.txt'),


        valid_datasets = {'data1to128': lambda: MyDataset1to128(
                              txtpath1[0],
                              txtpath2[0],
                              txtpath3[0],
                              txtpath4a[0],
                              txtpath5b[0],
                              txtpath4b[0],
                              txtpath10b[0],
                              txtpath9b[0],
                              txtpath5a[0],
                              txtpath6[0],
                              txtpath7[0],
                              txtpath8[0],
                              txtpath9a[0],
                              txtpath10a[0]),

                          'data1to64': lambda: MyDataset1to64(
                              txtpath1[0],
                              txtpath2[0],
                              txtpath3[0],
                              txtpath4a[0],
                              txtpath5b[0],
                              txtpath4b[0],
                              txtpath10b[0],
                              txtpath9b[0]),

                          'data64to128': lambda: MyDataset64to128(
                              txtpath5a[0],
                              txtpath6[0],
                              txtpath7[0],
                              txtpath8[0],
                              txtpath9a[0],
                              txtpath10a[0])
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
