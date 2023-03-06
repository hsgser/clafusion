import torch
import torchvision
from data_transforms import albumentations_transforms


class DataEngine(object):
    classes = ["%s" % i for i in range(200)]

    def __init__(self, args):
        super(DataEngine, self).__init__()
        self.batch_size = args["batch_size"]
        self.num_workers = args["num_workers"]
        self.train_data_path = args["train_data_path"]
        self.test_data_path = args["test_data_path"]
        self.load()

    def _transforms(self):
        # Data Transformations
        train_transform = albumentations_transforms(p=1.0, is_train=True)
        test_transform = albumentations_transforms(p=1.0, is_train=False)
        return train_transform, test_transform

    def _dataset(self):
        # Get data transforms
        train_transform, test_transform = self._transforms()

        # Dataset and Creating Train/Test Split
        train_set = torchvision.datasets.ImageFolder(root=self.train_data_path, transform=train_transform)
        test_set = torchvision.datasets.ImageFolder(root=self.test_data_path, transform=test_transform)
        return train_set, test_set

    def load(self):
        # Get Train and Test Data
        train_set, test_set = self._dataset()

        # Dataloader Arguments & Test/Train Dataloaders
        dataloader_args = dict(shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

        self.train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
        self.test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)
