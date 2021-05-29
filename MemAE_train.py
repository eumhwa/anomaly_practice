## Multi-scale feature based MemAE trainer
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from models.memAE import *
from models.memory_module import *

data_path = "./datasets/screw"
IMG_SIZE = (384, 384)
BATCH_SIZE = 4
EPOCHS = 50
device = "cuda"

trans = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

train_dataset = ImageFolder(root=os.path.join(data_path, "train"), transform=trans)
test_dataset = ImageFolder(root=os.path.join(data_path, "test"), transform=trans)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


