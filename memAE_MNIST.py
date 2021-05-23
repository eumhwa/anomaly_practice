import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import math
import numpy as np

from models.memAE import *
from models.memory_module import *

# train parameters
epoch = 100
batch_size = 8
lr = 0.0003

# additional parameters
chnum_in_ = 1
mem_dim_in = 2048
shrink_threshold = 0.0025
entropy_loss_weight = 0.0002

device = torch.device("cuda")
ckpt_nm = "memAE_MNIST.pt"

if __name__ == "__main__":
    
    # MNIST dataset
    mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,num_workers=1,drop_last=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False,num_workers=1,drop_last=True)

    # memory augmented AE 
    model = MemAE(batch_size, chnum_in_, mem_dim_in, shrink_thres=shrink_threshold)
    model.apply(weights_init)
    model.to(device)

    # loss & optimizer
    recon_loss_func = nn.MSELoss().to(device)
    entropy_loss_func = EntropyLossEncap().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # training
    for epoch_idx in range(0, epoch):
        for data, _ in train_loader:
            data = data.to(device)

            recon_res = model(data)
            recon_data = recon_res[0] # z
            
            att_w = recon_res[1] # att
            loss = recon_loss_func(recon_data, data)
            recon_loss_val = loss.item()
            entropy_loss = entropy_loss_func(att_w)
            entropy_loss_val = entropy_loss.item()
            loss = loss + entropy_loss_weight * entropy_loss
            loss_val = loss.item()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch_idx % 5 == 0:
            print(f"epoch: {epoch_idx} , total loss: {loss}")


    model_path =  f"./weight/{ckpt_nm}" 
    torch.save(model.state_dict(), model_path)