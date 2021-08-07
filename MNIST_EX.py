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

from config import get_params_mnist

def load_mnist(args):
    # MNIST dataset
    mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=True)

    return train_loader, test_loader

if __name__ == "__main__":
    
    parser = get_params_mnist()
    args = parser.parse_args()

    train_loader, test_loader = load_mnist(args)

    # memory augmented AE 
    model = MemAE(args.chnum_in, args.mem_dim_in, shrink_thres=args.shrink_threshold)
    model.apply(weights_init)
    model.to(args.device)

    # loss & optimizer
    recon_loss_func = nn.MSELoss().to(args.device)
    entropy_loss_func = EntropyLossEncap().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training
    for epoch_idx in range(args.epoch):
        for data, _ in train_loader:
            data = data.to(args.device)

            recon_res = model(data)
            recon_data = recon_res[0] # z
            
            att_w = recon_res[1] # att
            loss = recon_loss_func(recon_data, data)
            recon_loss_val = loss.item()
            entropy_loss = entropy_loss_func(att_w)
            entropy_loss_val = entropy_loss.item()
            loss = loss + args.entropy_loss_weight * entropy_loss
            loss_val = loss.item()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch_idx % 5 == 0:
            print(f"epoch: {epoch_idx} , total loss: {loss}")


    model_path =  f"./weight/{args.ckpt_name}" 
    torch.save(model.state_dict(), model_path)