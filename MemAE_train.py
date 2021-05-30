## Multi-scale feature based MemAE trainer
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from models.memAE import *
from models.memory_module import *

from config import get_params_AD

data_path = "./datasets/screw"
IMG_SIZE = (384, 384)
BATCH_SIZE = 4
EPOCHS = 50
device = "cuda"

def load_dataset(args, trans):
    train_dataset = ImageFolder(root=os.path.join(args.data_path, "train"), transform=trans)
    test_dataset = ImageFolder(root=os.path.join(args.data_path, "test"), transform=trans)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__": 

    parser = get_params_AD()
    args = parser.parse_args()

    IMG_SIZE = (args.width, args.height)

    trans = transforms.Compose(
            [
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    # training resnet using BYOL and extracting multi-scale features
    train_loader, test_loader = load_dataset(args, trans=trans)
    
    nf1, nf2, nf3, nf4, nf5 = [args.nf1 * (2**k) for k in range(5)]
    print(f"Filter size list: {[nf1, nf2, nf3, nf4, nf5]}")

    model = MemAE(
        args.chnum_in, args.mem_dim_in, shrink_thres=args.shrink_threshold, 
        nf1=nf1, nf2=nf2, nf3=nf3, nf4=nf4, nf5=nf5)
    #model = VanillaAE(chnum_in_, nf1=32, nf2=64, nf3=128, nf4=256, nf5=512)

    model.apply(weights_init)
    model.to(args.device)

    recon_loss_func = nn.MSELoss().to(args.device)
    entropy_loss_func = EntropyLossEncap().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    print("Training start --")
    for epoch_idx in range(args.epoch):
        for batch_idx, data in enumerate(train_loader):
            data = data.to(args.device)

            recon_res = model(data)
            recon_data = recon_res[0]

            att_w = recon_res[1]
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



    

