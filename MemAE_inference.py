import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from models.memAE import *
from models.memory_module import *

from config import get_params_AD

IMG_SIZE = (384, 384)
data_path = "/app/workspace/nas/mvtec_ad/screw"
nf1, nf2, nf3, nf4, nf5 = [32 * (2**k) for k in range(5)]
thr = 0.1
model_path =  './weight/model.pt'

def load_dataset(trans, data_path):
    train_dataset = ImageFolder(root=os.path.join(data_path, "train"), transform=trans)
    test_dataset = ImageFolder(root=os.path.join(data_path, "test"), transform=trans)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader



trans = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_loader, test_loader = load_dataset(trans=trans, data_path=data_path)

if args.model == "MemAE":
    model = MemAE(
        args.chnum_in, args.mem_dim_in, shrink_thres=args.shrink_threshold, 
        nf1=nf1, nf2=nf2, nf3=nf3, nf4=nf4, nf5=nf5)
elif args.model == "AE":
    model = VanillaAE(chnum_in_, nf1=nf1, nf2=nf2, nf3=nf3, nf4=nf4, nf5=nf5)


ckpt = torch.load(model_path, map_location=args.device)
model.load_state_dict(ckpt)

model = model.to(args.device)
recon_loss_func = nn.MSELoss().to(args.device)
entropy_loss_func = EntropyLossEncap().to(args.device)



model.eval()
with torch.no_grad():
    loss_list = []
    data_list = []
    recon_list = []
    for batch_idx, (data, label) in enumerate(test_loader):
        
        data = data.to(args.device)
        recon_res = model(data)
        recon_data = recon_res[0]
        loss = recon_loss_func(recon_data, data)

        att_w = recon_res[1]
            
        entropy_loss = entropy_loss_func(att_w)
        entropy_loss_val = entropy_loss.item()
        loss = loss + 0.0002 * entropy_loss
        loss_val = loss.detach().cpu().numpy().tolist()
        loss_list.append(loss_val)
        
        data_list.append(data.cpu())
        recon_list.append(recon_data.cpu())


label_list = test_loader.dataset.targets
anomaly_bin = [0 if l <= thr else 1 for l in loss_list] 
label_bin = [0 if y == 0 else 1 for y in label_list]

confusion_matrix(label_bin, anomaly_bin)

