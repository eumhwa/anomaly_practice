## Multi-scale feature based MemAE trainer
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from byol_pytorch import BYOL

from models.memAE import *
from models.memory_module import *
from models.resnet import *

from config import get_params_AD

# functions -- 
def aggregate_features(feature, n_in=256, n_out=2, n_downsample=3, pooling="max"):
  
  conv1x1_layer = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=1)
  
  if pooling == "max":
    pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
  elif pooling == "avg":
    pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

  down_ft = conv1x1_layer(feature)
  for _ in range(n_downsample):
    down_ft = pool(down_ft)
  
  return down_ft

def load_dataset(args, trans):
    train_dataset = ImageFolder(root=os.path.join(args.data_path, "train"), transform=trans)
    test_dataset = ImageFolder(root=os.path.join(args.data_path, "test"), transform=trans)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader

def train_resnet_by_byol(args, train_loader):

    # define cnn(resnet) and byol
    resnet = resnet50(pretrained=True).to(args.device)
    learner = BYOL(resnet, image_size = args.height, hidden_layer = 'avgpool')
    opt = torch.optim.Adam(learner.parameters(), lr=args.lr)

    # train resnet via BYOL
    print("BYOL training start -- ")
    for epoch in range(args.epoch):
        loss_history=[]
        for data, _ in train_loader:
            images = data.to(args.device)
            loss = learner(images)
            loss_float = loss.detach().cpu().numpy().tolist()
            loss_history.append(loss_float)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average() # update moving average of target encoder
            
        if epoch % 5 == 0:
            print(f"EPOCH: {epoch} / loss: {sum(loss_history)/len(train_loader)}")

    return resnet

def extract_cnn_features(args, resnet, train_loader):
    
    # for inferencing
    resnet.eval()
    with torch.no_grad():
        features = []
        for data, _ in train_loader:
        data = data.to(args.device)
        _, feature_lst = resnet(data)
        features.append(feature_lst)

    for i, f in enumerate(features):
    
        f = [cur_f.to("cpu") for cur_f in f]
        if i == 0:
            l1, l2, l3, l4 = f
        else:
            l1 = torch.cat([l1, f[0]], dim=0)
            l2 = torch.cat([l2, f[1]], dim=0)
            l3 = torch.cat([l3, f[2]], dim=0)
            l4 = torch.cat([l4, f[3]], dim=0)

    pooled_down_l1 = aggregate_features(l1, l1.shape[1], args.min_filter_size, n_downsample=3)
    pooled_down_l2 = aggregate_features(l2, l2.shape[1], args.min_filter_size*2, n_downsample=2)
    pooled_down_l3 = aggregate_features(l3, l3.shape[1], args.min_filter_size*4, n_downsample=1)
    pooled_down_l4 = aggregate_features(l4, l4.shape[1], args.min_filter_size*8, n_downsample=0)

    feature_stack = torch.cat([pooled_down_l1, pooled_down_l2, pooled_down_l3, pooled_down_l4], dim=1)
    feature_loader = DataLoader(feature_stack, batch_size=args.batch_size, shuffle=False)
    print(f"feature input dimension of MemAE: {feature_stack.shape}")

    return feature_loader

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
    resnet = train_resnet_by_byol(args, train_loader=train_loader)
    feature_loader = extract_cnn_features(args, resnet, train_loader)
    
    nf1, nf2, nf3, nf4, nf5 = [args.nf1 * (2**k) for k in range(5)]
    print(f"Filter size list: {[nf1, nf2, nf3, nf4, nf5]}")

    model = MemAE(
        feature_stack.shape[1], args.mem_dim_in, shrink_thres=args.shrink_threshold, 
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
        for batch_idx, data in enumerate(feature_loader):
            data = data.to(args.device)

            recon_res = model(data)
            recon_data = recon_res[0]
            #print(f"data shape: {data.shape} and recon_data shape: {recon_data.shape}")
            
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


