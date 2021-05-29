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

data_path = "./datasets/screw"
IMG_SIZE = (384, 384)
BATCH_SIZE = 4

min_filter_size = 2 # feature aggregation parameter


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


resnet = resnet50(pretrained=True).to(device)
learner = BYOL(resnet, image_size = IMG_SIZE[0], hidden_layer = 'avgpool')
opt = torch.optim.Adam(learner.parameters(), lr=3e-5)

# train resnet via BYOL
for epoch in range(EPOCHS):
    loss_history=[]
    for data, _ in train_loader:
        images = data.to(device)
        loss = learner(images)
        loss_float = loss.detach().cpu().numpy().tolist()
        loss_history.append(loss_float)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
        
    if epoch % 5 == 0:
        print(f"EPOCH: {epoch} / loss: {sum(loss_history)/len(train_loader)}")


resnet.eval()
with torch.no_grad():
    features = []
    for data, _ in train_loader:
      data = data.to(device)
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

pooled_down_l1 = aggregate_features(l1, l1.shape[1], min_filter_size, n_downsample=4)
pooled_down_l2 = aggregate_features(l2, l2.shape[1], min_filter_size*2, n_downsample=3)
pooled_down_l3 = aggregate_features(l3, l3.shape[1], min_filter_size*4, n_downsample=2)
pooled_down_l4 = aggregate_features(l4, l4.shape[1], min_filter_size*8, n_downsample=1)

feature_stack = torch.cat([pooled_down_l1, pooled_down_l2, pooled_down_l3, pooled_down_l4], dim=1)
feature_loader = DataLoader(feature_stack, batch_size=BATCH_SIZE, shuffle=False)
print(f"feature input dimension of MemAE: {feature_stack.shape}")


chnum_in_ = feature_stack.shape[1]
mem_dim_in = 512

model = MemAE(chnum_in_, mem_dim_in, shrink_thres=0.0025, nf1=32, nf2=64, nf3=128, nf4=256, nf5=512)
#model = VanillaAE(chnum_in_, nf1=32, nf2=64, nf3=128, nf4=256, nf5=512)

model.apply(weights_init)
model.to(device)

recon_loss_func = nn.MSELoss().to(device)
entropy_loss_func = EntropyLossEncap().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

model.train()
entropy_loss_weight = 0.0002

for epoch_idx in range(80):
    for batch_idx, data in enumerate(feature_loader):
        data = data.to(device)

        recon_res = model(data)
        recon_data = recon_res[0]
        print(f"data shape: {data.shape} and recon_data shape: {recon_data.shape}")
        
        att_w = recon_res[1]
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


