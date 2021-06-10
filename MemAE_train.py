## Multi-scale feature based MemAE trainer
import os, time
import torch
import torch.nn as nn

from models.memAE import *
from models.memory_module import *

from config import get_train_params
from utils import *


if __name__ == "__main__": 

    parser = get_train_params()
    args = parser.parse_args()
    print(args)

    # loading datasets
    IMG_SIZE = (args.width, args.height)
    trans = get_transform(IMG_SIZE)
    train_loader = load_MVTecAD_dataset(
        args.dataset_path, is_train=True, batch_size=args.batch_size, transform=trans)
    
    # model settings -- 
    nf1, nf2, nf3, nf4, nf5 = [args.nf1 * (2**k) for k in range(5)]
    print(f"Filter size list: {[nf1, nf2, nf3, nf4, nf5]}")

    if args.model == "MemAE":
        model = MemAE(
            args.chnum_in, args.mem_dim_in, shrink_thres=args.shrink_threshold, 
            nf1=nf1, nf2=nf2, nf3=nf3, nf4=nf4, nf5=nf5)
    elif args.model == "AE":
        model = VanillaAE(chnum_in_, nf1=nf1, nf2=nf2, nf3=nf3, nf4=nf4, nf5=nf5)

    model.apply(weights_init)
    model.to(args.device)

    # training options
    recon_loss_func = nn.MSELoss().to(args.device)
    entropy_loss_func = EntropyLossEncap().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    tic = time.time()
    print("Training start --")
    for epoch_idx in range(args.epoch):
        for batch_idx, (data, _) in enumerate(train_loader):
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

    toc = time.itme()
    print(f"End Training ---")
    print(f"Training time elapsed: {abs(tic-toc)}")

    if args.save_ckpt:
        torch.save(model.state_dict(), args.model_path)
        print("Model saved at: ", args.model_path)
    
    

