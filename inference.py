import os, time
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from models.memAE import *
from models.memory_module import *

from config import get_inference_params
from utils import *

def get_acc(ys, yhats):
    res = []
    for y, yhat in zip(ys, yhats):
        if y == yhat:
            res.append(1)
        else:
            res.append(0)
    
    return 100*sum(res)/len(res)

if __name__ == "__main__":
    
    parser = get_inference_params()
    args = parser.parse_args()
    print(args)

    # loading datasets
    IMG_SIZE = (args.width, args.height)
    trans = get_transform(IMG_SIZE)
    test_loader = load_MVTecAD_dataset(
        args.dataset_path, is_train=False, batch_size=args.batch_size, transform=trans)
    
    # loading CKPT
    nf1, nf2, nf3, nf4, nf5 = [args.nf1 * (2**k) for k in range(5)]
    print(f"Filter size list: {[nf1, nf2, nf3, nf4, nf5]}")

    if args.model == "MemAE":
        model = MemAE(
            args.chnum_in, args.mem_dim_in, shrink_thres=args.shrink_threshold, 
            nf1=nf1, nf2=nf2, nf3=nf3, nf4=nf4, nf5=nf5)
    elif args.model == "AE":
        model = VanillaAE(chnum_in_, nf1=nf1, nf2=nf2, nf3=nf3, nf4=nf4, nf5=nf5)


    ckpt = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(ckpt)
    print("Model loaded successfully ---")

    model = model.to(args.device)
    recon_loss_func = nn.MSELoss().to(args.device)
    entropy_loss_func = EntropyLossEncap().to(args.device)

    tic = time.time()
    print("Strat Inferencing --- ")
    model.eval()
    with torch.no_grad():
        loss_list = []
        data_list = []
        recon_list = []
        for data, _ in test_loader:
            
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

    toc = time.time()
    print("End inference --- ")
    print(f"Testing time elapsed: {abs(tic-toc)}")
    
    label_list = test_loader.dataset.targets
    anomaly_bin = [0 if l <= args.anomaly_threshold else 1 for l in loss_list] 
    label_bin = [0 if y == 0 else 1 for y in label_list]

    tb = confusion_matrix(label_bin, anomaly_bin)
    acc = get_acc(label_bin, anomaly_bin)
    
    print(f"Test accuracy: {acc}%")
    print("Confusion matrix")
    print(tb)
    

    if args.viz:
        print("Saving heatmap images --- ")
