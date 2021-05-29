import argparse

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

def get_params_mnist():
    parser = argparse.ArgumentParser(description='config parameters for memAE_MNIST.py')
    parser.add_argument('--epoch', type=int, default=100, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='training mini batch size')
    parser.add_argument('--lr', type=float, default=0.00003, help='training learning rate')
    parser.add_argument('--chnum_in', type=int, default=1, help='size of training input channel')
    parser.add_argument('--mem_dim_in', type=int, default=512, help='dimension of memory module')
    parser.add_argument('--shrink_threshold', type=float, default=0.0025, help='hard shrinkage parameter for memory modeule')
    parser.add_argument('--entropy_loss_weight', type=float, default=0.0002, help='loss parameter')
    parser.add_argument('--device', type=str, default="cpu", help='training device')
    parser.add_argument('--ckpt_name', type=str, default="memAE_MNIST.pt", help='checkpoint name')

    return parser