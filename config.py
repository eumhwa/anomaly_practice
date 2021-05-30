import argparse


def get_params_mnist():
    parser = argparse.ArgumentParser(description='config parameters for memAE_MNIST.py')
    
    parser.add_argument('--epoch', type=int, default=100, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='training mini batch size')
    parser.add_argument('--lr', type=float, default=0.00003, help='training learning rate')
    parser.add_argument('--chnum_in', type=int, default=1, help='size of training input channel')
    parser.add_argument('--mem_dim_in', type=int, default=512, help='dimension of memory module')
    parser.add_argument('--shrink_threshold', type=float, default=0.0025, help='hard shrinkage parameter for memory modeule')
    parser.add_argument('--entropy_loss_weight', type=float, default=0.0002, help='loss parameter')
    parser.add_argument('--device', type=str, default="cuda:0", help='training device')
    parser.add_argument('--ckpt_name', type=str, default="memAE_MNIST.pt", help='checkpoint name')

    return parser

def get_params_AD():
    parser = argparse.ArgumentParser(description='config parameters for anomaly detection')

    parser.add_argument('--data_path', type=str, default="./datasets/screw", help='training dataset path')
    parser.add_argument('--width', type=int, default=384, help='image width')
    parser.add_argument('--height', type=int, default=384, help='image height')
    parser.add_argument('--epoch', type=int, default=100, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='training mini batch size')
    parser.add_argument('--lr', type=float, default=0.00003, help='training learning rate')
    parser.add_argument('--chnum_in', type=int, default=3, help='size of training input channel')
    parser.add_argument('--mem_dim_in', type=int, default=512, help='dimension of memory module')
    parser.add_argument('--shrink_threshold', type=float, default=0.0025, help='hard shrinkage parameter for memory modeule')
    parser.add_argument('--entropy_loss_weight', type=float, default=0.0002, help='loss parameter')
    parser.add_argument('--device', type=str, default="cuda:0", help='training device')
    parser.add_argument('--ckpt_name', type=str, default="memAE_MNIST.pt", help='checkpoint name')
    parser.add_argument('--nf1', type=int, default=32, help='number of first filters')
    parser.add_argument('--min_filter_size', type=int, default=2, help='number of desired size of filters at first residual block')
    parser.add_argument('--model', type=str, default="MemAE", choices=['MemAE', 'AE'], help='architecture of Autoencoder')

    return parser
