from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class Encoder(nn.Module):
    def __init__(self, chnum_in, nf1, nf2, nf3, nf4, nf5):
        super(Encoder,self).__init__()
        self.block1 = nn.Sequential(
                        nn.Conv2d(chnum_in, nf1, 3, padding=1),     
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm2d(nf1),
                        nn.Conv2d(nf1, nf2, 3, padding=1),                           
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm2d(nf2),
                        nn.Conv2d(nf2, nf3, 3, padding=1),                          
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm2d(nf3),
                        nn.MaxPool2d(2, 2)                                    
        )
        self.block2 = nn.Sequential(
                        nn.Conv2d(nf3, nf4, 3, padding=1),                         
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm2d(nf4),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(nf4, nf5, 3, padding=1),                       
                        nn.LeakyReLU(0.2, inplace=True),
        )
        
                
    def forward(self,x):
        out = self.block1(x)
        out = self.block2(out)
        
        return out
        

class Decoder(nn.Module):
    def __init__(self, chnum_in, nf1, nf2, nf3, nf4, nf5):
        super(Decoder,self).__init__()
        self.block1 = nn.Sequential(
                        nn.ConvTranspose2d(nf5, nf4, 3, 1, 1),                 
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm2d(nf4),
                        nn.ConvTranspose2d(nf4, nf3, 3, 1, 1),       
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm2d(nf3)
        )
        self.block2 = nn.Sequential(
                        nn.ConvTranspose2d(nf3, nf2, 3, 2, 1, 1),                     
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm2d(nf2),
                        nn.ConvTranspose2d(nf2, nf1, 3, 1, 1),                     
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.BatchNorm2d(nf1),
                        nn.ConvTranspose2d(nf1, chnum_in, 3, 2, 1, 1),                      
                        nn.LeakyReLU(0.2, inplace=True),
        )
        
    def forward(self,x):
        out = self.block1(x)
        out = self.block2(out)

        return out

class MemAE(nn.Module):
    def __init__(
        self, batch_size, chnum_in, mem_dim, shrink_thres=0.0025,
        nf1=16, nf2=32, nf3=64, nf4=128, nf5=256):
        
        super(MemAE, self).__init__()

        self.encoder = Encoder(chnum_in, nf1, nf2, nf3, nf4, nf5) # Encoder
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=256, shrink_thres=shrink_thres)
        self.decoder = Decoder(chnum_in, nf1, nf2, nf3, nf4, nf5) # Decoder
        
        
    def forward(self, x):
        latent_z = self.encoder(x)
        res_mem = self.mem_rep(latent_z)
        latent_z = res_mem['output']
        att = res_mem['att']
        rec_x = self.decoder(latent_z)
        return rec_x, att

    def encode(self, x):
        latent_z = self.encoder(x)
        return latent_z

    def decode(self, z):
        return self.decoder(z)


class VanillaAE(nn.Module):
    def __init__(
        self, chnum_in,
        nf1=16, nf2=32, nf3=64, nf4=128, nf5=256):
        
        super(VanillaAE, self).__init__()

        self.encoder = Encoder(chnum_in, nf1, nf2, nf3, nf4, nf5) # Encoder
        self.decoder = Decoder(chnum_in, nf1, nf2, nf3, nf4, nf5) # Decoder
        
        
    def forward(self, x):
        latent_z = self.encoder(x)
        rec_x = self.decoder(latent_z)
        return rec_x

    def encode(self, x):
        latent_z = self.encoder(x)
        return latent_z

    def decode(self, z):
        return self.decoder(z)


def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)