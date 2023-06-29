import math

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from einops import rearrange,repeat,reduce
from torch.nn.parameter import Parameter
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
    
def trans_data(x, y, alpha=1.0, natt=1, c_in=2048):
    
    batch_size = x.shape[0]
    
    x_new = torch.zeros(x.shape).cuda()
    y_new = torch.zeros(y.shape).cuda()
    for i in range(natt):
        index = torch.randperm(batch_size).cuda()
        x_new[:,i,:] = x[index][:,i,:]
        y_new[:,i] = y[index][:,i]
    
    lam = np.random.beta(alpha, alpha, natt)
    lam = torch.from_numpy(lam).cuda().float()
    lam = torch.reshape(lam,[1, natt, 1])
    
    lam_n = np.random.beta(alpha, alpha, [batch_size, natt])
    lam_n = torch.from_numpy(lam_n).cuda().float()
    lam_n = torch.reshape(lam_n,[batch_size, natt, 1])
    
    lam_v = np.random.beta(alpha, alpha, [batch_size, natt])
    lam_v = torch.from_numpy(lam_v).cuda().float()
    lam_v = torch.reshape(lam_v,[batch_size, natt, 1])
    
    norm = torch.reshape(torch.norm(x, p=2, dim=2),[batch_size,natt,1])
    vec = nn.functional.normalize(x, p=2, dim=2, eps=1e-12)
    
    norm_new = torch.reshape(torch.norm(x_new, p=2, dim=2),[batch_size,natt,1])
    vec_new = nn.functional.normalize(x_new, p=2, dim=2, eps=1e-12)
    
    vec = vec * lam_v + vec_new * (1 - lam_v)
    norm = norm * lam_n + norm_new * (1 - lam_n)
    x_m = vec * norm
    
    eq_index = y == y_new
    eq_index = repeat(eq_index, 'b a -> b a c', c=c_in)
    
    x_u = lam * x + (1 - lam) * x_new
    mixed_x = torch.where(eq_index, x_m, x_u)
    
    lam = torch.reshape(lam,[1, natt])
    y = lam * y + (1 - lam) * y_new
    
    return mixed_x, y

def initialize_weights(module):
    for m in module.children():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, _BatchNorm):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)

class Classifier(nn.Module):

    def __init__(self, c_in, nattr, bb):
        super(Classifier, self).__init__()
        self.bb = bb
        self.nattr = nattr
        self.c = c_in
        
        if bb == 'resnet50':
            self.separte = nn.Sequential(nn.Linear(c_in, nattr*c_in), nn.BatchNorm1d(nattr*c_in), nn.ReLU(nattr*c_in))
        else:
            self.separte = nn.Sequential(nn.Linear(c_in, nattr*c_in), nn.GELU())
            
        self.logits = nn.Sequential(nn.Linear(c_in, nattr))

    def forward(self, x, label=None, mode='train'):
        
        if self.bb == 'resnet50':
            x = rearrange(x, 'n c h w ->n (h w) c')
            x = reduce(x,'n k c ->n c', reduction = 'mean')
        
        x = self.separte(x)
        
        x = torch.reshape(x,[x.shape[0],self.nattr,self.c])
        if mode == 'train':
            x,label = trans_data(x,label,natt=self.nattr,c_in=self.c)
    
        x = x.sum(1)
        
        logits = self.logits(x)
        
        return logits, label
    
class Network(nn.Module):
    def __init__(self, backbone, classifier):
        super(Network, self).__init__()

        self.backbone = backbone  
        self.classifier = classifier

    def forward(self, x, label=None,mode='train'):
        
        x = self.backbone(x)
        x,label = self.classifier(x,label,mode)
        
        return [x],label
    
    
    
