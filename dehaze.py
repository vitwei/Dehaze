from math import gcd
import time
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import ipdb
from collections import OrderedDict


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.LeakyReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class MultiAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.BatchNorm2d(dim)
        # Simple Channel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.LeakyReLU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.LeakyReLU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.LeakyReLU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp(x)
        x = identity + x
        return x


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))
    
class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x
           
class TriScaleConv(nn.Module):
    def __init__(self, in_channels,outchannel,dilation=3, res=True,group=False):
        super(TriScaleConv, self).__init__()
        self.res = res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,dilation=5,padding=5),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,padding=1),
            nn.PReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,dilation=3,padding=3),
            nn.PReLU(),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(outchannel*4,outchannel*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(outchannel*2,outchannel,kernel_size=1),
        )
    def forward(self,x):
        x1 = self.conv1(x) + x
        x2 = self.conv2(x1) + x1
        x3 = self.conv3(x2) + x2
        out = torch.cat([x,x1,x2,x3],dim=1)
        out = self.merge(out)
        return x+out

class DualScaleConv(nn.Module):
    def __init__(self, in_channels,outchannel,dilation=3, res=True,group=False):
        super(DualScaleConv, self).__init__()
        self.res = res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,dilation=4,padding=4),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,padding=1),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(outchannel,outchannel*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(outchannel*2,outchannel,kernel_size=1),
        )
    def forward(self,x):
        out = self.conv1(x) + self.conv2(x)
        out = self.merge(out)
        return x + out if self.res else out
    
class DynamicConv(nn.Module):
    def __init__(self, inchannels, mode='highPass', dilation=0, kernel_size=3, stride=1, kernelNumber=8):
        super(DynamicConv, self).__init__()
        self.stride = stride
        self.mode = mode
        self.kernel_size = kernel_size
        self.kernelNumber = inchannels
        self.conv = nn.Conv2d(inchannels, self.kernelNumber*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(self.kernelNumber*kernel_size**2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.unfoldMask = []
        self.unfoldSize = kernel_size + dilation * (kernel_size - 1)
        self.pad = nn.ReflectionPad2d(self.unfoldSize//2)
        if mode == 'lowPass':
            for i in range(self.unfoldSize):
                for j in range(self.unfoldSize):
                    if (i % (dilation + 1) == 0) and (j % (dilation + 1) == 0):
                        self.unfoldMask.append(i * self.unfoldSize + j)
        elif mode != 'highPass':
            raise ValueError("Invalid mode. Expected 'lowPass' or 'highPass'.")
        
    def forward(self, x):
        copy = x
        filter = self.ap(x)
        filter = self.conv(filter)
        filter = self.bn(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.unfoldSize).reshape(n, self.kernelNumber, c//self.kernelNumber, self.unfoldSize**2, h*w)
        if self.mode == 'lowPass':
            x = x[:,:,:,self.unfoldMask,:]
        n,c1,p,q = filter.shape
        filter = filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
        filter = self.act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out, copy - out

class localFusionBlock(nn.Module):  
    def __init__(self, in_channels):  
        super(localFusionBlock, self).__init__() 
        self.conv1 = nn.Conv2d(in_channels*3,in_channels*6,kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels*6,in_channels,kernel_size=1)
    def forward(self,x,a,b):
        out = self.conv1(torch.cat([x,a,b],dim=1))
        out = self.act(out)
        out = self.conv2(out)
        return x + out
    
class localFusion(nn.Module):  
    def __init__(self, in_channels):  
        super(localFusion, self).__init__() 
        self.conv1 = nn.ModuleList([
            CCAM(in_channels) for i in range(8)
        ])

    def forward(self,x0,x1,x2,x3,x4,x5,x6,x7):
        (x0,x1,x2,x3,x4,x5,x6,x7) =  (self.conv1[0](x0,x1,x2)+x0,self.conv1[1](x1,x0,x2)+x1,
                                      self.conv1[2](x2,x1,x3)+x2,self.conv1[3](x3,x2,x4)+x3,
                                      self.conv1[4](x4,x3,x5)+x4,self.conv1[5](x5,x4,x6)+x5,
                                      self.conv1[6](x6,x5,x7)+x6,self.conv1[7](x7,x6,x5)+x7
                                      )

        out = torch.cat([x0,x1,x2,x3,x4,x5,x6,x7],dim=1)
        return out

class CCAM(nn.Module):
    def __init__(self,inchannels):
        super(CCAM, self).__init__()
        self.inchannels=inchannels
        self.fc = nn.Linear(inchannels*3, inchannels*3*inchannels)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y, z):
        m = torch.cat((x, y, z), dim=1)  # n, 12, h, w
        gap = F.adaptive_avg_pool2d(m, (1, 1))  # n, 12, 1, 1
        gap = gap.view(m.size(0), self.inchannels*3)  # n, 12
        fc_out = self.fc(gap)  # n, 48
        conv1d_input = fc_out.unsqueeze(1)  # n, 1, 48
        conv1d_out = self.conv1d(conv1d_input)  # n, 1, 48
        conv1d_out = conv1d_out.view(m.size(0), self.inchannels*3, self.inchannels)  # n, 12, 4
        softmax_out = self.softmax(conv1d_out)  # n, 12, 4
        out = torch.einsum('nchw,ncm->nmhw', (m, softmax_out))  # n, 4, h, w
        
        return out

class DFS(nn.Module):
    def __init__(self, in_channels,outchannel,basechannel, mergeNum,res = 1,attn=False):
        super(DFS, self).__init__()
        self.mergeNum = mergeNum
        self.low_pass_filter = DynamicConv(in_channels,mode='highPass')
        self.enlarger = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels,outchannel,kernel_size=3,padding=1)
        )
        self.fe = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels,basechannel//8,kernel_size=1),
            TriScaleConv(basechannel//8,basechannel//8,res=True),
            TriScaleConv(basechannel//8,basechannel//8,res=True),
        )
    def forward(self,x):
        low,high = self.low_pass_filter(x)
        low = self.fe(low)
        high = self.enlarger(high)
        return high,low

class DFLSBlock(nn.Module):  
    def __init__(self, in_channels,split=3):  
        super(DFLSBlock, self).__init__()  

        self.split = split
        self.frequency_enlarge = DualScaleConv(in_channels,in_channels,res=True) 

        self.blocks = nn.ModuleList([
            DFS(in_channels,in_channels*7//8,in_channels,0,res=1),
            DFS(in_channels*7//8,in_channels*6//8,in_channels,in_channels*2//8,res=1,attn=True),
            DFS(in_channels*6//8,in_channels*5//8,in_channels,in_channels*3//8,res=1),
            DFS(in_channels*5//8,in_channels*4//8,in_channels,in_channels*3//8,res=1),
            DFS(in_channels*4//8,in_channels*3//8,in_channels,in_channels*3//8,res=1),
            DFS(in_channels*3//8,in_channels*2//8,in_channels,in_channels*3//8,res=1),
            DFS(in_channels*2//8,in_channels*1//8,in_channels,in_channels*3//8,res=1),
        ])

        self.local = localFusion(in_channels//8)

        self.synthesizer = nn.Sequential(
            MultiAttn(in_channels),
        )
        self.merger = nn.Sequential(
            nn.Conv2d(in_channels,in_channels*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels*2,in_channels,kernel_size=1),
        )
        self.merger2 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels*2,in_channels,kernel_size=1),
        )
          
    def forward(self, m):  
        m0 = self.frequency_enlarge(m) 

        m1,x0 = self.blocks[0](m0)
        m1,x1 = self.blocks[1](m1)
        m1,x2 = self.blocks[2](m1)
        m1,x3 = self.blocks[3](m1)
        m1,x4 = self.blocks[4](m1)
        m1,x5 = self.blocks[5](m1)
        x7,x6 = self.blocks[6](m1)

        m2 = self.local(x0,x1,x2,x3,x4,x5,x6,x7)
        m2 = self.merger(m2)
        m2 = m2 + m0
        out = self.synthesizer(m2)
        out = self.merger2(out)
        out = out + m2
        return out

class DFLSNet(nn.Module):
    def __init__(self):
        super(DFLSNet, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            DFLSBlock(base_channel),
            DFLSBlock(base_channel*2),
            DFLSBlock(base_channel*4),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DFLSBlock(base_channel * 4),
            DFLSBlock(base_channel * 2),
            DFLSBlock(base_channel)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])


        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        # 256
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        # 128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        # 64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z = self.Decoder[0](z)
        # 128
        z = self.feat_extract[3](z)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        # 256
        z = self.feat_extract[4](z)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)

        return z
    

import math
class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_ad, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class Conv2d_rd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_rd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        if math.fabs(self.theta - 0.0) < 1e-8:
            out_normal = self.conv(x)
            return out_normal 
        else:
            conv_weight = self.conv.weight
            conv_shape = conv_weight.shape
            if conv_weight.is_cuda:
                conv_weight_rd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5).fill_(0)
            else:
                conv_weight_rd = torch.zeros(conv_shape[0], conv_shape[1], 5 * 5)
            conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
            conv_weight_rd[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_weight[:, :, 1:]
            conv_weight_rd[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -conv_weight[:, :, 1:] * self.theta
            conv_weight_rd[:, :, 12] = conv_weight[:, :, 0] * (1 - self.theta)
            conv_weight_rd = conv_weight_rd.view(conv_shape[0], conv_shape[1], 5, 5)
            out_diff = nn.functional.conv2d(input=x, weight=conv_weight_rd, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)

            return out_diff


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_hd, self).__init__() 
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_hd)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):

        super(Conv2d_vd, self).__init__() 
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_vd)
        return conv_weight_vd, self.conv.bias


class DEConv(nn.Module):
    def __init__(self, dim):
        super(DEConv, self).__init__() 
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        w = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5
        res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)

        return res

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect' ,bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn

from einops.layers.torch import Rearrange

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn
    
class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result
    
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
    
class DEABlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, reduction=8):
        super(DEABlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        cattn = self.ca(res)
        sattn = self.sa(res)
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)
        res = res * pattn2
        res = res + x
        return res


class DEBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DEBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        res = res + x
        return res

class DEANet(nn.Module):
    def __init__(self, base_dim=32):
        super(DEANet, self).__init__()
        # down-sample
        self.down1 = nn.Sequential(nn.Conv2d(3, base_dim, kernel_size=3, stride = 1, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        # level1
        self.down_level1_block1 = DEBlock(default_conv, base_dim, 3)
        self.down_level1_block2 = DEBlock(default_conv, base_dim, 3)
        self.down_level1_block3 = DEBlock(default_conv, base_dim, 3)
        self.down_level1_block4 = DEBlock(default_conv, base_dim, 3)
        self.up_level1_block1 = DEBlock(default_conv, base_dim, 3)
        self.up_level1_block2 = DEBlock(default_conv, base_dim, 3)
        self.up_level1_block3 = DEBlock(default_conv, base_dim, 3)
        self.up_level1_block4 = DEBlock(default_conv, base_dim, 3)
        # level2
        self.fe_level_2 = nn.Conv2d(in_channels=base_dim * 2, out_channels=base_dim * 2, kernel_size=3, stride=1, padding=1)
        self.down_level2_block1 = DEBlock(default_conv, base_dim * 2, 3)
        self.down_level2_block2 = DEBlock(default_conv, base_dim * 2, 3)
        self.down_level2_block3 = DEBlock(default_conv, base_dim * 2, 3)
        self.down_level2_block4 = DEBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block1 = DEBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block2 = DEBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block3 = DEBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block4 = DEBlock(default_conv, base_dim * 2, 3)
        # level3
        self.fe_level_3 = nn.Conv2d(in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=3, stride=1, padding=1)
        self.level3_block1 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block2 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block3 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block4 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block5 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block6 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block7 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block8 = DEABlock(default_conv, base_dim * 4, 3)
        # up-sample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1))
        # feature fusion
        self.mix1 = CGAFusion(base_dim * 4, reduction=8)
        self.mix2 = CGAFusion(base_dim * 2, reduction=4)

    def forward(self, x):
        x_down1 = self.down1(x)
        x_down1 = self.down_level1_block1(x_down1)
        x_down1 = self.down_level1_block2(x_down1)
        x_down1 = self.down_level1_block3(x_down1)
        x_down1 = self.down_level1_block4(x_down1)

        x_down2 = self.down2(x_down1)
        x_down2_init = self.fe_level_2(x_down2)
        x_down2_init = self.down_level2_block1(x_down2_init)
        x_down2_init = self.down_level2_block2(x_down2_init)
        x_down2_init = self.down_level2_block3(x_down2_init)
        x_down2_init = self.down_level2_block4(x_down2_init)

        x_down3 = self.down3(x_down2_init)
        x_down3_init = self.fe_level_3(x_down3)
        x1 = self.level3_block1(x_down3_init)
        x2 = self.level3_block2(x1)
        x3 = self.level3_block3(x2)
        x4 = self.level3_block4(x3)
        x5 = self.level3_block5(x4)
        x6 = self.level3_block6(x5)
        x7 = self.level3_block7(x6)
        x8 = self.level3_block8(x7)
        x_level3_mix = self.mix1(x_down3, x8)

        x_up1 = self.up1(x_level3_mix)
        x_up1 = self.up_level2_block1(x_up1)
        x_up1 = self.up_level2_block2(x_up1)
        x_up1 = self.up_level2_block3(x_up1)
        x_up1 = self.up_level2_block4(x_up1)

        x_level2_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_level2_mix)
        x_up2 = self.up_level1_block1(x_up2)
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)
        out = self.up3(x_up2)

        return out



import clip
import numpy as np
import scipy.io as sio

class TextEncoder_withvgg(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

        self.synvgg=UNet_emb_oneBranch_symmetry()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.35)

        self.layer1 = nn.Sequential(self.fc1,
                                    self.elu,
                                    self.dropout,
                                    self.fc2,
                                    self.elu,
                                    self.dropout,
                                    self.fc3
                                    )
        self.U=torch.tensor(sio.loadmat("./clip_model/u_matrix.mat")['u_matrix'],dtype=torch.float32).cuda()
    def extract_y(self,i):
        
        y = None
        if isinstance(i, np.ndarray):
            if len(i.shape) == 4:
                y = 0.299 * i[:, :, :, 2] + 0.587 * i[:, :, :, 1] + 0.114 * i[:, :, :, 0]
            elif len(i.shape) == 3:
                y = 0.299 * i[:, :, 2] + 0.587 * i[:, :, 1] + 0.114 * i[:, :, 0]

        elif isinstance(i, torch.Tensor):
            if i.ndim == 4:
                y = 0.299 * i[:, 2, :, :] + 0.587 * i[:, 1, :, :] + 0.114 * i[:, 0, :, :]
            elif i.ndim == 3:
                y = 0.299 * i[2, :, :] + 0.587 * i[1, :, :] + 0.114 * i[0, :, :]

        return y
    def new_Y_gen(self,y_tensor, tf):
        y_tensor = y_tensor * 255.0
        bs, _, h, w = y_tensor.shape
        y0 = torch.floor(y_tensor)
        y1 = (y0 + 1).clamp(max=255.0)
        alpha = (y_tensor - y0)
        y0 = y0.long().view(bs, 1, -1)
        y1 = y0.long().view(bs, 1, -1)
        tf = tf.unsqueeze(1)
        tf0 = torch.take_along_dim(tf, y0, dim=2)           # [B,1,HW]
        tf1 = torch.take_along_dim(tf, y1, dim=2)           # [B,1,HW]
        new_y = (1.0 - alpha.view(bs,1,-1)) * tf0 + alpha.view(bs,1,-1) * tf1
        
        new_y = new_y.view(bs, 1, h, w)

        return new_y / 255.0
    
    def apply_tf(self,rgb,C):
        tf = F.softplus(self.U @ C).T * 255.0
        y = self.extract_y(rgb).unsqueeze(1)
        new_y = self.new_Y_gen(y, tf)
        scale_Y = new_y / (y + 1e-6)  
        y_enhanced = rgb * scale_Y

        return torch.clamp(y_enhanced,0,1), tf


    def apply_ccm(self,y_enhanced,ccm_pred):

        a, b, c, d, e, f = ccm_pred[:, 0], ccm_pred[:, 1], ccm_pred[:, 2], ccm_pred[:, 3], ccm_pred[:, 4], ccm_pred[:, 5]
        ccm = torch.stack([
            torch.stack([1 - a - b, a, b], dim=-1),
            torch.stack([c, 1 - c - d, d], dim=-1),
            torch.stack([e, f, 1 - e - f], dim=-1)
        ], dim=1)

        # a = ccm.transpose(1, 2)
        y_enhanced_reshape = y_enhanced.permute(0, 2, 3, 1).reshape(y_enhanced.shape[0], -1, 3)
        ccm=ccm.expand(y_enhanced_reshape.shape[0], -1, -1)
        pred = torch.bmm(y_enhanced_reshape, ccm.transpose(1, 2))
        pred = pred.reshape(y_enhanced.shape[0], y_enhanced.shape[2], y_enhanced.shape[3], 3).permute(0, 3, 1, 2) 

        return pred


    def forward(self, prompts, tokenized_prompts,syn_base):

        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] 
        ccm = self.layer1(x)
        x = x @ self.text_projection
        syn_Tf,_=self.apply_tf(syn_base,ccm[:1].T)
        return x,self.synvgg(syn_Tf)
    

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype




    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] 
        x = x @ self.text_projection

        return x

class Prompts(nn.Module):
    def __init__(self,model,initials=None):
        super(Prompts,self).__init__()
        print("The initial prompts are:",initials)
        self.text_encoder = TextEncoder(model)
        if isinstance(initials,list):
            text = clip.tokenize(initials).cuda()
            # print(text)
            self.embedding_prompt = nn.Parameter(model.token_embedding(text).requires_grad_()).cuda()
        elif isinstance(initials,str):
            prompt_path=initials

            state_dict = torch.load(prompt_path)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.embedding_prompt=nn.Parameter(new_state_dict['embedding_prompt']).cuda()
            self.embedding_prompt.requires_grad = True
        else:
            self.embedding_prompt=torch.nn.init.xavier_normal_(nn.Parameter(model.token_embedding([" ".join(["X"]*16)," ".join(["X"]*16)]).requires_grad_())).cuda()

    def forward(self,tensor,flag=1):
        tokenized_prompts= torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*16)]])
        text_features= self.text_encoder(self.embedding_prompt,tokenized_prompts)
        
        for i in range(tensor.shape[0]):
            image_features=tensor[i]
            nor=torch.norm(text_features,dim=-1, keepdim=True)
            if flag==0:
                similarity = (100.0 * image_features @ (text_features/nor).T)#.softmax(dim=-1)
                if(i==0):
                    probs=similarity
                else:
                    probs=torch.cat([probs,similarity],dim=0)
            else:
                similarity = (100.0 * image_features @ (text_features/nor).T).softmax(dim=-1)#/nor
                if(i==0):
                    probs=similarity[:,0]
                else:
                    probs=torch.cat([probs,similarity[:,0]],dim=0)
        return probs

class model_total(nn.Module):
    def __init__(self,clip):
        super().__init__()
        self.length_prompt=16
        self.prompt = Prompts(clip,[" ".join(["X"]*(self.length_prompt))," ".join(["X"]*(self.length_prompt))]).cuda()
        self.syn = TextEncoder_withvgg(clip).cuda()
        self.fix = DEANet()
        
        

    


    
    def forward(self,good):
        embedding_prompt=self.prompt.embedding_prompt.cuda()
        tokenized_prompts= torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*self.length_prompt)]]).cuda()
        
        text_features,syn = self.syn(embedding_prompt, tokenized_prompts,good)
        syn_pred = self.fix(syn) 
        
        
        return syn_pred,text_features,syn




def build_net(clipmodel):
    model=model_total(clipmodel)
    return model

def auto_build_net():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clipmodel, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")#ViT-B/32
    clipmodel.to(device)
    for para in clipmodel.parameters():
        para.requires_grad = False
    model=model_total(clipmodel)
    return model



pi = 3.141592653589793

class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1],0.2)) # k is reciprocal to the paper mentioned
        self.gated = False
        self.gated2= False
        self.alpha = 1.0
        self.alpha_s = 1.3
        self.this_k = 0
        #self._gamma_v = nn.Parameter(torch.tensor(0.0))   # 实际 gamma = softplus + 0.5，>0
        #self._gain_v  = nn.Parameter(torch.tensor(0.0))   # 实际 gain_v = 0.5 + sigmoid * 1.5 ∈ (0.5, 2.0)
        #self._bias_v  = nn.Parameter(torch.tensor(0.0))   # 实际 bias_v = tanh * 0.1        ∈ (-0.1, 0.1)
#
        #self._gain_s  = nn.Parameter(torch.tensor(0.0))   # 实际 gain_s = sigmoid * 1.5      ∈ (0, 1.5)
        #self._bias_s  = nn.Parameter(torch.tensor(0.0))   # 实际 bias_s = tanh * 0.05        ∈ (-0.05, 0.05)
    def _adapt_vs(self, value, saturation):
        # value,saturation: [B,1,H,W] 或 [B,H,W] 都行，保持广播一致性
        # 约束到稳定范围
        gamma_v = F.softplus(self._gamma_v) + 0.5           # >0，且不会太小
        gain_v  = 0.5 + torch.sigmoid(self._gain_v) * 1.5   # (0.5, 2.0)
        bias_v  = torch.tanh(self._bias_v) * 0.1            # (-0.1, 0.1)

        gain_s  = torch.sigmoid(self._gain_s) * 1.5         # (0, 1.5)
        bias_s  = torch.tanh(self._bias_s) * 0.05           # (-0.05, 0.05)

        # 单调 tone curve + 轻仿射：v' = clamp( gain * v^gamma + bias )
        v_adj = torch.clamp(gain_v * torch.clamp(value, 0, 1).pow(gamma_v) + bias_v, 0.0, 1.0)
        # 饱和度线性微调（再 clamp）
        s_adj = torch.clamp(gain_s * saturation + bias_s, 0.0, 1.0)

        return v_adj, s_adj
    
    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:,2]==value] = 4.0 + ( (img[:,0]-img[:,1]) / (value - img_min + eps)) [img[:,2]==value]
        hue[img[:,1]==value] = 2.0 + ( (img[:,2]-img[:,0]) / (value - img_min + eps)) [img[:,1]==value]
        hue[img[:,0]==value] = (0.0 + ((img[:,1]-img[:,2]) / (value - img_min + eps)) [img[:,0]==value]) % 6

        hue[img.min(1)[0]==value] = 0.0
        hue = hue/6.0

        saturation = (value - img_min ) / (value + eps )
        saturation[value==0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        #ipdb.set_trace()
        #v_adj, s_adj = self._adapt_vs(value, saturation)

        k = self.density_k
        self.this_k = k.item()
        
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value
        #H = color_sensitive * s_adj * ch
        #V = color_sensitive * s_adj * cv
        #I = v_adj
        xyz = torch.cat([H, V, I],dim=1)
        return xyz
    
    def PHVIT(self, img):
        eps = 1e-8
        H,V,I = img[:,0,:,:],img[:,1,:,:],img[:,2,:,:]
        
        # clip
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        I = torch.clamp(I,0,1)
        
        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        h = torch.atan2(V + eps,H + eps) / (2*pi)
        h = h%1
        s = torch.sqrt(H**2 + V**2 + eps)
        
        if self.gated:
            s = s * self.alpha_s
        
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
        
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))
        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
                
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb

class UNet_emb_oneBranch_symmetry_noreflect(nn.Module):

    def __init__(self, in_channels=3, out_channels=3,bias=False):
        super(UNet_emb_oneBranch_symmetry_noreflect, self).__init__()

        self.cond1 = nn.Conv2d(in_channels,32,3,1,1,bias=True) 
        self.cond_add1 = nn.Conv2d(32,out_channels,3,1,1,bias=True)           

        self.condx = nn.Conv2d(32,64,3,1,1,bias=True) 
        self.condy = nn.Conv2d(64,32,3,1,1,bias=True) 

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.ResidualBlock1=ResidualBlock(32,32)
        self.ResidualBlock2=ResidualBlock(32,32)
        self.ResidualBlock3=ResidualBlock(64,64)
        self.ResidualBlock4=ResidualBlock(64,64)
        self.ResidualBlock5=ResidualBlock(32,32)
        self.ResidualBlock6=ResidualBlock(32,32)

        self.PPM1 = PPM1(32,8,bins=(1,2,3,6))


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight.data)
                m.weight.data.normal_(0.0, 0.02)
                #nn.init.zeros_(m.bias.data)


    def forward(self, x):

        light_conv1=self.lrelu(self.cond1(x))
        res1=self.ResidualBlock1(light_conv1)
        
        res2=self.ResidualBlock2(res1)
        res2=self.PPM1(res2)
        res2=self.condx(res2)
        
        res3=self.ResidualBlock3(res2)
        res4=self.ResidualBlock4(res3)

        res4=self.condy(res4)
        res5=self.ResidualBlock5(res4)
        
        res6=self.ResidualBlock6(res5)
        
        light_map=self.relu(self.cond_add1(res6))
 
        return light_map

class UNet_emb_oneBranch_symmetry(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=3,bias=False):
        super(UNet_emb_oneBranch_symmetry, self).__init__()

        self.cond1 = nn.Conv2d(in_channels,32,3,1,1,bias=True,padding_mode='reflect') 
        self.cond_add1 = nn.Conv2d(32,out_channels,3,1,1,bias=True,padding_mode='reflect')           

        self.condx = nn.Conv2d(32,64,3,1,1,bias=True,padding_mode='reflect') 
        self.condy = nn.Conv2d(64,32,3,1,1,bias=True,padding_mode='reflect') 

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.ResidualBlock1=ResidualBlock(32,32)
        self.ResidualBlock2=ResidualBlock(32,32)
        self.ResidualBlock3=ResidualBlock(64,64)
        self.ResidualBlock4=ResidualBlock(64,64)
        self.ResidualBlock5=ResidualBlock(32,32)
        self.ResidualBlock6=ResidualBlock(32,32)

        self.PPM1 = PPM1(32,8,bins=(1,2,3,6))
        self.trans = RGB_HVI()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight.data)
                m.weight.data.normal_(0.0, 0.02)
                #nn.init.zeros_(m.bias.data)

    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi
    
    def forward(self, x):
        hvi = self.trans.HVIT(x)
        light_conv1=self.lrelu(self.cond1(hvi))
        res1=self.ResidualBlock1(light_conv1)
        
        res2=self.ResidualBlock2(res1)
        res2=self.PPM1(res2)
        res2=self.condx(res2)
        
        res3=self.ResidualBlock3(res2)
        res4=self.ResidualBlock4(res3)
        res4=self.condy(res4)
        
        res5=self.ResidualBlock5(res4)
        res6=self.ResidualBlock6(res5)

        light_map=self.relu(self.cond_add1(res6))
        output_rgb = self.trans.PHVIT(light_map+hvi)

        return output_rgb

class PPM1(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM1, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.PReLU()
            ))
        self.features = nn.ModuleList(self.features)
        self.fuse = nn.Sequential(
                nn.Conv2d(in_dim+reduction_dim*4, in_dim, kernel_size=3, padding=1, bias=False),
                nn.PReLU())

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out_feat = self.fuse(torch.cat(out, 1))
        return out_feat       
          
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.downsample = downsample
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.lrelu(out)
        return out

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False,padding_mode='reflect')