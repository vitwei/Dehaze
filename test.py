import os
# os.environ['CUDA_VISIBLE_DEVICES']='3'
import sys
import argparse
import time
import torch
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import ipdb
import numpy as np
from PIL import Image
import glob
import time
from dehaze import auto_build_net
from torchvision.utils import save_image
# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='PyTorch implementation of CLIP-LIT (liang. 2023)')
parser.add_argument('-i', '--input', help='directory of input folder', default='/home/huangweiyan/workspace/dataset/myhaze/test/')
parser.add_argument('-o', '--output', help='directory of output folder', default='./inference_result/')
#parser.add_argument('-c', '--ckpt', help='test ckpt path', default='./pretrained_models/enhancement_model.pth')

args = parser.parse_args()
device = torch.device('cuda:0')
ck=torch.load('train0/snapshots_train_train0/iter_2080.pth')
U_net=auto_build_net().cuda()
U_net.load_state_dict(ck)
trans = transforms.Compose([
    ToTensor()
])
channel_swap = (1, 2, 0)

filePath = args.input
file_list = os.listdir(filePath)
print(file_list)
for file_name in file_list:
	with torch.no_grad():
		image=filePath+file_name
		LL_images = Image.open(image).convert('RGB')
		img_in = trans(LL_images)
		LL_tensor = img_in.unsqueeze(0).to(device)
		prediction,_,syn = U_net(LL_tensor)
		prediction = prediction.clamp(0, 1)
		output_path = args.output
		outpath=os.path.join(output_path,file_name)
		save_image(prediction, outpath)
		

		

