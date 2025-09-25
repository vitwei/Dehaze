import torch
import os
import time

import numpy as np
from PIL import Image
import glob
import time
import torchvision
import random
import ipdb
import torch.nn.functional as F
 
#data_lowlight = data_lowlight.resize((size,size),  Image.Resampling.LANCZOS)

def lowlight(image_path,image_list_path,result_list_path,DCE_net,size=768):
    img = Image.open(image_path).convert("RGB")
    # 转 numpy -> tensor
    img = np.asarray(img) / 255.0
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).cuda()

    B, C, H, W = img.shape
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    data_lowlight = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
    with torch.no_grad():
        enhanced_image,_,_ = DCE_net(data_lowlight)
    enhanced_image=torch.clamp(enhanced_image,0,1)
    enhanced_image = enhanced_image[:, :, :H, :W]
    image_path = image_path.replace(image_list_path,result_list_path)
    image_path = image_path.replace('.JPG','.png')
    output_path = image_path
    if not os.path.exists(output_path.replace('/'+image_path.split("/")[-1],'')):
        os.makedirs(output_path.replace('/'+image_path.split("/")[-1],''))
    torchvision.utils.save_image(enhanced_image, output_path)


def inference(image_list_path,result_list_path,DCE_net,size=256):
    with torch.no_grad():
        filePath = image_list_path
        file_list = os.listdir(filePath)
        print(f"Inferencing...{result_list_path}")
        for file_name in file_list:
            test_list = glob.glob(filePath+file_name)
            for image in test_list:
                lowlight(image,image_list_path,result_list_path,DCE_net,size)