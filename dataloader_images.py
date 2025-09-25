import os
import sys

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import glob
import random
import cv2
import clip
import torch.functional as F
random.seed(1143)

def transform_matrix_offset_center(matrix, x, y):
	"""Return transform matrix offset center.

	Parameters
	----------
	matrix : numpy array
		Transform matrix
	x, y : int
		Size of image.

	Examples
	--------
	- See ``rotation``, ``shear``, ``zoom``.
	"""
	o_x = float(x) / 2 + 0.5
	o_y = float(y) / 2 + 0.5
	offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
	reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
	transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
	return transform_matrix 

def img_rotate(img, angle, center=None, scale=1.0):
	"""Rotate image.
	Args:
		img (ndarray): Image to be rotated.
		angle (float): Rotation angle in degrees. Positive values mean
			counter-clockwise rotation.
		center (tuple[int]): Rotation center. If the center is None,
			initialize it as the center of the image. Default: None.
		scale (float): Isotropic scale factor. Default: 1.0.
	"""
	(h, w) = img.shape[:2]

	if center is None:
		center = (w // 2, h // 2)

	matrix = cv2.getRotationMatrix2D(center, angle, scale)
	rotated_img = cv2.warpAffine(img, matrix, (w, h),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT, borderValue=(0,0,0),)
	return rotated_img

def zoom(x, zx, zy, row_axis=0, col_axis=1):
	zoom_matrix = np.array([[zx, 0, 0],
							[0, zy, 0],
							[0, 0, 1]])
	h, w = x.shape[row_axis], x.shape[col_axis]

	matrix = transform_matrix_offset_center(zoom_matrix, h, w) 
	x = cv2.warpAffine(x, matrix[:2, :], (w, h),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT, borderValue=(0,0,0),)
	return x

def augmentation(img1,img2):
	hflip=random.random() < 0.5
	vflip=random.random() < 0.5
	rot90=random.random() < 0.5
	rot=random.random() <0.3
	zo=random.random()<0.3
	angle=random.random()*180-90
	if hflip:
		img1=cv2.flip(img1,1)
		img2=cv2.flip(img2,1)
	if vflip:
		img1=cv2.flip(img1,0)
		img2=cv2.flip(img2,0)
	if rot90:
		img1 = img1.transpose(1, 0, 2)
		img2 = img2.transpose(1,0,2)
	if zo:
		zoom_range=(0.7, 1.3)
		zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
		img1=zoom(img1, zx, zy)
		img2=zoom(img2,zx,zy)
	if rot:
		img1=img_rotate(img1,angle)
		img2=img_rotate(img2,angle)
	return img1,img2

def preprocess_aug(img1,img2):
	img1 = np.uint8((np.asarray(img1)))
	img2 = np.uint8((np.asarray(img2)))
	img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
	img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
	img1,img2=augmentation(img1,img2)
	img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
	img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
	return img1,img2

device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
#load clip

model, preprocess = clip.load("ViT-B/32", device=device, download_root="./clip_model/")#ViT-B/32
for para in model.parameters():
	para.requires_grad = False

def populate_train_list(lowlight_images_path,overlight_images_path=None):
	if overlight_images_path!=None:
		image_list_lowlight = glob.glob(lowlight_images_path + "*")
		image_list_overlight = glob.glob(overlight_images_path + "*")
		image_list_lowlight += image_list_overlight
	else:
		image_list_lowlight = glob.glob(lowlight_images_path + "*")

	train_list = sorted(image_list_lowlight)
	#print(train_list)
	random.shuffle(train_list)

	return train_list
def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    
    return img
class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor

augment=Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

class PairedImageDataset(data.Dataset):
    def __init__(self, data_dir,patchsize=None,aug=True,training=True):
        self.data_dir = data_dir
        self.input_dir=self.data_dir
        # 获取 input 和 target 目录下的文件名
        self.low_images = sorted([
            f for f in os.listdir(self.input_dir)
            if os.path.isfile(os.path.join(self.input_dir, f)) and f.lower().endswith(('.jpg', '.png'))
        ])
    
        self.dir_size = len(self.low_images) 
        self.aug=aug
        self.crop_size=patchsize
        self.train=training

    def __len__(self):
        # 返回数据集的大小.target_dir
        return self.dir_size

    def __getitem__(self, idx):
        # 根据索引获取文件名
        try:
            low_image_path = os.path.join(self.input_dir, self.low_images[idx])
            ps=self.crop_size
            # 打开 input 和 target 图像
            noisy  = torch.from_numpy(np.float32(load_img(low_image_path)))

            noisy = noisy.permute(2,0,1)

            if self.train and self.crop_size:
                ps = self.crop_size
                H = noisy.shape[1]
                W = noisy.shape[2]

                if H < ps or W < ps:
                    pad_h = max(ps - H, 0)
                    pad_w = max(ps - W, 0)
                    pad = [0, pad_w, 0, pad_h]  # 左右上下
                    noisy = F.pad(noisy.unsqueeze(0), pad, mode='reflect').squeeze(0)

                if H-ps==0:
                    r=0
                    c=0
                else:
                    r = np.random.randint(0, H - ps)
                    c = np.random.randint(0, W - ps)

                noisy = noisy[:, r:r + ps, c:c + ps]
                if self.aug==True:
                    apply_trans = transforms_aug[random.getrandbits(3)]
                    noisy = getattr(augment, apply_trans)(noisy)   
            return noisy, os.path.basename(low_image_path)
        except Exception as e:
            print(f"Error loading sample at index {low_image_path}")
            raise None  





class lowlight_loader(data.Dataset):

	def __init__(self, lowlight_images_path,overlight_images_path=None):

		self.train_list = populate_train_list(lowlight_images_path,overlight_images_path) 
		self.size = 256

		self.data_list = self.train_list
		print("Total training examples (Backlit):", len(self.train_list))


	def __getitem__(self, index):

		data_lowlight_path = self.data_list[index]
		data_lowlight = Image.open(data_lowlight_path)

		if("result" not in data_lowlight_path):
			data_lowlight = data_lowlight.resize((self.size,self.size), Image.Resampling.LANCZOS)
		data_lowlight,_=preprocess_aug(data_lowlight,data_lowlight)
		
		data_lowlight = (np.asarray(data_lowlight)/255.0) 
		data_lowlight_output = torch.from_numpy(data_lowlight).float().permute(2,0,1)
		
		return data_lowlight_output,data_lowlight_path

	def __len__(self):
		return len(self.data_list)

