from math import sqrt
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0,1'
import torch
import torch.nn as nn
from torch.nn import functional as F
# import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim
import argparse
import ipdb
import dataloader_prompt_margin
import dataloader_prompt_add
import dataloader_images as dataloader_sharp 
import copy
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
from test_function import inference
from CLIP.loss import color_loss,TVLoss
import clip_score
import random
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import clip
from dehaze import build_net
import pyiqa
import shutil
import math

torch.backends.cudnn.benchmark = True 
task_name="train1"
writer = SummaryWriter('./'+task_name+"/"+'tensorboard_'+task_name)

dstpath="./"+task_name+"/"+"train_scripts"
if not os.path.exists(dstpath):
    os.makedirs(dstpath)
shutil.copy("train.py",dstpath)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def resize_to_multiple_of(x, m=14, mode="bilinear"):
    # x: (N, C, H, W)
    N, C, H, W = x.shape
    H2 = math.ceil(H / m) * m
    W2 = math.ceil(W / m) * m
    if H2 == H and W2 == W:
        return x
    
    return F.interpolate(x, size=(H2, W2), mode=mode, align_corners=False)

def random_crop(img):
    b,c,h,w=img.shape
    hs=random.randint(0,h-224)
    hw=random.randint(0,w-224)
    return img[:,:,hs:hs+224,hw:hw+224]

def train(config):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    clipmodel, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")#ViT-B/32
    clipmodel.to(device)
    for para in clipmodel.parameters():
        para.requires_grad = False
    model=build_net(clipmodel)
    #ck=torch.load('clip_model/PSNR3426_SSIM9885.pth', map_location='cpu')
    #model.fix.load_state_dict(ck)
    model=model.cuda()    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    Depth=DepthAnythingV2(**model_configs['vits'])
    Depth.load_state_dict(torch.load('clip_model/depth_anything_v2_vits.pth', map_location='cpu'))
    Depth = Depth.to(device).eval()
    for p in Depth.parameters():
        p.requires_grad = False
    
    #load dataset



    if config.load_pretrain_prompt == True:
        ck=torch.load('/home/huangweiyan/workspace/model/CLIP-LIT/train1/snapshots_prompt_train1/best_prompt_round0.pth', map_location='cpu')
        model.prompt.load_state_dict(ck,strict=False)
        torch.save(model.prompt.state_dict(), config.prompt_snapshots_folder + "pretrained_prompt" + '.pth')
        if config.train_syn:
            config.num_clip_pretrained_iters=5000
    else:
        if config.num_clip_pretrained_iters < 3000:
            print("WARNING: For training from scratch, num_clip_pretrained_iters should not lower than 3000 iterations!\nAutomatically reset num_clip_pretrained_iters to 8000 iterations...")
            config.num_clip_pretrained_iters=5000
    train_low_dataset = dataloader_sharp.lowlight_loader(config.lowlight_images_path,config.overlight_images_path)    #dataloader
    train_low_loader = torch.utils.data.DataLoader(train_low_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    train_norm_dataset = dataloader_sharp.PairedImageDataset(config.normallight_images_path,512)    #dataloader
    train_norm_loader = torch.utils.data.DataLoader(train_norm_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    
    prompt_train_dataset = dataloader_prompt_margin.lowlight_loader(config.lowlight_images_path,config.normallight_images_path)#,config.overlight_images_path)        
    prompt_train_loader = torch.utils.data.DataLoader(prompt_train_dataset, batch_size=config.prompt_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    
    prompt_train_dataset_1 = dataloader_prompt_add.lowlight_loader(config.overlight_images_path,config.normallight_images_path)
    prompt_train_loader_1 = torch.utils.data.DataLoader(prompt_train_dataset_1, batch_size=config.prompt_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True,drop_last=True )
    

    L_clip = clip_score.L_clip_from_feature()
    L_clip_MSE = clip_score.L_clip_MSE()
    L_margin_loss = clip_score.four_margin_loss(0.9,0.2)
    TV_loss = TVLoss()
    TV_loss=TV_loss.to(device)
    train_optimizer = torch.optim.Adam(model.parameters(), lr=config.train_lr, weight_decay=config.weight_decay)
    train_prompt_optimizer = torch.optim.Adam(model.prompt.parameters(), lr=config.train_lr, weight_decay=config.weight_decay)
    train_syn_optimizer = torch.optim.Adam(model.syn.parameters(), lr=config.train_lr, weight_decay=config.weight_decay)


    #initial parameters
    model.train()
    total_iteration=0
    cur_iteration=0
    max_score_psnr=-10000
    pr_last_few_iter=0
    score_psnr=[0]*30
    semi_path=['','']
    pr_semi_path=0

    best_model=model
    best_syn=model.syn
    min_prompt_loss=100
    best_prompt_iter=0
    best_model_iter=0
    min_syn_loss=100
    best_syn_iter=0
    rounds=0
    reconstruction_iter=0
    reinit_flag=0
    #Start training!
    for epoch in range(config.num_epochs):
        if total_iteration<config.num_clip_pretrained_iters:
            if config.train_syn==True:
                total_iteration=2000
                train_thre=0
                total_thre=config.num_clip_pretrained_iters
            else:
                train_thre=0
                total_thre=config.num_clip_pretrained_iters
        elif total_iteration<config.num_reconstruction_iters+config.num_clip_pretrained_iters:
            train_thre=config.num_reconstruction_iters
            total_thre=config.num_reconstruction_iters
        elif cur_iteration==0:
            train_thre=5000#800#2100#800#200
            total_thre=6100#2800#3100#1200#500
            print("cur using prompt from: iteration ", best_prompt_iter)
            print("cur using best model from: iteration ", best_model_iter)
            print("cur using best model from: iteration ", best_syn_iter)
        if cur_iteration+1<=train_thre:
            model.prompt.embedding_prompt.requires_grad = False
            for para in model.syn.parameters():
                para.requires_grad = False
            for iteration, item in enumerate(train_norm_loader): 
                img_norm_train,img_norm_train_path=item
                img_norm_train = img_norm_train.cuda()

                img_norm,text_features,syn  = model(img_norm_train)

                final=torch.clamp(img_norm,0,1)
                final_resized = resize_to_multiple_of(final, 14)  # 变成14倍数
                train_resized = resize_to_multiple_of(img_norm_train, 14)
                with torch.no_grad():
                    depthloss=F.l1_loss(Depth(final_resized),Depth(train_resized))
                
                cliploss=16*20*L_clip(final, text_features)
                
                clip_MSEloss = 25*L_clip_MSE(final, img_norm_train,[1.0,1.0,1.0,1.0,1.0])

                color_l=color_loss(final,img_norm_train)

                loss = cliploss + 0.9*clip_MSEloss+depthloss+color_l+TV_loss(final)

                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()
                
                with torch.no_grad():
                    score_psnr[pr_last_few_iter] = -loss
                    pr_last_few_iter+=1
                    if pr_last_few_iter==30:
                        pr_last_few_iter=0
                    if (sum(score_psnr).item()/30.0)>max_score_psnr and ((total_iteration+1) % config.display_iter) == 0:
                        max_score_psnr=sum(score_psnr).item()/30.0
                        torch.save(model.state_dict(), config.train_snapshots_folder + "best_model_round"+str(rounds) + '.pth')    
                        best_model=model
                        best_model_iter=total_iteration+1
                        print(max_score_psnr)
                        model.eval()
                        inference(config.test_path,'./'+task_name+'/result_'+task_name+'/result_jt_'+str(total_iteration+1)+"_psnr_or_-loss"+str(max_score_psnr)[:8]+'/',model)
                        if total_iteration >config.num_reconstruction_iters+config.num_clip_pretrained_iters:
                            semi_path[pr_semi_path]='./'+task_name+'/result_'+task_name+'/result_jt_'+str(total_iteration+1)+"_psnr_or_-loss"+str(max_score_psnr)[:8]+'/'
                            print(semi_path)
                        torch.save(best_model.state_dict(), config.train_snapshots_folder + "iter_" + str(total_iteration+1) + '.pth') 
                if ((total_iteration+1) % config.display_iter) == 0:
                    print("training current learning rate: ",train_optimizer.state_dict()['param_groups'][0]['lr'])
                    print("Loss at iteration", total_iteration+1,"epoch",epoch, ":", loss.item())
                    print("loss_clip",cliploss," reconstruction loss",clip_MSEloss)
                    writer.add_scalars('Loss_train', {'train': loss,"clip": cliploss," reconstruction loss":clip_MSEloss},total_iteration+1)
                    idx = random.randint(0, syn.size(0) - 1)   # 随机选一个样本
                    writer.add_image("syn_Sample", syn[idx], total_iteration+1)
                    print(cur_iteration+1," ",total_iteration+1)
                    print(train_thre,' ',total_thre)
                if cur_iteration+1==train_thre and total_iteration>config.num_reconstruction_iters+config.num_clip_pretrained_iters and (cliploss+0.9*clip_MSEloss>config.thre_train):
                    train_thre+=60
                    total_thre+=60
                elif cur_iteration+1==train_thre:
                    cur_iteration+=1
                    total_iteration+=1
                    print("switch to fine-tune the prompt pair")
                    break
                cur_iteration+=1
                total_iteration+=1
            model.prompt.embedding_prompt.requires_grad =True      
            for para in model.syn.parameters():
                para.requires_grad = True  
        else:
            #prompt initialization
            if total_iteration<config.num_clip_pretrained_iters:
                for iteration, item in enumerate(prompt_train_loader_1):
                    img_lowlight,label,img_lowlight_tensor=item    
                    img_lowlight = img_lowlight.cuda()
                    label = label.cuda()
                    img_lowlight_tensor=img_lowlight_tensor.squeeze(1).cuda()
                    if total_iteration>2000:
                        model.prompt.embedding_prompt.requires_grad =False      

                        _,_,syn=model(img_lowlight_tensor)
                        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                                device=syn.device).view(1, 3, 1, 1)
                        clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                                device=syn.device).view(1, 3, 1, 1)
                        syn_mean = (syn - clip_mean) / clip_std   
                        syn_resized = resize_to_multiple_of(syn, 14)
                        syn_gt_resized = resize_to_multiple_of(img_lowlight_tensor, 14)

                        depthloss=F.l1_loss(Depth(syn_resized),Depth(syn_gt_resized))

                        syn_features = clipmodel.encode_image(syn_mean)
                        syn_features = syn_features / (syn_features.norm(dim=-1, keepdim=True) + 1e-6)
                        syn_output=model.prompt(syn_features,0)

                        syn_cross_entropyloss=F.cross_entropy(syn_output,torch.from_numpy(np.array(0)).cuda())
                        syn_loss=depthloss+syn_cross_entropyloss

                        train_syn_optimizer.zero_grad()
                        syn_loss.backward()
                        train_syn_optimizer.step()
                        loss=syn_loss
                        if ((total_iteration+1) % config.prompt_display_iter) == 0:
                            if loss<min_syn_loss:
                                min_syn_loss=loss
                                best_syn=model.syn
                                best_syn_iter=total_iteration+1
                                torch.save(best_syn.state_dict(), config.prompt_snapshots_folder + "best_syn_round"+str(rounds) + '.pth')
                            print("prompt current learning rate: ",train_syn_optimizer.state_dict()['param_groups'][0]['lr'])
                            print("Loss at iteration", total_iteration+1, ":", syn_loss.item())
                            print("loss",syn_loss)
                            writer.add_scalars('Loss_syn', {'train':syn_loss}, total_iteration)
                            writer.add_scalars('Loss_syn_cross', {'train':syn_cross_entropyloss}, total_iteration)
                            idx = random.randint(0, syn.size(0) - 1) 
                            writer.add_image("syn_Sample_prompt", syn[idx],total_iteration)
                            print(cur_iteration+1," ",total_iteration+1)
                            print(train_thre,' ',total_thre)

                    if total_iteration<2000:
                        model.prompt.embedding_prompt.requires_grad =True 

                        output = model.prompt(img_lowlight, 0) 
                        cross_entropyloss=  F.cross_entropy(output,label)
                        prompt_loss=cross_entropyloss

                        train_prompt_optimizer.zero_grad()
                        prompt_loss.backward()
                        train_prompt_optimizer.step()
                        loss=prompt_loss
                        if ((total_iteration+1) % config.prompt_display_iter) == 0:
                            if loss<min_prompt_loss:
                                min_prompt_loss=loss
                                best_prompt=model.prompt
                                best_prompt_iter=total_iteration+1
                                torch.save(best_prompt.state_dict(), config.prompt_snapshots_folder + "best_prompt_round"+str(rounds) + '.pth')
                            print("prompt current learning rate: ",train_prompt_optimizer.state_dict()['param_groups'][0]['lr'])
                            print("Loss at iteration", total_iteration+1, ":", loss.item())
                            print("output",output.softmax(dim=-1),"label",label)
                            print("cross_entropy_loss",cross_entropyloss)
                            print("loss",loss)
                            writer.add_scalars('Loss_prompt', {'train':loss}, total_iteration)
                            print(cur_iteration+1," ",total_iteration+1)
                            print(train_thre,' ',total_thre)
                            
                    if ((total_iteration+1) % config.prompt_snapshot_iter) == 0:
                        torch.save(model.prompt.state_dict(), config.prompt_snapshots_folder + "iter_" + str(total_iteration+1) + '.pth')  
                    if cur_iteration+1==total_thre and loss>config.thre_prompt:#loss>last_prompt_loss[flag_prompt]*0.95:#loss>0.01:#
                        #train_thre+=20
                        total_thre+=100
                    elif cur_iteration+1==total_thre:
                        cur_iteration+=1
                        total_iteration+=1
                        break
                    cur_iteration+=1
                    total_iteration+=1  
            

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('-b','--lowlight_images_path', type=str, default="/home/huangweiyan/workspace/dataset/myhaze/train/negative/") 
    parser.add_argument('--overlight_images_path', type=str, default="/home/huangweiyan/workspace/dataset/myhaze/train/negative/")
    parser.add_argument('--test_path', type=str, default="/home/huangweiyan/workspace/dataset/myhaze/test/") 
    parser.add_argument('-r','--normallight_images_path', type=str, default='/home/huangweiyan/workspace/dataset/myhaze/train/positive/') 
    parser.add_argument('--length_prompt', type=int, default=16)
    parser.add_argument('--thre_train', type=float, default=90)
    parser.add_argument('--thre_prompt', type=float, default=60)
    parser.add_argument('--reconstruction_train_lr',type=float,default=0.00005)#0.0001
    parser.add_argument('--train_lr', type=float, default=1e-4)#0.00002#0.00005#0.0001
    parser.add_argument('--prompt_lr', type=float, default=0.000005)#0.00001#0.00008
    parser.add_argument('--T_max', type=float, default=100)
    parser.add_argument('--eta_min', type=float, default=5e-6)#1e-6
    parser.add_argument('--weight_decay', type=float, default=0.001)#0.0001
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=2000)#3000
    parser.add_argument('--num_reconstruction_iters', type=int, default=0)#1000
    parser.add_argument('--num_clip_pretrained_iters', type=int, default=0)#8000
    parser.add_argument('--noTV_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--prompt_batch_size', type=int, default=32)#32
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--display_iter', type=int, default=20)
    parser.add_argument('--snapshot_iter', type=int, default=20)
    parser.add_argument('--prompt_display_iter', type=int, default=20)
    parser.add_argument('--prompt_snapshot_iter', type=int, default=100)
    parser.add_argument('--train_snapshots_folder', type=str, default="./"+task_name+"/"+"snapshots_train_"+task_name+"/")
    parser.add_argument('--prompt_snapshots_folder', type=str, default="./"+task_name+"/"+"snapshots_prompt_"+task_name+"/")
    parser.add_argument('--load_pretrain', type=lambda x: (str(x).lower() == 'true'), default= False)
    parser.add_argument('--pretrain_dir', type=str, default= './pretrained_models/init_pretrained_models/init_enhancement_model.pth')
    parser.add_argument('--load_pretrain_prompt', type=lambda x: (str(x).lower() == 'true'), default= True)
    parser.add_argument('--train_syn', type=lambda x: (str(x).lower() == 'true'), default= True)
    parser.add_argument('--prompt_pretrain_dir', type=str, default= './pretrained_models/init_pretrained_models/best_prompt_round0.pth')
    
    config = parser.parse_args()

    if not os.path.exists(config.train_snapshots_folder):
        os.mkdir(config.train_snapshots_folder)
    if not os.path.exists(config.prompt_snapshots_folder):
        os.mkdir(config.prompt_snapshots_folder)
  

    train(config)
