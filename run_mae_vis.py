# -*- coding: utf-8 -*-
# @Time    : 2021/11/18 22:40
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : run_mae_vis.py
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import torch.nn as nn

from PIL import Image

from pathlib import Path

from timm.models import create_model

import utils
from utils import plot_10_1a, plot_10_1b
import modeling_pretrain
from datasets import build_yz_dataset


from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



# RG_GRFf_file_path = '/home/liullhappy/imageNet/rgDatasets/GRFf.pkl'
# RG_GRFl_file_path = '/home/liullhappy/imageNet/rgDatasets/GRFl.pkl'

def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    parser.add_argument('img_path', type=str, help='input image path')
    parser.add_argument('save_path', type=str, help='save image path')
    parser.add_argument('model_path', type=str, help='checkpoint path of model')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters | pretrain_mae_base_patch16_224
    parser.add_argument('--model', default='pretrain_mae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    # Drop Path
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    return parser.parse_args()



def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model


def main(args):
    print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    # 16
    print("Patch size = %s" % str(patch_size))
    # args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    # checkpoint-best.pth
    checkpoint = torch.load(args.model_path, map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    model.load_state_dict(checkpoint['model'], strict=False)
    #
    print ('MODEL:', model)
   
    model.eval()

    grf_yz, bool_masked_pos_yz = build_yz_dataset(args)
    
    loss_func = nn.MSELoss()

    # 
    with torch.no_grad():
        # 1 × 10 × 101
        # grf_yz = grf_yz[None, :]
        # 1 × 101
        # bool_masked_pos_yz = bool_masked_pos_yz[None, :]
        grf_yz = grf_yz.to(device, non_blocking=True)
        #
        bool_masked_pos_yz = bool_masked_pos_yz.to(device, non_blocking=True).flatten(1).to(torch.bool)
        # 1 × 75 × 10
        outputs = model(grf_yz, bool_masked_pos_yz)

        
        #save original img
        # 
        # ① 
        ori_grf_yz = grf_yz.squeeze(0).cpu().numpy()
        # print ('1:', ori_grf_yz.shape)
        # print ('2:', type(ori_grf_yz))
        
        # 
        plt = plot_10_1a(ori_grf_yz)
        plt.savefig(f"{args.save_path}/ori_grf.jpg", dpi = 330)
        plt.cla()
        
        
        # 224 × 224 → (h w)=n → (14 14) → 196; (p1 p2)=p → (16 16) → 256
        # img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0], p2=patch_size[0])
        # 
        # img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        # b n (p c) → 1 196 (16*16*3) → [1 196 768]
        # img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
        # [1 147 768]
        # img_patch[bool_masked_pos] = outputs
        
        #  1 × 101 × 10
        grf_frame = torch.transpose(grf_yz, 1, 2)
        #
        print ('1*:', grf_frame[bool_masked_pos_yz].shape)
        print ('2*:', outputs.squeeze(0).shape)
        print ('3*:', grf_frame[bool_masked_pos_yz])
        print ('4*:', outputs.squeeze(0))
        print ('4**', grf_frame[bool_masked_pos_yz].unsqueeze(0).shape)

        loss = loss_func(input=outputs, target=grf_frame[bool_masked_pos_yz].unsqueeze(0))
        loss_value = loss.item()

        print ('5*', loss_value)
        #
        grf_frame[bool_masked_pos_yz] = outputs.squeeze(0)



        
        # 
        rec_grf_yz = torch.transpose(grf_frame,1,2).squeeze(0).cpu().numpy()
        
        print ('3:', rec_grf_yz.shape)
        # print ('4:', type(rec_grf_yz))
        
        # 
        plt = plot_10_1a(rec_grf_yz)
        plt.savefig(f"{args.save_path}/rec_grf.jpg", dpi = 330)
        plt.cla()
       

        # save random mask img 
        # 
        # img_mask = rec_img * mask #  [1 3 224 224] × [1 3 224 224]

        # 
        # mask_grf_yz = grf_frame * mask #  [1 101 10] × [1 101 10] = 1 × 101 × 10
        
        # img = ToPILImage()(img_mask[0, :])
        
        # mask_grf_yz = torch.transpose(mask_grf_yz,1,2).squeeze(0).cpu().numpy() # 1 × 101 × 10 → 10 × 101
        
        # 
        # 1 × (101-75) × 10
        grf_yz = torch.transpose(grf_yz, 1, 2)
        # print ('11:', grf_yz.shape)
        mask_grf_yz = grf_yz[~bool_masked_pos_yz]
        print ('22:', mask_grf_yz.shape)
        # 10 × 26
        mask_grf_yz = torch.transpose(mask_grf_yz,0,1).cpu().numpy()

        print ('5:', mask_grf_yz.shape)
        print ('6:', type(mask_grf_yz))     
        
        # ③
        plt = plot_10_1b(mask_grf_yz)
        plt.savefig(f"{args.save_path}/mask_grf.jpg", dpi = 330)
        plt.cla()
        plt.close()



if __name__ == '__main__':
    opts = get_args()
    main(opts)
