# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import torch

from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from imblearn.over_sampling import SMOTE

from timm.data import create_transform

from masking_generator import RandomMaskingGenerator, RandomMaskingGenerator1d
from dataset_folder import ImageFolder
import pickle
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import maxmin1


# class DataAugmentationForMAE(object):
    # def __init__(self, args):
        # imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        # mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        # self.transform = transforms.Compose([
        #     transforms.RandomResizedCrop(args.input_size),
        #     0-1        
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=torch.tensor(mean),
        #         std=torch.tensor(std))
        # ])

  # 
    # def __call__(self, image):

          # self.masked_position_generator = RandomMaskingGenerator(
          #       args.window_size, args.mask_ratio
          #   )
    #     return self.transform(image), self.masked_position_generator()

    #
    # def __repr__(self):
    #     repr = "(DataAugmentationForBEiT,\n"
    #     repr += "  transform = %s,\n" % str(self.transform)
    #     repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
    #     repr += ")"
    #     return repr


#
def DataAugmentationForMAE_1d(RG_GRFf_file_path, RG_GRFl_file_path):
    #
    val_percent = 0.3
    # mask_ratio = 0.1 # ablation
    # mask_ratio = 0.2 # ablation
    # mask_ratio = 0.3 # ablation
    # mask_ratio = 0.4 # ablation
    # mask_ratio = 0.5 # ablation
    # mask_ratio = 0.6 # ablation
    # mask_ratio = 0.75 # default ablation
    # mask_ratio = 0.8 # ablation
    # mask_ratio = 0.9 # ablation

    print ('mask_ratio:', mask_ratio)
    with open(RG_GRFf_file_path, 'rb') as file:
        GRFf = pickle.load(file).astype(np.float32)
    file.close()
    
    # with open(RG_GRFl_file_path, 'rb') as file:
    #     GRFl = pickle.load(file).astype(np.float32)
    # file.close()
    #
    # GRFl = torch.from_numpy(GRFl)

    #
    # GRFl = torch.from_numpy(GRFl)

    GRFf = maxmin1(GRFf)
    sample_size = GRFf.shape[0]
    L_size = GRFf.shape[2]
    mask_samples_T = RandomMaskingGenerator1d(sample_size, L_size, mask_ratio).astype(np.float32)

    print ('8***:', GRFf.dtype)
    print ('8****:', mask_samples_T.dtype)
    

    GRFf = torch.from_numpy(GRFf)
    GRfm = torch.from_numpy(mask_samples_T)
    

    # print ('GRFf:', GRFf.shape)
    # print ('GRfm:', GRfm.shape)
    

    GRFfdataset = TensorDataset(GRFf, GRfm)
    # n_val = int(len(GRFfdataset) * val_percent)
    # n_train = len(GRFfdataset) - n_val
    # train_set, val_set = random_split(GRFfdataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    n_train = GRFfdataset
    #           
    # window_size = 14 × 14
    # mask_ratio = 0.75, 147
    return n_train


def DataAugmentationForMAE_yz(RG_GRFf_file_path, RG_GRFl_file_path):
    #
    # val_percent = 0.3
    mask_ratio = 0.75
    #        
    with open(RG_GRFf_file_path, 'rb') as file:
        GRFf = pickle.load(file).astype(np.float32)
    file.close()
    #
    GRFf = maxmin1(GRFf)
    sample_size = GRFf.shape[0]
    L_size = GRFf.shape[2]
    #
    mask_samples_T = RandomMaskingGenerator1d(sample_size, L_size, mask_ratio).astype(np.float32)

    #
    yz_indices = np.random.choice(GRFf.shape[0], size = 1, replace=False)

    # 1 × 10 × 101
    grf_yz = GRFf[yz_indices]
    # 1 × 101
    bool_masked_pos_yz = mask_samples_T[yz_indices]
    
    
    grf_yz = torch.from_numpy(grf_yz)
    # mask
    bool_masked_pos_yz = torch.from_numpy(bool_masked_pos_yz)
    

    print ('grf_yz:', grf_yz.shape)
    print ('bool_masked_pos_yz:', bool_masked_pos_yz.shape)

    return grf_yz, bool_masked_pos_yz
  
    
def build_pretraining_dataset(args):
    RG_GRFf_file_path = '/home/liullhappy/imageNet/rgDatasets/GRFf.pkl'
    RG_GRFl_file_path = '/home/liullhappy/imageNet/rgDatasets/GRFl.pkl'

    n_train = DataAugmentationForMAE_1d(RG_GRFf_file_path, RG_GRFl_file_path)
    
    # transform = DataAugmentationForMAE(args)
    # print("Data Aug = %s" % str(transform))
    # 
    # return ImageFolder(args.data_path, transform=transform)
    return n_train


def build_yz_dataset(args):
    RG_GRFf_file_path = '/home/liullhappy/imageNet/rgDatasets/GRFf.pkl'
    RG_GRFl_file_path = '/home/liullhappy/imageNet/rgDatasets/GRFl.pkl'
    #
    ##
    ###
    grf_yz, bool_masked_pos_yz = DataAugmentationForMAE_yz(RG_GRFf_file_path, RG_GRFl_file_path)

    return grf_yz, bool_masked_pos_yz


def partition(RG_GRFf_file_path, RG_GRFl_file_path, abn_ratio):

    with open(RG_GRFf_file_path, 'rb') as file:
        GRFf = pickle.load(file).astype(np.float32)
    file.close()

    with open(RG_GRFl_file_path, 'rb') as file:
        GRFl = pickle.load(file).astype(np.float32)
    file.close()
    
    # stat indice for 0(abnormal)/1(healthy)
    GRFl_0_indice = np.where(GRFl==0)
    GRFl_1_indice = np.where(GRFl==1)
    
    # print ('6:', GRFl_0_indice)
    # print ('7:', GRFl_1_indice)
    
    # select the corresponding features for indice
    GRFl_features_0 = GRFf[list(GRFl_0_indice[0])]
    GRFl_features_1 = GRFf[list(GRFl_1_indice[0])]
    
    # select the corresponding features for indice
    GRFl_0 = GRFl[list(GRFl_0_indice[0])]
    GRFl_1 = GRFl[list(GRFl_1_indice[0])]

    # print (GRFl_features_0.shape)
    
    # print (GRFl_features_1.shape)
    
    # 67977 abnoraml / 16574 healthy,
    # print (round(len(GRFl_0)/len(GRFl_1),3))
    
    # print (GRFl_0.shape)
    
    # print (GRFl_1.shape)

    # 
    total_sample_size = GRFl_features_0.shape[0]
    total_sample_range = list(np.arange(0, total_sample_size))
    train_sample_size = int(GRFl_features_0.shape[0] * abn_ratio)
    # 
    train_indices = np.random.choice(GRFl_features_0.shape[0], size = train_sample_size, replace=False)
    # print ('1:', indices)
    # print ('2:', type(indices))
    # print ('3:', indices.shape)
    GRFl_features_0_portion_train = GRFl_features_0[train_indices]
    GRFl_0_portion_train = GRFl_0[train_indices]
    # print ('4:', GRFl_features_0_portion)
    # print ('5:', type(GRFl_features_0_portion))
    # print ('6:', GRFl_features_0_portion.shape)
    # print ('44:', GRFl_0_portion)
    # print ('7:', type(GRFl_0_portion))
    # print ('8:', GRFl_0_portion.shape)    
    #### Concatenate of  0(abnormal) + 1(healthy)
    ## 3398 + 16574 = 19972
    GRFf_train = np.concatenate((GRFl_features_0_portion_train, GRFl_features_1),axis = 0)
    GRFl_train = np.concatenate((GRFl_0_portion_train, GRFl_1),axis = 0)
    # print ('9:', type(GRFf)) 
    # print ('10:', GRFf.shape) 
    # print ('11:', type(GRFl)) 
    # print ('12:', GRFl.shape)
    
    # 
    test_indices = []
    for e in total_sample_range:
        if e not in train_indices:
            # test_indices.append(np.where(total_sample_range==e)[0][0])
            test_indices.append(e)
            
    ##
    test_indices = np.array(test_indices)
    GRFl_features_0_portion_test = GRFl_features_0[test_indices]
    GRFl_0_portion_test = GRFl_0[test_indices]
    ##
    GRFf_test = np.concatenate((GRFl_features_0_portion_test, GRFl_features_1), axis = 0)
    GRFl_test = np.concatenate((GRFl_0_portion_test, GRFl_1), axis = 0)

    ####
    print ('13:', GRFf_train.shape)
    print ('14:', GRFl_train.shape)
    print ('15:', GRFf_test.shape)
    print ('16:', GRFl_test.shape)

    return GRFf_train, GRFl_train, GRFf_test, GRFl_test


def balance_01(X, Y):
    GRFf = X
    GRFl = Y
    #
    smo = SMOTE(n_jobs=-1)
    #
    ####
    GRFf_1d = GRFf.reshape(GRFf.shape[0], -1)
    # print ('test1:', GRFf.shape)
    # 
    GRFf_1d_fnum = GRFf.shape[1]
    
    # print ('1:', GRFf_1d.shape)
    GRFf_re, GRFl_re = smo.fit_resample(GRFf_1d, GRFl)
    
    # print ('2:', GRFf_re.shape)
    # print ('3:', GRFl_re.shape)
    # 
    GRFf_re = GRFf_re.reshape(GRFf_re.shape[0], GRFf_1d_fnum, -1)
    #
    # print ('4:', GRFf_re.shape)
    
    
    #### test the distribution proportion
    #
    #
    GRFl_re_0_indice = np.where(GRFl_re==0)
    GRFl_re_1_indice = np.where(GRFl_re==1)
    
    ####
    # print ('5:', len(list(GRFl_re_0_indice[0]))) 
    # print ('6:', len(list(GRFl_re_1_indice[0]))) 
    # print ('7:', GRFf_re)
    #
    #
    #
    print ('17:', len(list(GRFl_re_0_indice[0])))
    print ('18:', len(list(GRFl_re_1_indice[0])))    
    
    return GRFf_re, GRFl_re



#
#
# 
def build_dataset(args):
    RG_GRFf_file_path = '/home/liullhappy/imageNet/rgDatasets/GRFf.pkl'
    RG_GRFl_file_path = '/home/liullhappy/imageNet/rgDatasets/GRFl.pkl'
    # 
    abn_ratio = 0.3 # 30%
    # abn_ratio = 0.2 # 20%
    # abn_ratio = 0.1 # 10%
    # abn_ratio = 0.05 # 5%
    # abn_ratio = 0.01 # 1% 

    
    #   
    # 
    GRFf_train_part, GRFl_train_part, GRFf_val_part, GRFl_val_part  =  partition(RG_GRFf_file_path, RG_GRFl_file_path, abn_ratio)
    # 
    GRFf_train_ba, GRFl_train_ba = balance_01(GRFf_train_part, GRFl_train_part)
    # 
    GRFf_val_ba, GRFl_val_ba = balance_01(GRFf_val_part, GRFl_val_part)
    
    # ------------------------ 2、normlization max-min scaler------------------------
    GRFf_train = torch.from_numpy(GRFf_train_ba)
    GRFf_train = maxmin1(GRFf_train)

    GRFf_val = torch.from_numpy(GRFf_val_ba)
    GRFf_val = maxmin1(GRFf_val)
    
    #
    GRFl_train = torch.from_numpy(GRFl_train_ba)
    GRFl_val = torch.from_numpy(GRFl_val_ba)

    # ------------------------ 3、Tensor Dataset------------------------
    n_train = TensorDataset(GRFf_train, GRFl_train)
    n_test = TensorDataset(GRFf_val, GRFl_val)
    
    return n_train, n_test, 1