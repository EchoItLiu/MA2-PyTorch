# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import random
import math
import numpy as np

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        #
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        #
        self.height, self.width = input_size
        
        #
        self.num_patches = self.height * self.width
        # 196 * 0.75 = 147
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        '''
        Maks: total patches 196, mask patches 147
        '''
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str
    
    def __call__(self):
        #
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        #
        np.random.shuffle(mask)
        return mask # [196]

def RandomMaskingGenerator1d(sample_size, input_size, mask_ratio):
    # masks samples
    # 101 * 0.75 = 75
    mask_samples_T = []
    num_mask = int(mask_ratio * input_size)
    
    for i in range(sample_size):
        #
        mask = np.hstack([
            np.zeros(input_size - num_mask),
            np.ones(num_mask),
        ])
        # print ('m1:', mask.shape)
        # 
        np.random.shuffle(mask)
        #
        mask_samples_T.append(mask)

    # print ('I:', np.array(mask_samples_T).shape)
    # print ('IR:', np.array(mask_samples_T))

    return np.array(mask_samples_T)
    


    return np.array(mask_samples_T) # [84551 × 101]


