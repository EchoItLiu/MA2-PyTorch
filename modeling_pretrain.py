# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_mae_base_patch16_224', 
    'pretrain_mae_large_patch16_224', 
]


# 
class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=10, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        
        self.patch_embed = PatchEmbed(
            img_size = img_size, patch_size = patch_size, in_chans = in_chans, embed_dim=embed_dim)
        num_patches = 101

        print ('****Encoder Depth****:', depth)
        print ('****Encoder embed_dim****:', self.num_features)

        # TODO: Add the cls token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            # 1 × 196 × 768
            # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            
            # 1 × 101 × 768
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            # print ('m1:', self.pos_embed.shape)
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
            # 1 × 101 × 768
            # print ('m1-:', self.pos_embed.shape)
          
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        #
        #
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # 
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        #
        self.apply(self._init_weights)

    
    def _init_weights(self, m):
        #
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            # bias
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # ln        
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    # This decorator indicates to the compiler that a function or method should be ignored and left as a Python function
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    
    def forward_features(self, x, mask):
        # mask: 128 × 101
        # x:128 × 10 × 101 → 128 × 101 × 768
        x = self.patch_embed(x)
        
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        # x = torch.cat((cls_tokens, x), dim=1) # 196 → 197
        # 1 × 101 × 768
        # print ('m2:', self.pos_embed.shape)
        # 1 × 101 × 768
        # print ('m3:', self.pos_embed.type_as(x).to(x.device).clone().detach().shape)
        #
        #
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        # print ('m4:', x.shape)
        #
        # 128 × 101 × 768
        B, _, C = x.shape

        # ~
        # 128 × (101- int(101*0.75)) × 768 → 128 × 26 × 768 
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        #
        # 128 × 26 × 768 
        for blk in self.blocks:
            x_vis = blk(x_vis)
            
        #
        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        #
        #
        #
        x = self.forward_features(x, mask)
        #
        #
        #
        x = self.head(x)
        return x

#
#
#
##### decoder_num_classes
class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=10, embed_dim=768, depth=4,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196,
                 ):
        super().__init__()
        # 768  
        self.num_classes = num_classes
        # 3 × (16 * 16)
        # assert num_classes == 3 * patch_size ** 2
        #
        #
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        
        print ('*****Decoder_Embed_Dim*****:', self.num_features)
        # 16
        self.patch_size = patch_size
        # 
        #
        # × 101 × dim(512)
        # 
        # decay,
        print ('*****depth_ablation*****:', depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        
        #
        self.norm =  norm_layer(embed_dim)
        # 
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # 
        self.apply(self._init_weights)

    #
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    # 512 → 10
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        #
        for blk in self.blocks:
            x = blk(x)

        #
        # 128 × 75 × dim(768)
        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        # 128 × 101 × 768    
        else:
            x = self.head(self.norm(x)) # [B, N, 3*16^2]

        return x



class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=10,  # 
                 encoder_num_classes=0, # 
                 encoder_embed_dim=768, # 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values)
        
        # 768 → 384
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)) # ablation-4
        
        # 1 × 101  × 384
        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)
        # print ('VT1:', self.pos_embed.shape)
        trunc_normal_(self.mask_token, std=.02)
        # mask token ablation
        # trunc_normal_(self.mask_token, std=.02)
        
        # print ('VT2:', self.mask_token.shape)
        #
        #
        #
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    
    def forward(self, x, mask):
        x_vis = self.encoder(x, mask) # [B, N_vis, C_e] → 128 ×  26 × 768 
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d] →  128 ×  26 × 384
        # B: 128
        # L: 26
        # C: 384
        B, L, C = x_vis.shape

        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        #
        # 1 × 101  × 384 → 128 × 101  × 384
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        # print ('D1:', expand_pos_embed.shape)
        # 128 × 26 × 384
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        # print ('D2:', pos_emd_vis.shape)
        # 128 × 75 × 384 
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        # print ('D3:', pos_emd_mask.shape)
        # 128 × 101 × 384
        # print ('D4:', self.mask_token.shape) 
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)

        # x_full = torch.cat([x_vis + pos_emd_vis, pos_emd_mask], dim=1) # ablation-4
        
        # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        # print ('VT3:', x_full.shape)



        
        # decoder_x: 128 × 75 × 10  | pos_emd_mask: 75   
        x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]
        # print ('VT4:', pos_emd_mask.shape[1])
        # print ('VT5:', x.shape)

        return x

@register_model
def pretrain_mae_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


########################################################################


@register_model
def pretrain_mae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        # encoder_embed_dim=1024, # ablation
        # encoder_embed_dim=768, # ablation Default
        # encoder_embed_dim=512, # ablation
        # encoder_embed_dim=384, # ablation
        encoder_embed_dim=256, # ablation
        # encoder_embed_dim=128, # ablation
        #
        # encoder_depth=18, 
        encoder_depth=12,   # ablation Default
        # encoder_depth=8,  # ablation
        # encoder_depth=4,  # ablation
        # encoder_depth=2,  # ablation
        # encoder_depth=1,  # ablation
        
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=10,
        decoder_embed_dim=128, # ablation
        # decoder_embed_dim=256,  # ablationcc
        # decoder_embed_dim=3c84  # ablation Default
        # decoder_embed_dim=512,  # ablation
        # decoder_embed_dim=768,  # ablation
        # decoder_embed_dim=1024,  # ablation
        
        # decoder_depth=1, # ablation
        # decoder_depth=2, # ablation
        decoder_depth=4, # ablation Default
        # decoder_depth=8, # ablation
        # decoder_depth=12, # ablation
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
    

##################################################################

@register_model
def pretrain_mae_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=1024, 
        encoder_depth=24, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
    