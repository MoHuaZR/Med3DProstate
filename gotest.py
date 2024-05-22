import torch.nn as nn
import torch
from timm.models.layers import DropPath, to_3tuple, trunc_normal_

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=768, multi_conv=False):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        num_patches = (img_size[2] // patch_size[2]) * (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.Sequential(
                    nn.Conv3d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv3d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                )
        else:
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            # print("AAAAAAAAAAA")
    def forward(self, x):
        B, C, D, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input image size ({H}*{W}*{D}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        print("AAAAAAAAAAAAAAAA:", x.shape)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = x.shape
        q = self.wq(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
class Transpose3d(nn.Module):
    def __init__(self, embed_dim, patch_size, in_channel = [64, 256, 512, 1024], img_size = [32,32,16,16]):
        super().__init__()
        self.UpConvls = nn.ModuleList()
        self.patch_size = patch_size
        self.num_patches_ls = []
        self.embed_dim = embed_dim
        for i in range(len(img_size)):
            self.num_patch = img_size[i] // self.patch_size
            self.num_patches_ls.append(self.num_patch)
        for i in range(len(in_channel)):
            if img_size[i] == 32:
                self.UpConvls.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(embed_dim, embed_dim, kernel_size=4, stride=4),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose3d(embed_dim, in_channel[i], kernel_size=2, stride=2),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose3d(in_channel[i], in_channel[i], kernel_size=2, stride=2))
                    )
            elif img_size[i] == 16:
                self.UpConvls.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(embed_dim, embed_dim, kernel_size=2, stride=2),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose3d(embed_dim, in_channel[i], kernel_size=2, stride=2),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose3d(in_channel[i], in_channel[i], kernel_size=2, stride=2))
                    )
            
    def forward(self, num, x):  # (B N C)
        B, _, _= x.shape
        #(B C D_N H_N W_N) 
        x = x.transpose(1,2).reshape(B, self.embed_dim, self.num_patches_ls[num], self.num_patches_ls[num], self.num_patches_ls[num])  
        UpConv = self.UpConvls[num]
        #(B C D H W)
        x = UpConv(x) 
        return x
    
class MultimodalAttention3D(nn.Module):
    def __init__(self, in_channel = [64, 256, 512, 1024], img_size = [32,32,16,16], embed_dim = 768, patch_size=16):
        super(MultimodalAttention3D, self).__init__()
        self.patch_x_ls = nn.ModuleList()
        self.patch_y_ls = nn.ModuleList()
        self.cross_ls = nn.ModuleList()
        self.in_channels = in_channel
        self.imgs_size = img_size
        for i in range(len(in_channel)):
            self.patch_x_ls.append(PatchEmbed(img_size=img_size[i], patch_size=16, in_chans=in_channel[i], embed_dim=768, multi_conv=False))
        for i in range(len(in_channel)):
            self.patch_y_ls.append(PatchEmbed(img_size=img_size[i], patch_size=16, in_chans=in_channel[i], embed_dim=768, multi_conv=False))
        self.cross_1 = CrossAttention(dim=768, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        self.cross_2 = CrossAttention(dim=768, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        self.upconv1 = Transpose3d(embed_dim, patch_size, in_channel = [64, 256, 512, 1024], img_size = [32,32,16,16])
        self.upconv2 = Transpose3d(embed_dim, patch_size, in_channel = [64, 256, 512, 1024], img_size = [32,32,16,16])
        
    def forward(self,num, x, y):
        #(B C D H N) --> (B N C)
        patch_x = self.patch_x_ls[num]
        patch_y = self.patch_y_ls[num]
        x = patch_x(x)
        y = patch_y(y)
        
        #(B N C) --> (B N C)
        x = self.cross_1(x, y)
        y = self.cross_2(y, x)
        
        # #(B N C) --> (B C D H N)
        x = self.upconv1(num, x)
        y = self.upconv2(num, y)
        
        return x, y 
    
if __name__ == '__main__':
    a = torch.randint(1,3,(4, 64, 32, 32, 32))
    b = torch.randint(4,6,(4, 64, 32, 32, 32))
    a = a.float()
    b = b.float()
     
    # in_chans = 64
    # embed_dim = 768
    # patch_size = 16
    # proj = nn.ConvTranspose3d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)
    # small = nn.Conv3d(in_chans,embed_dim,kernel_size=patch_size, stride=patch_size)
    
    # c = small(a)
    # d = proj(c)
    # print(a.shape)
    # print(c.shape)
    # print(d.shape)
    # trans = Transpose3d(768, 16, in_channel = [64, 256, 512, 1024], img_size = [32,32,16,16])
    multi = MultimodalAttention3D(in_channel = [64, 256, 512, 1024], img_size= [32,32,16,16], embed_dim = 768, patch_size=16)
    # patch = PatchEmbed(img_size=32, patch_size=16, in_chans=1, embed_dim=768, multi_conv=False)
    # cross = CrossAttention(dim=768, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
    # c_a = patch(a)
    # c_b = patch(b)
    # c = torch.randint(1,3,(4,768,8))
    # c = c.float()
    d_1, d_2= multi(0, a, b)
    print(d_1.shape)
    print(d_2.shape)