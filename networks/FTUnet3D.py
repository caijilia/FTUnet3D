from typing import Tuple, Union
import torch
import torch.nn as nn
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT
import sys
from blocks.unet3d import UNet3D
import torch.nn.functional as F


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class CBAM3D(nn.Module):
    def __init__(self, in_ch, rate, only_ch=0, only_sp=0):
        super(CBAM3D, self).__init__()
        self.fc1 = nn.Conv3d(in_channels=in_ch, out_channels=int(in_ch / rate), kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_channels=int(in_ch / rate), out_channels=in_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        self.fc3 = nn.Conv3d(in_channels=in_ch, out_channels=int(in_ch / rate), kernel_size=1)
        self.fc4 = nn.Conv3d(in_channels=int(in_ch / rate), out_channels=in_ch, kernel_size=1)

        self.ch_use = only_ch
        self.ch_sp_use = only_sp

    def forward(self, x):
        x_in = x  
        x = torch.mean(x.mean((3, 4), keepdim=True), 2, keepdim=True) 
        x = self.fc1(x) 
        x = self.relu(x)  
        x = self.fc2(x)  
        if self.ch_use == 1:
            return x * x_in 
        elif self.ch_use == 0:
            x = x * x_in

        s_in = x  
        s = self.compress(x)  
        s = self.spatial(s)  
        if self.ch_sp_use == 1:
            return s 
        elif self.ch_sp_use == 0:
            s = s * s_in

        return s


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv3d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm3d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class HFF3D(nn.Module): 
    def __init__(self, channels, res=0):
        super(HFF3D, self).__init__()
        self.in_channel_0 = channels[0] 
        self.in_channel_1 = channels[1] 
        self.in_channel_2 = channels[2] 
        self.fuse_conv1 = Conv(inp_dim=(channels[0]+channels[1]), out_dim=channels[2], bn=True) 
        self.res_fc = res
        self.att = CBAM3D(in_ch=(channels[2]), rate=4)
        self.fuse_conv2 = Conv(inp_dim=(channels[2]+channels[2]), out_dim=channels[2], bn=True)
    
    def forward(self, x):
        hi3 = x[0] 
        hi6 = x[1] 
        x_in = x[2] 

        hi3_4 = F.interpolate(hi3, size=[4, 4, 4], mode='trilinear', align_corners=False) 
        hi6_4 = F.interpolate(hi6, size=[4, 4, 4], mode='trilinear', align_corners=False) 
        hi_3  = self.fuse_conv1(torch.cat([hi3_4, hi6_4], dim=1)) 
        hi_3  = F.interpolate(hi_3, size=[8, 8, 8], mode='trilinear', align_corners=False) 
        hi_3_att = self.att(hi_3)
        x_out = self.fuse_conv2(torch.cat([hi_3_att, x_in], dim=1))
        if self.res_fc == 1:
            x_out = x_out + x_in

        return x_out


class Focus_TransUnet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.fri_vit_imgsize = [self.patch_size[0], self.patch_size[1], self.patch_size[2]]
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=self.fri_vit_imgsize,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.pre_coding = nn.MaxPool3d(kernel_size=self.feat_size[0], stride=self.feat_size[0])
        self.SpaMap_1 = Conv(inp_dim=hidden_size, out_dim=feature_size*1)
        self.SpaMap_2 = Conv(inp_dim=hidden_size, out_dim=feature_size*1)
        self.SpaMap_3 = Conv(inp_dim=hidden_size, out_dim=feature_size*2)
        self.HFF3D_fc = HFF3D(channels=[16, 16, 32], res=1)

        self.fri_imsz = img_size
        self.sec_unet = UNet3D(in_channels=in_channels, out_channels=out_channels, init_features=64)
        self.ResConv = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.DeepFusion = nn.Sequential(
            UnetrUpBlock(
                spatial_dims=3,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            ),
            UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=1)  
        )
        self.att = nn.Sequential(
            CBAM3D(in_ch=feature_size, rate=4),
            nn.BatchNorm3d(feature_size)
        ) 
        self.one_hot = nn.Sigmoid() 

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights["state_dict"]:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.position_embeddings_3d"]
            )
            self.vit.patch_embedding.cls_token.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.cls_token"]
            )
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.weight"]
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.bias"]
            )

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights["state_dict"]["module.transformer.norm.weight"])
            self.vit.norm.bias.copy_(weights["state_dict"]["module.transformer.norm.bias"])

    def forward(self, x_in):
        HFF3D_in = []
        fri_x = self.pre_coding(x_in)
        x, hidden_states_out = self.vit(fri_x)
        hi_3 = self.SpaMap_1(self.proj_feat(hidden_states_out[3], self.hidden_size, [1, 1, 1]))
        HFF3D_in.append(hi_3)
        hi_6 = self.SpaMap_2(self.proj_feat(hidden_states_out[6], self.hidden_size, [1, 1, 1]))
        HFF3D_in.append(hi_6)
        hi_12 = self.SpaMap_3(self.proj_feat(x, self.hidden_size, [1, 1, 1])) 
        hi_12 = F.interpolate(hi_12, size=[8, 8, 8], mode='trilinear', align_corners=False) 

        HFF3D_in.append(hi_12)
        fuse_out = self.HFF3D_fc(HFF3D_in)
        spacomp = self.ResConv(fri_x) 
        spacomp = self.att(spacomp)
        HFF_out = self.DeepFusion(fuse_out, spacomp)

        HFF_out = F.interpolate(HFF_out, size=self.fri_imsz, mode='trilinear', align_corners=False)
        ActMap = self.one_hot(HFF_out)
        sec_in = x_in * ActMap + x_in

        logits = self.sec_unet(sec_in)
        return logits
