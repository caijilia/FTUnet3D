from typing import Tuple, Union
import torch
import torch.nn as nn
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT
import sys
sys.path.append('/home/lihao2021/PycharmProjects/FTUNET3D/networks/')
from blocks.unet3d import UNet3D
import torch.nn.functional as F


class PFC(nn.Module):
    def __init__(self, in_ch, channels, kernel_size=7):
        super(PFC, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv3d(in_ch, channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(channels))
        self.depthwise = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size, groups=channels, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(channels))
        self.pointwise = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(channels))

    def forward(self, x_in):
        x_in = self.input_layer(x_in)
        residual = x_in
        x = self.depthwise(x_in)
        x = x + residual
        x = self.pointwise(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class IRE(nn.Module):
    def __init__(self, in_ch, rate, only_ch=0, only_sp=0):
        super(IRE, self).__init__()
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
        x_in = x  # 4 16 64 64 64
        x = torch.mean(x.mean((3, 4), keepdim=True), 2, keepdim=True)  # 8 256 1 1
        x = self.fc1(x)  # 8 256 1 1 -> 8 64 1 1
        x = self.relu(x)  # 8 64 1 1 -> 8 64 1 1
        x = self.fc2(x)  # 8 64 1 1 -> 8 256 1 1
        if self.ch_use == 1:
            return x * x_in  # 注意力已赋予输入feature map
        elif self.ch_use == 0:
            x = x * x_in

        # 在这里再加上空间注意力
        s_in = x  # 8 256 12 16
        s = self.compress(x)  # 8 256 12 16 -> 8 2 12 16
        s = self.spatial(s)  # 8 2 12 16 -> 8 1 12 16
        if self.ch_sp_use == 1:
            return s  # 单独输出 注意力att
        elif self.ch_sp_use == 0:
            s = s * s_in

        """ # 再加上上下文注意力  或许是上下文吧
        c_in = s  # 8 256 12 16
        c = self.fc3(s)  # 8 256 12 16 -> 8 64 12 16
        c = self.relu(c)
        c = self.fc4(c)  # 8 64 12 16 -> 8 256 12 16
        c = self.sigmoid(c) * c_in  # 8 256 12 16 -> 8 256 12 16 """

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


class SSF(nn.Module): # Step_by_Step_Fuse
    def __init__(self, channels, res=0):
        super(SSF, self).__init__()
        self.in_channel_0 = channels[0] # 16
        self.in_channel_1 = channels[1] # 16
        self.in_channel_2 = channels[2] # 32 feature*2
        self.fuse_conv1 = Conv(inp_dim=(channels[0]+channels[1]), out_dim=channels[2], bn=True) # or IRE?
        self.res_fc = res
        self.att = IRE(in_ch=(channels[2]), rate=4)
        self.fuse_conv2 = Conv(inp_dim=(channels[2]+channels[2]), out_dim=channels[2], bn=True)
    
    def forward(self, x):
        hi3 = x[0] 
        hi6 = x[1] 
        x_in = x[2] # b 32 8 8 8

        hi3_4 = F.interpolate(hi3, size=[4, 4, 4], mode='trilinear', align_corners=False) # b 16 4 4 4
        hi6_4 = F.interpolate(hi6, size=[4, 4, 4], mode='trilinear', align_corners=False) # b 16 4 4 4 
        hi_3  = self.fuse_conv1(torch.cat([hi3_4, hi6_4], dim=1)) # b 32 4 4 4
        hi_3  = F.interpolate(hi_3, size=[8, 8, 8], mode='trilinear', align_corners=False) # b 32 8 8 8
        hi_3_att = self.att(hi_3)
        x_out = self.fuse_conv2(torch.cat([hi_3_att, x_in], dim=1))
        if self.res_fc == 1:
            x_out = x_out + x_in

        return x_out


class fine_UNETR(nn.Module):
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
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

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
        self.fri_down = nn.MaxPool3d(kernel_size=self.feat_size[0], stride=self.feat_size[0])
        self.conv_fri = Conv(inp_dim=hidden_size, out_dim=feature_size*2)
        self.ssf_conv1 = Conv(inp_dim=hidden_size, out_dim=feature_size*1)
        self.ssf_conv2 = Conv(inp_dim=hidden_size, out_dim=feature_size*1)
        self.ssf_fc = SSF(channels=[16, 16, 32], res=1)

        self.fri_imsz = img_size
        self.sec_unet = UNet3D(in_channels=in_channels, out_channels=out_channels, init_features=64)
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.att = nn.Sequential(
            IRE(in_ch=feature_size, rate=4),
            nn.BatchNorm3d(feature_size)
            )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=1)  # type: ignore
        self.one_hot = nn.Sigmoid()  # IRE(in_ch=1, rate=1)

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
        ssf_in = []
        fri_x_down = self.fri_down(x_in)
        x, hidden_states_out = self.vit(fri_x_down)
        hi_3 = self.ssf_conv1(self.proj_feat(hidden_states_out[3], self.hidden_size, [1, 1, 1]))
        ssf_in.append(hi_3)
        hi_6 = self.ssf_conv2(self.proj_feat(hidden_states_out[6], self.hidden_size, [1, 1, 1]))
        ssf_in.append(hi_6)
        dec4 = self.conv_fri(self.proj_feat(x, self.hidden_size, [1, 1, 1])) # b 1 768 -> b 768 1 1 1 -> b 32 1 1 1
        dec4 = F.interpolate(dec4, size=[8, 8, 8], mode='trilinear', align_corners=False) # b 32 1 1 1 -> b 32 8 8 8

        ssf_in.append(dec4)
        dec4 = self.ssf_fc(ssf_in)

        enc1 = self.encoder1(fri_x_down) # b 1 16 16 16 -> b 16 16 16 16
        enc1 = self.att(enc1)

        out = self.decoder2(dec4, enc1)
        fri_out = self.out(out)
        fri_out = F.interpolate(fri_out, size=self.fri_imsz, mode='trilinear', align_corners=False)
        # ------
        fri_out = self.one_hot(fri_out)
        sec_in = x_in * fri_out + x_in
        logits = self.sec_unet(sec_in)
        return logits

