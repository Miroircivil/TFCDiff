import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
import numpy as np
from timm.layers import to_2tuple
import torch_dct as dct

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

# We tried to replace Conv1d with Dynamic_conv1d, but it doesn't perform as well as it does in image deblurring tasks
class dynamic_attention(nn.Module):
    def __init__(self, dim, ratios, K, temperature, init_weight=True):
        super(dynamic_attention, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if dim!=3:
            hidden_dim = int(dim*ratios)+1
        else:
            hidden_dim = K
        self.fc1 = Conv1d(dim, hidden_dim, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = Conv1d(hidden_dim, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self):
        if self.temperature!=1:
            self.temperature -= 3
            #print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv1d(nn.Module):
    def __init__(self, dim, dim_out, kernel_size, ratio=0.25, stride=1, padding=1, dilation=1, groups=1, bias=True, K=4, temperature=34, init_weight=True):
        super(Dynamic_conv1d, self).__init__()
        assert dim%groups==0
        self.dim = dim
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = dynamic_attention(dim, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, dim_out, dim//groups, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, dim_out))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.update_temperature()

    def forward(self, x):
        softmax_attention = self.attention(x)
        batch_size, channel, length = x.size()
        x = x.view(1, -1, length, )
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.dim_out, self.dim//self.groups, self.kernel_size,)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv1d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv1d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.dim_out, output.size(-1))
        return output
    
# PositionalEncoding Source�� https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        noise_level=noise_level.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Detour Upsample
class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(16, dim)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = Conv1d(dim, dim, 3, padding=1, padding_mode='reflect')

    def forward(self, x):
        _, _, length = x.shape
        x = self.norm(x)
        x = dct.idct(F.pad(x, (0, int(length*2.6)), mode='constant', value=0), norm='ortho')
        x = self.up(x)
        x = dct.dct(x, norm='ortho')[:, :, :length*2]
        return self.conv(x)

# Detour Downsample
class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(16, dim)
        self.conv = Conv1d(dim, dim, 3, stride=2, padding=1, padding_mode='reflect')

    def forward(self, x):
        _, _, length = x.shape
        x = self.norm(x)
        x = dct.idct(F.pad(x, (0, int(length*2.6)), mode='constant', value=0), norm='ortho')
        x = self.conv(x)
        x = dct.dct(x, norm='ortho')[:, :, :length//2]
        return x


# We tried to implement SELayer in the Skip Connection, but it degrade performance
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.conv = Conv1d(channel, channel, 3, padding=1, padding_mode='reflect')
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            Linear(channel, channel // reduction, bias=False),
            Swish(),
            Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel

    def forward(self, x):
        b, c, _ = x.shape
        input = x
        x = self.conv(x)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return input + x * y.expand_as(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            Conv1d(dim, dim_out, 3, padding=1, padding_mode='reflect')
        )

    def forward(self, x):
        return self.block(x)
    
# not used
class Blockdym(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            # use dynamic convolution
            Dynamic_conv1d(dim, dim_out, 3)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=True, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = Conv1d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.dw = Conv1d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel, bias=False)
        self.qkv = Conv1d(in_channel, in_channel * 3, 1, bias=True)
        self.out = Conv1d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, seq_len = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        norm = self.dw(norm)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, seq_len)
        query, key, value = qkv.chunk(3, dim=2)  # bncl

        attn = torch.einsum(
            "bncl, bncL -> bnlL", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, seq_len, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, seq_len, seq_len)

        out = torch.einsum("bnlL, bncL -> bncl", attn, value).contiguous()
        out = self.out(out.view(batch, channel, seq_len))

        return out + input

class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

# Temporal Feature Extraction (TFE)
class TFE(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=True, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = Conv1d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        _, _, seq_len = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = F.pad(h, (0, int(seq_len*2.6)), mode='constant', value=0)
        h = dct.idct(h, norm='ortho')
        h = self.block2(h)
        h = dct.dct(h, norm='ortho')
        h = h[:, :, :seq_len]
        return h + self.res_conv(x)

# Temporal Feature Fusion (TFF)
class TFF(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.dw = Conv1d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel, bias=False)
        self.qkv = Conv1d(in_channel, in_channel * 3, 1, bias=True)
        self.out = Conv1d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, seq_len = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        norm = self.dw(norm)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, seq_len)
        query, key, value = qkv.chunk(3, dim=2)  # bncl

        q_idct = dct.idct(F.pad(query, (0, int(seq_len*2.6)), mode='constant', value=0), norm='ortho')
        k_idct = dct.idct(F.pad(key, (0,int(seq_len*2.6)), mode='constant', value=0), norm='ortho')
        v_idct = dct.idct(F.pad(value, (0, int(seq_len*2.6)), mode='constant', value=0), norm='ortho')

        attn = torch.einsum(
            "bncl, bncL -> bnlL", q_idct, k_idct
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, int(seq_len*3.6), -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, int(seq_len*3.6), int(seq_len*3.6))

        out = torch.einsum("bnlL, bncL -> bncl", attn, v_idct).contiguous()
        out = dct.dct(out, norm='ortho')
        out = out[:, :, :, :seq_len]
        out = self.out(out.view(batch, channel, seq_len))

        return out + input

# Temporal Feature Enhancement Mechanism (TFEM)
class TFEM(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = TFE(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = TFF(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=2,
        out_channel=1,
        inner_channel=64,
        norm_groups=32,
        channel_mults=(1, 2, 2, 2),
        attn_res=(250,),
        res_blocks=2,
        dropout=0,
        with_noise_level_emb=True,
        seq_len=1000
    ):
        super().__init__()
    
        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                Linear(inner_channel, inner_channel * 4),
                Swish(),
                Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = seq_len
        downs = [Conv1d(in_channel, inner_channel,
                           kernel_size=3, padding=1, padding_mode='reflect')]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(TFEM(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            # downs.append(SELayer(channel_mult, reduction=4))
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
            # SELayer(pre_channel, reduction=4)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                # feat_channel = feat_channels.pop()
                # ups.append(SELayer(feat_channel, reduction=4))
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            # ups.append(SELayer(pre_channel, reduction=4))
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = nn.Sequential(
            nn.GroupNorm(16, pre_channel),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            Conv1d(pre_channel, default(out_channel, in_channel), 3, padding=1, padding_mode='reflect')
        )

    def forward(self, x, time):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, TFEM):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)
            # if not isinstance(layer, SELayer):
            #     feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            # elif isinstance(layer, TFEM):
            #     x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            # if isinstance(layer, SELayer):
            #     feats_concat = layer(feats.pop())
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)
