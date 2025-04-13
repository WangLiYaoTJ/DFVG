import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import torch.nn.functional as f
import math
from einops import rearrange
from timm.models.layers import trunc_normal_
from einops import repeat

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.branch1_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)

        self.branch2_qkv = nn.Conv2d(dim, dim * 3, kernel_size=3, padding=1)

        self.merge = nn.Conv2d(dim * 3 * 2, dim * 3, kernel_size=1)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv1 = self.branch1_qkv(x)
        qkv2 = self.branch2_qkv(x)

        combined = torch.cat([qkv1, qkv2], dim=1)
        combined = self.merge(combined)

        q, k, v = combined.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm=None, bias=True):
        super(ConvLayer, self).__init__()

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.norm in ["BN" or "IN"]:
            out = self.norm_layer(out)

        return out


class UpsampleConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, norm=None, bias=True):
        super(UpsampleConvLayer, self).__init__()

        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample, mode='nearest')

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):

        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)

        if self.norm in ["BN" or "IN"]:
            out = self.norm_layer(out)

        return out

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()
    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out
class FCA(nn.Module):
    def __init__(self,channel,b=1, gamma=2):
        super(FCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()
    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)
        out = self.mix(out1,out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return input*out


class ResidualBlock0(nn.Module):

    def __init__(self, channels, norm=None, bias=True):
        super(ResidualBlock0, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, bias=bias, norm=norm)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, bias=bias, norm=norm)

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        input = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + input

        return out

class ResidualBlock1(nn.Module):

    def __init__(self, channels,norm=None, bias=True):
        super(ResidualBlock1, self).__init__()
        self.conv = ConvLayer(channels, channels, kernel_size=3, stride=1, bias=bias, norm=norm)
        self.fca = FCA(channels)

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        input = x
        out = self.relu(self.conv(x))
        out = self.fca(out)
        out = self.conv(out)
        out = out + input

        return out

class ResidualBlock(nn.Module):

    def __init__(self, channels, norm=None, bias=True):
        super(ResidualBlock, self).__init__()
        self.resblock0 = ResidualBlock0(channels, norm, bias)
        self.resblock1 = ResidualBlock1(channels, norm, bias)


    def forward(self, x):
        out = self.resblock0(x)
        out = self.resblock1(out)
        return out

class ConvLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.zeros(state_size).to(input_.device),
                torch.zeros(state_size).to(input_.device)
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        return hidden, cell


class DeflickerNet(nn.Module):
    def __init__(self, opts, nc_in, nc_out):
        super(DeflickerNet, self).__init__()

        self.blocks = opts.blocks
        self.epoch = 0
        nf = opts.nf
        use_bias = (opts.norm == "IN")
        self.conv1a = ConvLayer(6, nf * 1, kernel_size=7, stride=1, bias=use_bias, norm=opts.norm)
        self.conv1b = ConvLayer(6, nf * 1, kernel_size=7, stride=1, bias=use_bias, norm=opts.norm)
        self.conv2a = ConvLayer(nf * 1, nf * 2, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)
        self.conv2b = ConvLayer(nf * 1, nf * 2, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)
        self.conv3 = ConvLayer(nf * 4, nf * 4, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)
        self.deconv1 = UpsampleConvLayer(nf * 4, nf * 2, kernel_size=3, stride=1,
                                         upsample=2, bias=use_bias, norm=opts.norm)
        self.deconv2 = UpsampleConvLayer(nf * 4, nf * 1, kernel_size=3, stride=1, upsample=2, bias=use_bias,
                                         norm=opts.norm)
        self.deconv3 = ConvLayer(nf * 2, nc_out, kernel_size=7, stride=1, bias=True, norm=None)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=nf * 1, num_heads=nf, ffn_expansion_factor=4, bias=use_bias, LayerNorm_type=opts.norm)
            for i in range(2)
        ])

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=nf * 2, num_heads=nf, ffn_expansion_factor=4, bias=use_bias, LayerNorm_type=opts.norm)
            for i in range(2)
        ])

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=nf * 4, num_heads=nf, ffn_expansion_factor=4, bias=use_bias, LayerNorm_type=opts.norm)
            for i in range(2)
        ])

        self.ResBlocks = nn.ModuleList()

        for b in range(self.blocks):
            self.ResBlocks.append(ResidualBlock(nf * 4,  bias=use_bias, norm=opts.norm))

        self.convlstm = ConvLSTM(input_size=nf * 4, hidden_size=nf * 4, kernel_size=3)

        self.decoder_level2 = TransformerBlock(dim=nf * 2, num_heads=nf, ffn_expansion_factor=4, bias=use_bias,
                                               LayerNorm_type=opts.norm)
        self.decoder_level1 = TransformerBlock(dim=nf * 1, num_heads=nf, ffn_expansion_factor=4, bias=use_bias,
                                               LayerNorm_type=opts.norm)

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, X, prev_state):
        Xa = X[:, :6, :, :]
        Xb = X[:, 6:, :, :]

        E1a = self.relu(self.encoder_level1(self.conv1a(Xa)))
        E1b = self.relu(self.encoder_level1(self.conv1b(Xb)))

        E2a = self.relu(self.encoder_level2(self.conv2a(E1a)))
        E2b = self.relu(self.encoder_level2(self.conv2b(E1b)))

        E3 = self.relu(self.conv3(self.encoder_level3(torch.cat((E2a, E2b), 1))))

        RB = E3

        for b in range(self.blocks):
            RB = self.ResBlocks[b](RB)

        state = self.convlstm(RB, prev_state)

        D2 = self.relu(self.decoder_level2(self.deconv1(state[0])))
        C2 = torch.cat((D2, E2a), 1)
        D1 = self.relu(self.decoder_level1(self.deconv2(C2)))
        C1 = torch.cat((D1, E1a), 1)
        Y = self.deconv3(C1)

        Y = self.tanh(Y)

        return Y, state