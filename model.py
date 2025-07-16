import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, disable_bn=False):
        super().__init__()
        self.disable_bn = disable_bn
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c) if not disable_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c) if not disable_bn else nn.Identity()
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.tanh(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, disable_bn=False):
        super().__init__()
        self.conv = conv_block(in_c, out_c, disable_bn=disable_bn)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, disable_bn=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2, padding=0)
        self.conv = conv_block(out_c*2, out_c, disable_bn=disable_bn)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class conv_block_bottleneck(nn.Module):
    def __init__(self, in_c, out_c, disable_bn=False):
        super().__init__()
        self.disable_bn = disable_bn
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c) if not disable_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c) if not disable_bn else nn.Identity()
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.tanh(x)
        return x

class build_net(nn.Module):
    def __init__(self, disable_bn=False, zero_skip_s1=False):
        super().__init__()
        self.disable_bn = disable_bn
        self.zero_skip_s1 = zero_skip_s1

        self.e1 = encoder_block(3, 16, disable_bn=disable_bn)
        self.e2 = encoder_block(16, 32, disable_bn=disable_bn)
        self.e3 = encoder_block(32, 64, disable_bn=disable_bn)
        self.e4 = encoder_block(64, 128, disable_bn=disable_bn)
        self.e5 = encoder_block(128, 256, disable_bn=disable_bn)
        self.b = conv_block_bottleneck(256, 512, disable_bn=disable_bn)
        self.d1 = decoder_block(512, 256, disable_bn=disable_bn)
        self.d2 = decoder_block(256, 128, disable_bn=disable_bn)
        self.d3 = decoder_block(128, 64, disable_bn=disable_bn)
        self.d4 = decoder_block(64, 32, disable_bn=disable_bn)
        self.d5 = decoder_block(32, 16, disable_bn=disable_bn)
        self.outputs = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout2d(0.3)
        self.dropout3 = nn.Dropout2d(0.3)
        self.dropout4 = nn.Dropout2d(0.3)
        self.dropout5 = nn.Dropout2d(0.3)

    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        p1 = self.dropout1(p1)
        s2, p2 = self.e2(p1)
        p2 = self.dropout2(p2)
        s3, p3 = self.e3(p2)
        p3 = self.dropout3(p3)
        s4, p4 = self.e4(p3)
        p4 = self.dropout4(p4)
        s5, p5 = self.e5(p4)
        p5 = self.dropout5(p5)
        b = self.b(p5)
        # Padding/cropping for dimensional matching in U-Net-style skip connections
        s5 = F.pad(s5, (0,0,1,0))
        b = F.pad(b, (0,0,0,0))
        d1 = self.d1(b, s5)
        s4 = F.pad(s4, (0,0,1,1))
        d2 = self.d2(d1, s4)
        s3 = F.pad(s3, (1,0,1,0))
        d2 = d2[:,:,1:-1,:]
        d3 = self.d3(d2, s3)
        d3 = d3[:,:,1:-1,:]
        s2 = s2[:,:,1:,:]
        s2 = F.pad(s2, (1,2,0,0))
        d4 = self.d4(d3, s2)
        d4 = F.pad(d4, (0,0,0,1))
        d4 = d4[:,:,:,2:-1]
        s1 = F.pad(s1, (0,0,0,0))
        d5 = self.d5(d4, torch.zeros_like(s1) if self.zero_skip_s1 else s1)
        outputs = self.outputs(d5)
        return outputs