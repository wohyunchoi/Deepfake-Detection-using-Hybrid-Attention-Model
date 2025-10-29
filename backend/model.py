import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm import create_model
from torchvision.models.feature_extraction import get_graph_node_names

def rgb_fft_magnitude(image_tensor):
    # image_tensor: [B, 3, H, W]
    fft = torch.fft.fft2(image_tensor, dim=(-2, -1))
    fft_mag = torch.log(torch.abs(fft).mean(dim=1, keepdim=True) + 1e-8)
    return torch.cat([image_tensor, fft_mag], dim=1)  # [B, 4, H, W]

def kaiming_init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.shared_mlp(self.avg_pool(x).view(b, c))
        max = self.shared_mlp(self.max_pool(x).view(b, c))
        scale = torch.sigmoid(avg + max).view(b, c, 1, 1)
        x = x * scale

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att
    
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x
    
class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, stride=1, grow_first=True, dropout_rate=0.2):
        super(Block, self).__init__()
        layers = []
        filters = in_filters
        if grow_first:
            layers.append(SeparableConv2d(in_filters, out_filters))
            layers.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            layers.append(nn.ReLU())
            layers.append(SeparableConv2d(filters, filters))
            layers.append(nn.BatchNorm2d(filters))

        if not grow_first:
            layers.append(SeparableConv2d(in_filters, out_filters))
            layers.append(nn.BatchNorm2d(out_filters))

        self.rep = nn.Sequential(*layers)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, stride, 1)
        self.dropout = nn.Dropout2d(dropout_rate)

        if stride != 1 or in_filters != out_filters:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

    def forward(self, x):
        res = x
        x = self.rep(x)
        x = self.pool(x)
        x = self.dropout(x)
        if self.skip is not None:
            res = self.skip(res)
            res = self.skipbn(res)
        return x + res

class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.2):
        super(DeepfakeDetector, self).__init__()

        self.entry = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )

        self.block1 = Block(64, 128, 2, stride=2, grow_first=True, dropout_rate=dropout_rate)
        self.block2 = Block(128, 256, 2, stride=2, grow_first=True, dropout_rate=dropout_rate)
        self.block3 = Block(256, 728, 2, stride=2, grow_first=True, dropout_rate=dropout_rate)
        self.cbam3 = CBAM(728)

        self.middle = nn.Sequential(
            *[Block(728, 728, 3, dropout_rate=dropout_rate) for _ in range(4)]
        )
        self.cbam_middle = CBAM(728)

        self.exit = nn.Sequential(
            Block(728, 1024, 2, stride=2, grow_first=False, dropout_rate=dropout_rate),
            SeparableConv2d(1024, 1536),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            SeparableConv2d(1536, 2048),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
        self.cbam_exit = CBAM(2048)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

        self.apply(kaiming_init_weights)

    def forward(self, x):
        x = self.entry(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.cbam3(x)
        x = self.middle(x)
        x = self.cbam_middle(x)
        x = self.exit(x)
        x = self.cbam_exit(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)