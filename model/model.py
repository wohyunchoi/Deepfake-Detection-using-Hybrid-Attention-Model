import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm import create_model
from torchvision.models.feature_extraction import get_graph_node_names

class XceptionDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(XceptionDeepfakeDetector, self).__init__()
        # XceptionNet
        self.backbone = create_model("xception", pretrained=False, num_classes=0)
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

        # Xavier Initialization
        self._initialize_weights()

    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SwinDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(SwinDeepfakeDetector, self).__init__()
        #Swin Transformer
        self.backbone = create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=num_classes)

        # Xavier Initialization
        self._init_weights()

    def forward(self, x):
        return self.backbone(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



class HybridDeepfakeDetector_XS(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridDeepfakeDetector_XS, self).__init__()

        # XceptionNet
        self.xception = create_model('xception', pretrained=False, num_classes=0)

        # Swin Transformer
        self.swin = create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=512)

        # Hybrid
        self.feature_fusion = nn.Linear(512 + 2048, 1024)
        self.fc = nn.Linear(1024, num_classes)

        # Xavier Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        xception_features = self.xception(x)  # (batch, 2048)

        swin_features = self.swin(x)  # (batch, 512)

        fused_features = torch.cat((xception_features, swin_features), dim=1)  # (batch, 2560)
        fused_features = self.feature_fusion(fused_features)
        
        out = self.fc(fused_features)
        return out
    


class HybridDeepfakeDetector_ES(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridDeepfakeDetector_ES, self).__init__()

        # EfficientNet-b3 (원하는 등급으로 변경 가능)
        self.efficientnet = create_model('efficientnet_b3', pretrained=False, num_classes=0)
        self.efficientnet_out_dim = self._get_output_dim(self.efficientnet)

        # Swin Transformer
        self.swin = create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=512)

        # Hybrid Fusion
        self.feature_fusion = nn.Linear(self.efficientnet_out_dim + 512, 1024)
        self.fc = nn.Linear(1024, num_classes)

        self._init_weights()

    def _get_output_dim(self, model):
        # 테스트 입력으로 feature dimension 계산
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        return out.shape[1]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        eff_features = self.efficientnet(x)  # (batch, ~1536 for b3)
        swin_features = self.swin(x)        # (batch, 512)

        fused_features = torch.cat((eff_features, swin_features), dim=1)
        fused_features = self.feature_fusion(fused_features)
        out = self.fc(fused_features)
        return out
    

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

class XceptionCBAM_FFT(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.2):
        super(XceptionCBAM_FFT, self).__init__()

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
    
class XceptionCBAM_FFT2(nn.Module):
    def __init__(self, num_classes=2):
        super(XceptionCBAM_FFT2, self).__init__()

        # 입력: 4채널 (RGB + FFT)
        self.entry = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Middle Flow
        self.block1 = Block(64, 128, 2, stride=2, grow_first=True)
        self.block2 = Block(128, 256, 2, stride=2, grow_first=True)
        self.block3 = Block(256, 728, 2, stride=2, grow_first=True)
        self.cbam3 = CBAM(728)  # block3 후 attention

        self.middle = nn.Sequential(
            *[Block(728, 728, 3) for _ in range(8)]  # XceptionNet만큼
        )

        self.cbam_middle = CBAM(728) # middle 후 attention

        self.exit = nn.Sequential(
            Block(728, 1024, 2, stride=2, grow_first=False),
            SeparableConv2d(1024, 1536),
            nn.BatchNorm2d(1536),  # BN 추가
            nn.ReLU(),
            SeparableConv2d(1536, 2048),
            nn.BatchNorm2d(2048),  # BN 추가
            nn.ReLU()
        )
        self.cbam_exit = CBAM(2048)  # exit 이후 attention

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

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
    
class EfficientCBAM_FFT(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientCBAM_FFT, self).__init__()
        base_model = models.efficientnet_b0(weights=None)

        # 입력 채널을 4로 맞추기 위해 첫 번째 conv 수정
        first_conv = base_model.features[0][0]
        new_first_conv = nn.Conv2d(
            in_channels=4,  # 채널 입력으로 변경
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )

        # 수정된 conv 삽입
        base_model.features[0][0] = new_first_conv

        self.features = base_model.features

        self.cbam_layers = nn.ModuleList([
            CBAM(24),    # after layer 2
            CBAM(112),   # after layer 5
            CBAM(1280),  # after layer 8
        ])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 2:
                x = self.cbam_layers[0](x)
            elif i == 5:
                x = self.cbam_layers[1](x)
            elif i == 8:
                x = self.cbam_layers[2](x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)
    
class ResCBAM_FFT(nn.Module):
    def __init__(self, num_classes=2):
        super(ResCBAM_FFT, self).__init__()
        base_model = models.resnet50(weights=None)

        # 3채널 → 4채널
        old_conv1 = base_model.conv1
        new_conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )

        self.conv1 = new_conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # CBAM 삽입
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.cbam1(x)

        x = self.layer2(x)
        x = self.cbam2(x)

        x = self.layer3(x)
        x = self.cbam3(x)

        x = self.layer4(x)
        x = self.cbam4(x)

        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)