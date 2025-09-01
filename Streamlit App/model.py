import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, p_drop=0.0):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        x = F.relu(self.dw_bn(self.dw(x)), inplace=True)
        x = F.relu(self.pw_bn(self.pw(x)), inplace=True)
        x = self.drop(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, p_drop=0.0):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_ch, out_ch, stride=stride, p_drop=p_drop)
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.conv(x)
        res = x if self.proj is None else self.proj(x)
        return F.relu(out + res, inplace=True)

class CitrusNet(nn.Module):
    def __init__(self, num_classes=4, width=(32, 64, 128, 256), p_drop=0.15):
        super().__init__()
        w1, w2, w3, w4 = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, w1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ResidualBlock(w1,  w2, stride=1, p_drop=p_drop)
        self.stage2 = ResidualBlock(w2,  w2, stride=1, p_drop=p_drop)
        self.stage3 = ResidualBlock(w2,  w3, stride=2, p_drop=p_drop)
        self.stage4 = ResidualBlock(w3,  w3, stride=1, p_drop=p_drop)
        self.stage5 = ResidualBlock(w3,  w4, stride=2, p_drop=p_drop)
        self.stage6 = ResidualBlock(w4,  w4, stride=1, p_drop=p_drop)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(w4, num_classes)
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.stage4(x)
        x = self.stage5(x); x = self.stage6(x)
        x = self.head(x)
        return x