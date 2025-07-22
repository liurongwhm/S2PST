import torch
import torch.nn as nn

class FE(nn.Module):
    def __init__(self, in_channels, patch_size, init_weights=True):
        super(FE, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        cfg = [64, 128, 128]
        layers = []
        for index in range(len(cfg)):
            v = cfg[index]
            bn = nn.BatchNorm2d(v)
            if index == len(cfg) - 2:
                Block = CenterAttentionAwareConvolutionBlock(in_channel=in_channels, kernel_size=3)
                conv3d = ReConvSetBlock(in_channels, v)
                layers += [Block, conv3d, bn, nn.LeakyReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, stride=1, kernel_size=3)
                layers += [conv2d, bn, nn.LeakyReLU(inplace=True)]

            in_channels = v

        self.features = nn.Sequential(*layers)
        self.flag_get_hidden_layer = False
        self.hidden_layer = self.hidden_layer if self.flag_get_hidden_layer else self._get_final_flattened_size()

        if init_weights:
            self._initialize_weights()

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((2, self.in_channels,
                             self.patch_size, self.patch_size))
            x = self.features(x)
            x = self.pool(x)
            t, c, w, h = x.size()
        self.flag_get_hidden_layer = True
        return int(t * c * w * h / 2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.contiguous().view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CenterAttentionAwareConvolutionBlock(nn.Module):
    def __init__(self, in_channel, kernel_size):
        super().__init__()
        self.in_channel = in_channel
        self.DWconv = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                groups=in_channel)
        self.act = nn.GELU()
        self.kv = nn.Conv2d(in_channels=self.in_channel, out_channels=2 * self.in_channel, kernel_size=1)
        self.query = nn.Linear(self.in_channel, self.in_channel)

    def forward(self, x):  # x: B * C * ps * ps
        input = x
        batch_size, in_channel, patch_size, _ = x.shape
        assert in_channel == self.in_channel
        center = patch_size // 2
        center_feature = x[:, :, center, center]

        kv = self.kv(x).reshape(batch_size, 2 * in_channel, -1)
        kv = self.act(kv)
        k, v = torch.chunk(kv, chunks=2, dim=1)

        q = self.query(center_feature).unsqueeze(1)
        attention = torch.bmm(q, k) / (self.in_channel ** (1 / 2))
        gate = (v * attention).reshape(batch_size, in_channel, patch_size, patch_size)

        x = self.DWconv(x)
        x = x * gate
        out = x + input
        return out

class ReConvSetBlock(nn.Module):
    def __init__(self, cin, cout):
        super(ReConvSetBlock, self).__init__()
        dilation = (1, 1, 1)
        self.conv = nn.ModuleList([
            nn.Conv3d(cin, cout, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(cin, cout, (1, 3, 1), dilation=dilation, stride=(1, 1, 1), padding=(0, 1, 0)),
            nn.Conv3d(cin, cout, (1, 1, 3), dilation=dilation, stride=(1, 1, 1), padding=(0, 0, 1))
        ])
        self.Conv_mixnas = nn.Conv3d(cout * 3, cout, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x.unsqueeze(2)
        spectralnas = []
        for layer in self.conv:
            nas_ = layer(x)
            spectralnas.append(nas_)
        spectralnas = torch.cat(spectralnas, dim=1)
        x = self.Conv_mixnas(spectralnas)
        x = x.squeeze(2)
        return x
