import torch
import torch.nn as nn

VGG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


class VGGNET(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(VGGNET, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.make_conv_layers(VGG16)

        self.fc_layer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )


    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # 가중치 초기화
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                # 편향 초기화
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layer(x)
        return x

    def make_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    # 더 나은 학습을 위해 추가
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGNET(in_channels=3, num_classes=1000).to(device)
    BATCH_SIZE = 3
    x = torch.randn(3, 3, 224, 224).to(device)
    assert model(x).shape == torch.Size([BATCH_SIZE, 1000])
    print(model(x).shape)