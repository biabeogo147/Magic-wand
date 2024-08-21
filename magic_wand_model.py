from pprint import pprint
from torch import nn


class magic_wand_model(nn.Module):
    def __init__(self, num_classes=10):
        super(magic_wand_model, self).__init__()

        self.conv1 = self.make_block_conv(1, 16)
        self.conv2 = self.make_block_conv(16, 32)
        self.conv3 = self.make_block_conv(32, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = self.make_block_linear(4096, 2024, 0.5)
        self.fc2 = self.make_block_linear(2024, 1024, 0.5)
        self.fc3 = nn.Linear(in_features=1024, out_features=num_classes)

    def make_block_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def make_block_linear(self, in_features, out_features, p=0.5):
        return nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = magic_wand_model()
    pprint(model)