import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, p_dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=p_dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride),
            nn.BatchNorm2d(out_ch)
        ) if stride != 1 or in_ch != out_ch else nn.Identity()

    def forward(self, x):
        skip = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        return self.relu(x + skip)

class ResUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, output_padding=0, p_dropout=0.2):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=stride, 
                                        padding=1, output_padding=output_padding)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=p_dropout)
        self.conv2 = nn.ConvTranspose2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 1, stride=stride,
                               output_padding=output_padding),
            nn.BatchNorm2d(out_ch)
        ) if stride != 1 or in_ch != out_ch else nn.Identity()

    def forward(self, x):
        skip = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        return self.relu(x + skip)

class FeatureExpert(nn.Module):
    def __init__(self):
        super(FeatureExpert, self).__init__()
        self.fc1 = nn.Linear(49, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 49)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            ResDownBlock(1, 8),
            ResDownBlock(8, 16),
            ResDownBlock(16, 32),
            ResDownBlock(32, 64)
        )
        #self.experts = nn.ModuleList([FeatureExpert() for _ in range(64)])
        self.decoder_conv = nn.Sequential(
            ResUpBlock(64, 32),
            ResUpBlock(32, 16),
            ResUpBlock(16, 8, output_padding=1),
            ResUpBlock(8, 1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_conv(x)       # (batch, 64, 7, 7)
        #x = x.view(x.size(0), 64, -1)
        #x = torch.stack([self.experts[i](x[:, i, :]) for i in range(64)], dim=1)
        #x = x.view(x.size(0), 64, 7, 7)
        x = self.decoder_conv(x)
        return x

import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

# Codificador
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ResBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = ResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.block3 = ResBlock(128, 256)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        return self.block3(x)

# Decodificador
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block3 = ResBlock(256, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block2 = ResBlock(128, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block1 = ResBlock(64, 1)

    def forward(self, x):
        x = self.block3(x)
        x = self.up2(x)
        x = self.block2(x)
        x = self.up1(x)
        return self.block1(x)

# Autoencoder completo
class ResNetAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


if __name__ == "__main__":
    # Crear instancia del modelo
    model = Autoencoder()

    # Crear tensor de entrada de prueba (batch_size=1, height=100, width=100)
    input_tensor = torch.randn(32, 1, 100, 100)  

    # Pasar el tensor por el modelo
    output = model(input_tensor)

    # Mostrar la forma de la salida final
    print("Forma de la salida después de `view`:", output.shape)