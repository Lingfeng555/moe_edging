import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: aplica 6 convoluciones para pasar por los canales 1 -> 16 -> 32 -> 32 -> 64 -> 64
        self.encoder = nn.Sequential(
            # Capa 1: 1 -> 16, reduce 100x100 a 50x50
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Capa 2: 16 -> 32, 50x50 a 25x25
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Capa 3: 32 -> 32, 25x25 a 13x13
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Capa 4: 32 -> 64, 13x13 a 7x7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Capa 5: 64 -> 64, 7x7 a 4x4
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Capa 6: 64 -> 64, mantiene 4x4 (procesamiento adicional)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder: invierte la secuencia, de 64 a 1
        self.decoder = nn.Sequential(
            # Capa 1: 64 -> 64, mantiene 4x4
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Capa 2: 64 -> 64, 4x4 a 7x7
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),
            # Capa 3: 64 -> 32, 7x7 a 13x13
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),
            # Capa 4: 32 -> 32, 13x13 a 25x25
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),
            # Capa 5: 32 -> 16, 25x25 a 50x50 (se usa output_padding=1)
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            # Capa 6: 16 -> 1, 50x50 a 100x100 (se usa output_padding=1)
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Salida en rango [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x