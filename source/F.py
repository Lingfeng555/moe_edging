import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
    def __init__(self, latent_dim=256):
        super(Autoencoder, self).__init__()
        # Encoder: 4 capas convolucionales
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # 100x100 -> 50x50
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),   # 50x50 -> 25x25
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 25x25 -> 13x13
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 13x13 -> 7x7
            nn.ReLU(inplace=True)
        )

        self.experts = nn.ModuleList([FeatureExpert() for _ in range(64)])

        # Decoder: 4 capas de convolución transpuesta
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),  # 7x7 -> 13x13
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=0),  # 13x13 -> 25x25
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),   # 25x25 -> 50x50
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),    # 50x50 -> 100x100
            nn.Sigmoid()  # Salida en rango [0,1]
        )

    def forward(self, x):
        # Encoder
        x = self.encoder_conv(x)              # Resultado: (batch, 64, 7, 7)
        x = x.view(x.size(0), 64, -1)               # Aplanar

        #print(x.shape)
        x = torch.stack([self.experts[i](x[:, i, :]) for i in range(64)], dim=1)
        #print(x.shape)

        #Decoder
        x = x.view(x.size(0), 64, 7, 7)         # Reconfigurar forma
        x = self.decoder_conv(x)              # Reconstrucción de imagen
        return x

if __name__ == "__main__":
    # Crear instancia del modelo
    model = Autoencoder()

    # Crear tensor de entrada de prueba (batch_size=1, height=100, width=100)
    input_tensor = torch.randn(32, 1, 100, 100)  

    # Pasar el tensor por el modelo
    output = model(input_tensor)

    # Mostrar la forma de la salida final
    print("Forma de la salida después de `view`:", output.shape)