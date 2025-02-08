import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from source.Prototype1 import Prototype1

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        # Extrae características de imagen 100x100 en 1 canal
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 100x100 -> 50x50
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1),  # 50x50 -> 25x25
            nn.ReLU()
        )
        # Proyecta a espacio latente
        self.fc = nn.Linear(128 * 25 * 25, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Aplanar
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=2):
        super(Decoder, self).__init__()
        # Combina vector latente y parámetros (sp, sr)
        self.fc = nn.Linear(latent_dim + cond_dim, 128 * 25 * 25)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 25x25 -> 50x50
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # 50x50 -> 100x100
            nn.Sigmoid()  # Salida en rango [0,1]
        )

    def forward(self, z, cond):
        # Concatenar vector latente y condición
        z = torch.cat([z, cond], dim=1)
        x = self.fc(z)
        x = x.view(x.size(0), 128, 25, 25)
        x = self.deconv(x)
        return x

class JointAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(JointAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)

        self.param_estimator = Prototype1(num_attention_heads=16)
        self.param_estimator.load_state_dict(torch.load("h2.pth", map_location=torch.device('cpu')))
        self.param_estimator.to("cuda")
        self.param_estimator.eval()

        self.decoder = Decoder(latent_dim, cond_dim=2)

    def forward(self, x):
        latent = self.encoder(x)
        with torch.no_grad():
            cond = self.param_estimator(x)
        recon = self.decoder(latent, cond)
        return recon


if __name__ == "__main__":
    # Crear instancia del modelo
    model = JointAutoencoder()

    # Crear tensor de entrada de prueba (batch_size=1, height=100, width=100)
    input_tensor = torch.randn(32, 1, 100, 100)  

    # Pasar el tensor por el modelo
    output = model(input_tensor)

    # Mostrar la forma de la salida final
    print("Forma de la salida después de `view`:", output.shape)