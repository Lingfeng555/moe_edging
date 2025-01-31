import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):

    def __init__(self):
        super(CNNBlock, self).__init__() ## We are assuming a grey scale image
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(32) 

    def forward(self, x):
        if x.ndim == 3:  # (batch_size, 100, 100)
            x = x.unsqueeze(1) 
        # x: [1, 100, 100]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)                   # 100 -> 50
        # print(x.shape)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)                   # 50 -> 25
        # print(x.shape)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)                   # 25 -> 12
        # print(x.shape)
        x = self.batch_norm(x)
        return x

if __name__ == "__main__":
    # Crear instancia del modelo
    model = CNNBlock()

    # Crear tensor de entrada de prueba (batch_size=1, height=100, width=100)
    input_tensor = torch.randn(32,1, 100, 100)  

    # Pasar el tensor por el modelo
    output = model(input_tensor)

    # Mostrar la forma de la salida final
    print("Forma de la salida después de `view`:", output.shape)