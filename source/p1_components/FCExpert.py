import torch
import torch.nn as nn
import torch.nn.functional as F

class FCExpert(nn.Module):
    def __init__(self):
        super(FCExpert, self).__init__()
        self.fc1 = nn.Linear(144, 72)
        self.fc2 = nn.Linear(72, 36)
        self.fc3 = nn.Linear(36, 18)
        self.fc4 = nn.Linear(18, 9)
        self.fc5 = nn.Linear(9, 2)

    def forward(self, x, attention_value):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return (x * attention_value)
    
if __name__ == "__main__":
    # Crear instancia del modelo
    model = FCExpert()

    # Crear tensor de entrada de prueba (batch_size=1, height=100, width=100)
    input_tensor = torch.randn(32, 144)  

    # Pasar el tensor por el modelo
    output = model(input_tensor, 4)

    # Mostrar la forma de la salida final
    print("Forma de la salida después de `view`:", output.shape)