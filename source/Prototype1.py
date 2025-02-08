import torch
import torch.nn as nn
from .p1_components.CNNBlock import CNNBlock
from .p1_components.AxialAttentionBlock import AttentionBlock
from .p1_components.FCExpert import FCExpert
from .p1_components.FinalBlock import FinalExpert

class Prototype1 (nn.Module):

    def __init__(self, num_attention_heads = 4, num_experts = 32, output_len = 2):
        super(Prototype1, self).__init__()
        self.num_experts = num_experts

        self.cnn_block = CNNBlock()
        self.attention_block = AttentionBlock(num_features=1, num_heads=num_attention_heads)
        self.experts = nn.ModuleList([FCExpert() for _ in range(num_experts)])
        self.wighted_sum = FinalExpert(output_len)

    def forward(self, x):
        features = self.cnn_block(x)

        attention_values = self.attention_block(features)
        #print(features.shape)
        features = features.view(features.shape[0], 32, -1)
        #print("Features.shape (batch_size, n_features, dataflatten) :",features.shape)
        #print("Features.shape (batch_size, n_features, ,attention value) :", attention_values.shape)
        x = torch.stack([self.experts[i](features[:, i, :], attention_values[:, i, :]) for i in range(self.num_experts)], dim=1)
        x = x.flatten(start_dim=1)
        x = self.wighted_sum(x)
        # x = torch.clamp(x, min=1.0,max=100) #activarlp solo para el modelo entrenado
        return x
    
if __name__ == "__main__":
    # Crear instancia del modelo
    model = Prototype1(output_len=6)

    # Crear tensor de entrada de prueba (batch_size=1, height=100, width=100)
    input_tensor = torch.randn(32, 1, 100, 100)  

    # Pasar el tensor por el modelo
    output = model(input_tensor)

    # Mostrar la forma de la salida final
    print("Forma de la salida después de `view`:", output.shape)
