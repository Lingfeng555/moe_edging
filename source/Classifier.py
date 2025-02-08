import torch
import torch.nn as nn
from .Prototype1 import Prototype1

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim):
        """
        Inicializa un MLP con capas ocultas definidas.
        
        Parámetros:
          input_dim (int): Dimensión de entrada.
          hidden_units (list): Número de neuronas por capa oculta.
          output_dim (int): Número de neuronas de salida.
        """
        super(MLP, self).__init__()
        layers = []
        # Primera capa oculta (entrada + activación)
        layers.append(nn.Linear(input_dim, hidden_units[0]))
        layers.append(nn.ReLU())
        # Capas ocultas adicionales
        for i in range(1, len(hidden_units)):
            layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            layers.append(nn.ReLU())
        # Capa de salida (sin activación para usar con CrossEntropyLoss)
        layers.append(nn.Linear(hidden_units[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MetalClassifier(nn.Module):

    def __init__(self, output_len = 2, num_feature_extractors = 4, features_per_extractor = 2):
        super(MetalClassifier, self).__init__()

        self.feature_extractors = []

        self.feature_extractors = nn.ModuleList([Prototype1(num_attention_heads=16, output_len=features_per_extractor) for _ in range(num_feature_extractors)])

        self.fc1 = MLP(input_dim=num_feature_extractors*features_per_extractor, output_dim=output_len, hidden_units= [(num_feature_extractors * features_per_extractor * output_len) for _ in range(num_feature_extractors * output_len)])

    def forward (self, x): # The image is (batch, 1 100, 100) image
        x = torch.stack([extractor(x) for extractor in self.feature_extractors], dim=1)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        #print(x.shape)
        return x
    
if __name__ == "__main__":
    # Crear instancia del modelo
    model = MetalClassifier(output_len=6)

    # Crear tensor de entrada de prueba (batch_size=1, height=100, width=100)
    input_tensor = torch.randn(32, 1, 100, 100)  

    # Pasar el tensor por el modelo
    output = model(input_tensor)

    # Mostrar la forma de la salida final
    print("Forma de la salida después de `view`:", output.shape)