import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedLinearUnit(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size * 2)

    def forward(self, x):
        x = self.fc(x)
        # La mitad de la salida es el valor, la otra mitad es el gate (sigmoide)
        return x[:, :, :x.shape[-1]//2] * torch.sigmoid(x[:, :, x.shape[-1]//2:])

class GatedResidualNetwork(nn.Module):
    """
    Componente central del TFT. Permite transformaciones no lineales
    pero manteniendo una ruta 'limpia' para la señal si no es necesaria.
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.glu = GatedLinearUnit(output_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        
        # Proyeccion residual si las dimensiones no coinciden
        self.project = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.project(x)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)
        return self.layer_norm(x + residual)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_inputs, input_dim, hidden_size, dropout=0.1):
        super().__init__()
        # Cada feature se procesa individualmente
        self.single_variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_size, hidden_size, dropout) 
            for _ in range(num_inputs)
        ])
        # Red que decide la importancia de cada variable
        self.flattened_grn = GatedResidualNetwork(num_inputs * input_dim, hidden_size, num_inputs, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch, time, num_inputs * input_dim]
        # 1. Calculamos pesos de atencion para las variables
        weights = self.softmax(self.flattened_grn(x)).unsqueeze(-1) # [batch, time, num_inputs, 1]
        
        # 2. Procesamos cada variable y las combinamos segun su peso
        # Separamos las variables del vector plano
        var_outputs = []
        for i, grn in enumerate(self.single_variable_grns):
            # Asumimos que cada variable tiene dimension 'input_dim'
            var_slice = x[:, :, i:i+1] 
            var_outputs.append(grn(var_slice))
        
        var_outputs = torch.stack(var_outputs, dim=-2) # [batch, time, num_inputs, hidden_size]
        
        # Multiplicamos por los pesos (Atencion sobre variables)
        selected_output = (weights * var_outputs).sum(dim=-2)
        return selected_output, weights