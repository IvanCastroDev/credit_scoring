import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class CreditScoringModel(nn.Module):
    def __init__(self, num_features: int, hidden_layers: List[int], dropout_rate: float = 0.1, use_batch_norm: bool = True, activation_fn: str = 'ReLU'):
        super(CreditScoringModel, self).__init__()

        self.num_features = num_features
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation_fn = activation_fn

        layers = []

        # el número de caracteristicas del dataset es utilizado como el total de valores de ingreso en cada neurona de la primera capa oculta
        input_size = num_features

        for i, layer_size in enumerate(hidden_layers):
            layers.append(nn.Linear(input_size, layer_size))

            if use_batch_norm:
                layers.append(nn.LazyBatchNorm1d(layer_size))

            if activation_fn == 'ReLU':
                layers.append(nn.ReLU())

            if activation_fn == 'LeakyReLU':
                layers.append(nn.LeakyReLU())

            if activation_fn == 'GELU':
                layers.append(nn.GELU())

            # El dropout funciona con base a probabilidades, con las cuales se decide si el valor de ingreso será cero (Si su valor será tomado para las siguientes capaz) para evitar el overfiting o sobre entrenamiento
            layers.append(nn.Dropout(dropout_rate))

            # Se actualiza el valor de valores de entrada de la siguiente capa de neuronas para que sea igual al total de las salidas en la capa actual
            input_size = layer_size

        layers.append(nn.Linear(input_size, 1))

        self.network = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args: 
            x: Input Tensor of shape (batch_size, num_features)

        Returns:
            Output tensor of shape (batch_size, 1) with probabilities
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities
        Args:
            x: Input Tensor

        Returns:
            Probabilities for both classes (good=1, bad=0)
        """

        with torch.no_grad():
            logits = self.forward(x)
            prob_good = torch.sigmoid(logits)
            prob_bad = 1 - prob_good
            return torch.cat([prob_bad, prob_good], dim=1)
        
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary prediction
        Args:
            x: Input Tensor
            threshold: Classification threshold

        Returns:
            Binary predictions (0 = bad, 1 = good) 
        """ 

        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            return (probabilities > threshold).int()
        
    def get_model_info(self) -> dict:
        return {
            "model_type": "CreditScoringModel",
            "num_features": self.num_features,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "activation_fn": self.activation_fn,
            "architecture": {
                "input_layer": self.num_features,
                "hidden_layers": self.hidden_layers,
                "output_layer": 1
            },
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }