import torch.nn as nn
import torch

class DilatedConvolution(nn.Module):
    """
    Implementa una capa de convolución dilatada para capturar dependencias de largo alcance.
    """
    def __init__(self, channels, kernel_size, dilation):
        """
        Inicializa el módulo DilatedConvolution.

        Args:
            channels (int): Número de canales de entrada y salida.
            kernel_size (int): Tamaño del kernel de convolución.
            dilation (int): Factor de dilatación.
        """
        super(DilatedConvolution, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding='same', dilation=dilation)
        
    def forward(self, x):
        """
        Realiza la pasada hacia adelante del módulo DilatedConvolution.

        Args:
            x (Tensor): Tensor de entrada de forma [batch_size, seq_length, channels].

        Returns:
            Tensor: Salida procesada por la convolución dilatada.
        """
        # Transponer para aplicar la convolución y luego volver a transponer
        return self.conv(x.transpose(1, 2)).transpose(1, 2)