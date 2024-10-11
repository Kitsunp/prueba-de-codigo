import torch.nn as nn
import torch.nn.functional as F
import torch

class EnhancedLocalAttention(nn.Module):
    """
    Implementa un mecanismo de atención local mejorado con ventanas superpuestas opcionales.
    """
    def __init__(self, embed_dim, num_heads, window_size=256, bidirectional=True, dropout=0.1):
        """
        Inicializa el módulo EnhancedLocalAttention.

        Args:
            embed_dim (int): Dimensión de los embeddings.
            num_heads (int): Número de cabezas de atención.
            window_size (int): Tamaño de la ventana de atención local.
            bidirectional (bool): Si se usa atención bidireccional o no.
            dropout (float): Tasa de dropout.
        """
        super(EnhancedLocalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.bidirectional = bidirectional
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        # Capa lineal para generar queries, keys y values
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        # Capa de salida
        self.out = nn.Linear(embed_dim, embed_dim)
        # Capa de dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Realiza la pasada hacia adelante del módulo EnhancedLocalAttention.

        Args:
            x (Tensor): Tensor de entrada de forma [batch_size, seq_length, embed_dim].

        Returns:
            Tensor: Salida procesada por la atención local.
        """
        B, L, C = x.shape
        # Padding para asegurar que la longitud de la secuencia es múltiplo del tamaño de la ventana
        pad_l = (self.window_size - L % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_l))
        _, L_padded, _ = x.shape
        
        # Generar queries, keys y values
        qkv = self.qkv(x).reshape(B, L_padded, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.bidirectional:
            # Atención bidireccional con ventanas superpuestas
            overlapping_size = self.window_size // 2
            step = overlapping_size
            window_size = self.window_size
            q = q.unfold(2, window_size, step).contiguous()
            k = k.unfold(2, window_size, step).contiguous()
            v = v.unfold(2, window_size, step).contiguous()
        else:
            # Atención unidireccional con ventanas no superpuestas
            q = q.unfold(2, self.window_size, self.window_size).contiguous()
            k = k.unfold(2, self.window_size, self.window_size).contiguous()
            v = v.unfold(2, self.window_size, self.window_size).contiguous()
        
        # Calcular puntajes de atención y aplicar softmax
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Aplicar atención a los values y reorganizar
        x = (attn @ v).reshape(B, self.num_heads, -1, self.head_dim).permute(0, 2, 1, 3).reshape(B, -1, C)
        x = self.out(x)
        x = self.dropout(x)
        
        # Devolver la salida sin el padding adicional
        return x[:, :L]