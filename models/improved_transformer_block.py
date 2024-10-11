import torch.nn as nn
import torch.nn.functional as F
import torch
from .enhanced_local_attention import EnhancedLocalAttention
from .dilated_convolution import DilatedConvolution
from .moe_layer import MoELayer

class ImprovedTransformerBlock(nn.Module):
    """
    Implementa un bloque de transformador mejorado con atención local, convolución dilatada y MoE.
    """
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, num_experts, expert_dim, window_size=256, bidirectional=True, dropout=0.12, entropy_weight=0.1, top_k=2, dynamic_k=False):
        """
        Inicializa el módulo ImprovedTransformerBlock.

        Args:
            embed_dim (int): Dimensión de los embeddings.
            num_heads (int): Número de cabezas de atención.
            ff_hidden_dim (int): Dimensión oculta de la capa feed-forward.
            num_experts (int): Número de expertos en la capa MoE.
            expert_dim (int): Dimensión de salida de cada experto.
            window_size (int): Tamaño de la ventana para la atención local.
            bidirectional (bool): Si se usa atención bidireccional.
            dropout (float): Tasa de dropout.
            entropy_weight (float): Peso para la regularización de entropía en MoE.
            top_k (int): Número de expertos a seleccionar en MoE.
            dynamic_k (bool): Si se ajusta dinámicamente el número de expertos.
        """
        super(ImprovedTransformerBlock, self).__init__()
        self.attention = EnhancedLocalAttention(embed_dim, num_heads, window_size, bidirectional)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dilated_conv = DilatedConvolution(embed_dim, kernel_size=3, dilation=2)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.moe = MoELayer(
            input_dim=embed_dim, 
            hidden_dim=embed_dim, 
            num_experts=num_experts, 
            expert_dim=expert_dim, 
            dropout=dropout, 
            entropy_weight=entropy_weight, 
            top_k=top_k, 
            dynamic_k=dynamic_k
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.ff_layer = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        
    def forward(self, x):
        """
        Realiza la pasada hacia adelante del módulo ImprovedTransformerBlock.

        Args:
            x (Tensor): Tensor de entrada.

        Returns:
            Tuple[Tensor, Tensor]: Salida procesada y pérdida de entropía.
        """
        # Aplicar atención local y residual
        x = x + self.dropout1(self.attention(self.norm1(x)))
        # Aplicar convolución dilatada y residual
        x = x + self.dropout2(self.dilated_conv(self.norm2(x)))
        # Aplicar MoE y residual
        moe_output, entropy_loss = self.moe(self.norm3(x))
        x = x + self.dropout3(moe_output)
        # Aplicar capa feed-forward y residual
        x = x + self.ff_layer(x)
        return x, entropy_loss