import torch.nn as nn
from .bidirectional_encoder import BidirectionalEncoder
from .liquid_embedding import LiquidEmbedding
from .improved_transformer_block import ImprovedTransformerBlock

class LiquidFoundationModelOptimized(nn.Module):
    """
    Implementa un modelo de fundación 'líquido' optimizado que combina un codificador bidireccional
    y un decodificador con capas de transformador mejoradas y embedding líquido.
    """
    def __init__(self, vocab_size, embed_dim=256, num_layers=4, num_heads=8, ff_hidden_dim=1024,
                 num_experts=4, expert_dim=256, max_length=2048, window_size=256, compression_ratio=0.5, 
                 entropy_weight=0.1, top_k=2, dynamic_k=False):
        """
        Inicializa el modelo LiquidFoundationModelOptimized.

        Args:
            vocab_size (int): Tamaño del vocabulario.
            embed_dim (int): Dimensión de los embeddings.
            num_layers (int): Número de capas en el codificador y decodificador.
            num_heads (int): Número de cabezas de atención en cada capa.
            ff_hidden_dim (int): Dimensión oculta de la capa feed-forward.
            num_experts (int): Número de expertos en la capa MoE.
            expert_dim (int): Dimensión de salida de cada experto.
            max_length (int): Longitud máxima de la secuencia.
            window_size (int): Tamaño de la ventana para la atención local.
            compression_ratio (float): Ratio de compresión para el embedding líquido.
            entropy_weight (float): Peso para la regularización de entropía en MoE.
            top_k (int): Número inicial de expertos a seleccionar en MoE.
            dynamic_k (bool): Si se ajusta dinámicamente el número de expertos en MoE.
        """
        super(LiquidFoundationModelOptimized, self).__init__()
        
        # Inicializar el codificador bidireccional
        self.encoder = BidirectionalEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_length=max_length,
            compression_ratio=compression_ratio,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            window_size=window_size,
            num_experts=num_experts,
            expert_dim=expert_dim,
            entropy_weight=entropy_weight,
            top_k=top_k,
            dynamic_k=dynamic_k
        )
        
        # Inicializar el embedding líquido para el decodificador
        self.decoder_embedding = LiquidEmbedding(
            vocab_size, 
            embed_dim, 
            max_length, 
            base_compression_ratio=0.5, 
            min_compression_ratio=0.1
        )
        
        # Inicializar las capas del decodificador
        self.decoder_layers = nn.ModuleList([
            ImprovedTransformerBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                ff_hidden_dim=ff_hidden_dim, 
                num_experts=num_experts, 
                expert_dim=expert_dim, 
                window_size=window_size, 
                bidirectional=False,  # El decodificador es unidireccional
                dropout=0.12, 
                entropy_weight=entropy_weight, 
                top_k=top_k, 
                dynamic_k=dynamic_k
            )
            for _ in range(num_layers)
        ])
        
        # Capa de normalización final
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Capa de salida para generar logits sobre el vocabulario
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
        self.max_length = max_length
        self.compression_ratio = compression_ratio

    def forward(self, encoder_input_ids, decoder_input_ids):
        """
        Realiza la pasada hacia adelante del modelo.

        Args:
            encoder_input_ids (Tensor): IDs de entrada para el codificador.
            decoder_input_ids (Tensor): IDs de entrada para el decodificador.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: 
                - Logits de salida
                - Pérdida de reconstrucción del codificador
                - Pérdida de reconstrucción del decodificador
                - Pérdida de entropía del codificador
                - Pérdida de entropía total del decodificador
        """
        # Procesar la entrada a través del codificador
        encoder_output, recon_loss_enc, entropy_loss_enc = self.encoder(encoder_input_ids)
        
        # Aplicar embedding líquido a la entrada del decodificador
        decoder_embeddings, recon_loss_dec = self.decoder_embedding(decoder_input_ids)
        
        total_entropy_loss_dec = 0
        # Pasar por cada capa del decodificador
        for layer in self.decoder_layers:
            decoder_embeddings, entropy_loss = layer(decoder_embeddings)
            total_entropy_loss_dec += entropy_loss
        
        # Aplicar normalización final
        decoder_embeddings = self.layer_norm(decoder_embeddings)
        
        # Generar logits de salida
        logits = self.output_layer(decoder_embeddings)
        
        # Limitar la salida a max_length
        return logits[:, :self.max_length, :], recon_loss_enc, recon_loss_dec, entropy_loss_enc, total_entropy_loss_dec