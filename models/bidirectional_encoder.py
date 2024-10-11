import torch.nn as nn
from .liquid_embedding import LiquidEmbedding
from .improved_transformer_block import ImprovedTransformerBlock

class BidirectionalEncoder(nn.Module):
    """
    Implements a bidirectional encoder with multiple layers of improved transformer blocks.
    """
    def __init__(self, vocab_size, embed_dim, max_length=2048, compression_ratio=0.5, num_layers=4, num_heads=8, ff_hidden_dim=1024, window_size=256, num_experts=2, expert_dim=256, entropy_weight=0.1, top_k=2, dynamic_k=False):
        """
        Initializes the BidirectionalEncoder module.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimension of the embeddings.
            max_length (int): Maximum sequence length.
            compression_ratio (float): Base compression ratio for liquid embedding.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            ff_hidden_dim (int): Hidden dimension of the feed-forward layer.
            window_size (int): Window size for local attention.
            num_experts (int): Number of experts in the MoE layer.
            expert_dim (int): Output dimension of each expert.
            entropy_weight (float): Weight for entropy regularization in MoE.
            top_k (int): Number of experts to select in MoE.
            dynamic_k (bool): Whether to dynamically adjust the number of experts.
        """
        super(BidirectionalEncoder, self).__init__()
        # Initialize liquid embedding layer
        self.embedding = LiquidEmbedding(vocab_size, embed_dim, max_length, base_compression_ratio=compression_ratio)
        # Create a list of improved transformer blocks
        self.layers = nn.ModuleList([
            ImprovedTransformerBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                ff_hidden_dim=ff_hidden_dim, 
                num_experts=num_experts, 
                expert_dim=expert_dim, 
                window_size=window_size, 
                bidirectional=True, 
                dropout=0.12, 
                entropy_weight=entropy_weight, 
                top_k=top_k, 
                dynamic_k=dynamic_k
            )
            for _ in range(num_layers)
        ])
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Performs the forward pass of the bidirectional encoder.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_length].

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Processed output, reconstruction loss, and total entropy loss.
        """
        # Apply liquid embedding
        x, recon_loss = self.embedding(x)
        total_entropy_loss = 0
        # Pass through each transformer layer
        for layer in self.layers:
            x, entropy_loss = layer(x)
            total_entropy_loss += entropy_loss
        # Final normalization
        x = self.layer_norm(x)
        return x, recon_loss, total_entropy_loss