# prueba/utils/liquid_embedding.py

"""
LiquidEmbedding module implementing dynamic compression embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LiquidEmbedding(nn.Module):
    """
    Liquid Embedding module with dynamic compression based on input complexity.
    """

    def __init__(self, vocab_size, embed_dim, max_length=2048, base_compression_ratio=0.5, min_compression_ratio=0.1):
        """
        Initialize the LiquidEmbedding module.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimension of the embeddings.
            max_length (int): Maximum sequence length.
            base_compression_ratio (float): Base compression ratio.
            min_compression_ratio (float): Minimum compression ratio to avoid negative values.
        """
        super(LiquidEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.base_compression_ratio = base_compression_ratio
        self.min_compression_ratio = min_compression_ratio
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.reconstruction_loss_fn = nn.MSELoss()

    def forward(self, x):
        """
        Forward pass for LiquidEmbedding.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_length].

        Returns:
            Tuple[Tensor, Tensor]: Processed embeddings and reconstruction loss.
        """
        batch_size, seq_length = x.size()
        device = x.device

        # Generate position embeddings
        positions = torch.arange(0, seq_length, device=device).unsqueeze(0).expand(batch_size, seq_length)
        
        # Combine token and position embeddings
        x = self.token_embedding(x) + self.position_embedding(positions)

        # Apply convolutional layers
        x = self.conv1(x.permute(0, 2, 1))
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        x = x.to(torch.float32)

        # Apply FFT for frequency analysis
        x_fft = torch.fft.fft(x, dim=1)

        # Calculate sequence complexity based on frequency magnitudes
        magnitude = torch.abs(x_fft)
        complexity = (magnitude > 0.1 * magnitude.max(dim=1, keepdim=True).values).float().mean(dim=(1, 2))
        
        # Calculate dynamic compression ratio based on complexity
        dynamic_compression_ratio = self.base_compression_ratio * (1 - complexity)
        dynamic_compression_ratio = torch.clamp(dynamic_compression_ratio, min=self.min_compression_ratio, max=1.0)

        # Calculate N for each sample in the batch
        N = (dynamic_compression_ratio * seq_length).long()
        N = torch.clamp(N, min=1, max=seq_length)

        max_N = N.max().item()

        # Compress x_fft for each sample
        x_fft_compressed = torch.zeros((batch_size, max_N, x_fft.size(-1)), dtype=torch.complex64, device=device)
        mask = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=device)
        
        for i, n in enumerate(N):
            n = n.item()
            x_fft_compressed[i, :n, :] = x_fft[i, :n, :]
            mask[i, :n] = 1  # Mark valid positions

        # Reconstruction using IFFT
        x_ifft = torch.fft.ifft(x_fft_compressed, n=seq_length, dim=1).real
        x_ifft = self.proj(x_ifft)
        x_ifft = x_ifft.to(x.dtype)

        recon_target = torch.fft.ifft(x_fft_compressed, n=seq_length, dim=1).real
        recon_target = self.proj(recon_target).to(x.dtype)
        
        # Expand mask to match the shape of x_ifft and recon_target
        mask_expanded = mask.unsqueeze(-1).expand_as(x_ifft)
        
        # Calculate reconstruction loss using the mask
        diff = (x_ifft - recon_target).abs() * mask_expanded.float()
        loss_recon = diff.sum() / mask_expanded.sum() if mask_expanded.sum() > 0 else torch.tensor(0.0, device=x.device)

        return x_ifft, loss_recon