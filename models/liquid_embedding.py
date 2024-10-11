import torch.nn as nn
import torch.nn.functional as F
import torch

class LiquidEmbedding(nn.Module):
    """
    Implementa un embedding 'líquido' que adapta dinámicamente la compresión basada en la complejidad de la entrada.
    """
    def __init__(self, vocab_size, embed_dim, max_length=2048, base_compression_ratio=0.5, min_compression_ratio=0.1):
        """
        Inicializa el módulo LiquidEmbedding.

        Args:
            vocab_size (int): Tamaño del vocabulario.
            embed_dim (int): Dimensión de los embeddings.
            max_length (int): Longitud máxima de la secuencia.
            base_compression_ratio (float): Ratio de compresión base.
            min_compression_ratio (float): Ratio de compresión mínimo para evitar valores negativos.
        """
        super(LiquidEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.base_compression_ratio = base_compression_ratio
        self.min_compression_ratio = min_compression_ratio
        # Capa de embedding para tokens
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Capa de embedding para posiciones
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        # Capas convolucionales para procesar los embeddings
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)
        # Capa de proyección final
        self.proj = nn.Linear(embed_dim, embed_dim)
        # Función de pérdida para la reconstrucción
        self.reconstruction_loss_fn = nn.MSELoss()

    def forward(self, x):
        """
        Realiza la pasada hacia adelante del módulo LiquidEmbedding.

        Args:
            x (Tensor): Tensor de entrada de forma [batch_size, seq_length].

        Returns:
            Tuple[Tensor, Tensor]: Embeddings procesados y la pérdida de reconstrucción.
        """
        batch_size, seq_length = x.size()
        device = x.device

        # Generar embeddings de posición
        positions = torch.arange(0, seq_length, device=device).unsqueeze(0).expand(batch_size, seq_length)
        
        # Combinar embeddings de token y posición
        x = self.token_embedding(x) + self.position_embedding(positions)

        # Aplicar capas convolucionales
        x = self.conv1(x.permute(0, 2, 1))
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        x = x.to(torch.float32)

        # Aplicar FFT para análisis de frecuencia
        x_fft = torch.fft.fft(x, dim=1)

        # Calcular la complejidad de la secuencia basada en la magnitud de las frecuencias
        magnitude = torch.abs(x_fft)
        complexity = (magnitude > 0.1 * magnitude.max(dim=1, keepdim=True).values).float().mean(dim=(1, 2))
        
        # Calcular el ratio de compresión dinámico basado en la complejidad
        dynamic_compression_ratio = self.base_compression_ratio * (1 - complexity)
        dynamic_compression_ratio = torch.clamp(dynamic_compression_ratio, min=self.min_compression_ratio, max=1.0)

        # Calcular N para cada muestra en el batch
        N = (dynamic_compression_ratio * seq_length).long()
        N = torch.clamp(N, min=1, max=seq_length)

        max_N = N.max().item()

        # Comprimir x_fft para cada muestra
        x_fft_compressed = torch.zeros((batch_size, max_N, x_fft.size(-1)), dtype=torch.complex64, device=device)
        mask = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=device)
        
        for i, n in enumerate(N):
            n = n.item()
            x_fft_compressed[i, :n, :] = x_fft[i, :n, :]
            mask[i, :n] = 1  # Marcar las posiciones válidas

        # Reconstrucción usando IFFT
        x_ifft = torch.fft.ifft(x_fft_compressed, n=seq_length, dim=1).real
        x_ifft = self.proj(x_ifft)
        x_ifft = x_ifft.to(x.dtype)

        recon_target = torch.fft.ifft(x_fft_compressed, n=seq_length, dim=1).real
        recon_target = self.proj(recon_target).to(x.dtype)
        
        # Expandir la máscara para que coincida con la forma de x_ifft y recon_target
        mask_expanded = mask.unsqueeze(-1).expand_as(x_ifft)
        
        # Calcular la pérdida de reconstrucción usando la máscara
        diff = (x_ifft - recon_target).abs() * mask_expanded.float()
        loss_recon = diff.sum() / mask_expanded.sum() if mask_expanded.sum() > 0 else torch.tensor(0.0, device=x.device)

        return x_ifft, loss_recon