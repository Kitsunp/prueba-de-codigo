# Importaciones
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from transformers import GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import nltk
nltk.download('all')
from torch.utils.tensorboard import SummaryWriter
import math
import os
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, precision_score, recall_score
from nltk.tokenize import word_tokenize
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
import hashlib
from torch.utils.checkpoint import checkpoint

# Configuración global
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Constantes globales
EMBED_DIM = 512
NUM_LAYERS = 12
NUM_HEADS = 8
FF_HIDDEN_DIM = 1024
NUM_EXPERTS = 8
EXPERT_DIM = 512
MAX_LENGTH = 8192
WINDOW_SIZE = 2048
COMPRESSION_RATIO = 0.5
BATCH_SIZE = 4
NUM_EPOCHS = 6
ACCUMULATION_STEPS = 6
TOP_K = 4       # Número inicial de expertos a preseleccionar
DYNAMIC_K = True # Activar el ajuste dinámico de K

# Clases y Funciones

class TextEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction()

    def calculate_metrics(self, questions, answers, labels):
        bleu_scores = []
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        f1_scores_list = []
        
        # Asegurarse de que todas las listas tengan la misma longitud
        min_length = min(len(questions), len(answers), len(labels))
        questions = questions[:min_length]
        answers = answers[:min_length]
        labels = labels[:min_length]
        
        for q, a, l in zip(questions, answers, labels):
            # Tokenización consistente
            a_tokens = word_tokenize(a.lower())
            l_tokens = word_tokenize(l.lower())
            
            # BLEU
            bleu = sentence_bleu([l_tokens], a_tokens, smoothing_function=self.smoothing.method1)
            bleu_scores.append(bleu)
            
            # ROUGE
            rouge_result = self.scorer.score(l, a)
            rouge_scores['rouge1'].append(rouge_result['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(rouge_result['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(rouge_result['rougeL'].fmeasure)
            
            # F1 (usando conjuntos para manejar longitudes diferentes)
            f1 = f1_score(
                [1 if token in l_tokens else 0 for token in set(a_tokens + l_tokens)],
                [1 if token in a_tokens else 0 for token in set(a_tokens + l_tokens)],
                average='binary',
                zero_division=1
            )
            f1_scores_list.append(f1)
        
        metrics = {
            'bleu': np.mean(bleu_scores),
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL']),
            'f1': np.mean(f1_scores_list)
        }
        return metrics

    def evaluate_coherence(self, question, cot_steps, answer):
        # Implementación simple de coherencia basada en la similitud de palabras
        all_text = question + " " + " ".join([step["text"] for step in cot_steps]) + " " + answer
        words = set(word_tokenize(all_text.lower()))
        
        question_words = set(word_tokenize(question.lower()))
        answer_words = set(word_tokenize(answer.lower()))
        
        overlap = len(question_words.intersection(answer_words))
        total_words = len(words)
        
        coherence_score = overlap / total_words if total_words > 0 else 0
        return coherence_score

# Implementación de Hooks para Monitoreo
class ActivationMonitor:
    def __init__(self, model):
        self.handles = []
        self.activations = {}
        self.gradients = {}
        self.register_hooks(model)

    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
                handle = module.register_forward_hook(self.save_activation(name))
                handle_grad = module.register_backward_hook(self.save_gradient(name))
                self.handles.append(handle)
                self.handles.append(handle_grad)

    def save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
            # Verificar rango de activaciones
            if not torch.isfinite(output).all():
                print(f"Activaciones no finitas en {name}")
        return hook

    def save_gradient(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()
            # Verificar rango de gradientes
            if not torch.isfinite(grad_output[0]).all():
                print(f"Gradientes no finitos en {name}")
        return hook

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, expert_dim, dropout=0.15, entropy_weight=0.1, top_k=2, dynamic_k=False, max_usage_ratio=0.3):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dynamic_k = dynamic_k
        self.experts = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
        self.dropout = nn.Dropout(dropout)
        self.entropy_weight = entropy_weight
        self.max_usage_ratio = max_usage_ratio
        self.expert_usage_counter = None

    def forward(self, x):
        batch_size, seq_length, input_dim = x.size()
        x_flat = x.view(-1, input_dim)
        
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Regularización de entropía
        entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-10), dim=-1).mean()
        entropy_loss = self.entropy_weight * entropy

        # Ajuste dinámico de K
        if self.dynamic_k:
            complexity = entropy.detach().item()
            K = max(1, min(self.num_experts, int(self.top_k * (1 + complexity))))
        else:
            K = self.top_k

        topk_probs, topk_indices = torch.topk(gate_probs, K, dim=-1)

        # Inicializar o reiniciar el contador de uso de expertos
        if self.expert_usage_counter is None:
            self.expert_usage_counter = torch.zeros(self.num_experts, device=x.device)
        else:
            self.expert_usage_counter = self.expert_usage_counter.to(x.device)

        expert_outputs = torch.zeros(batch_size * seq_length, self.experts[0].out_features, device=x.device, dtype=x.dtype)

        for k in range(K):
            expert_idx = topk_indices[:, k]
            mask = torch.arange(x_flat.size(0), device=x.device).unsqueeze(1) == expert_idx.unsqueeze(1)
            mask = mask.any(dim=1)
            selected_x = x_flat[mask]

            if selected_x.size(0) > 0:
                # Seleccionar el experto de manera eficiente
                unique_experts = expert_idx[mask].unique()
                for expert in unique_experts:
                    expert_mask = expert_idx[mask] == expert
                    inputs = selected_x[expert_mask]
                    output = self.dropout(self.experts[expert](inputs))
                    expert_outputs[mask][expert_mask] += output * topk_probs[:, k][mask][expert_mask].unsqueeze(1)
                    # Actualizar el contador de uso de expertos
                    self.expert_usage_counter[expert] += inputs.size(0)

        # Calcular la penalización por uso excesivo
        usage_ratios = self.expert_usage_counter / (batch_size * seq_length)
        overuse_penalty = torch.sum(F.relu(usage_ratios - self.max_usage_ratio))

        output = expert_outputs.view(batch_size, seq_length, -1)

        return output, entropy_loss + overuse_penalty

    def get_expert_usage_stats(self):
        if self.expert_usage_counter is None:
            return None
        total_usage = self.expert_usage_counter.sum().item()
        if total_usage == 0:
            return [0.0] * self.num_experts
        usage_percentages = (self.expert_usage_counter / total_usage * 100).tolist()
        return usage_percentages
def next_power_of_two(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()
class LiquidEmbedding(nn.Module):
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
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)
        self.proj = nn.Linear(embed_dim, embed_dim)
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
        # Eliminamos la conversión a float32 para mantener la precisión mixta
        # x = x.to(torch.float32)  # Esta línea se ha eliminado
        # Añadir padding para que seq_length sea potencia de 2
        padded_seq_length = next_power_of_two(seq_length)
        padding = padded_seq_length - seq_length
        if padding > 0:
            x = F.pad(x, (0, 0, 0, padding))  # Agregar padding al final de la secuencia

        # Aplicar FFT
        x_fft = torch.fft.fft(x, dim=1)

        # Calcular la complejidad de la secuencia
        magnitude = torch.abs(x_fft)
        complexity = (magnitude > 0.1 * magnitude.max(dim=1, keepdim=True).values).float().mean(dim=(1, 2))
        
        # Calcular el ratio de compresión dinámico
        dynamic_compression_ratio = self.base_compression_ratio * (1 - complexity)
        dynamic_compression_ratio = torch.clamp(dynamic_compression_ratio, min=self.min_compression_ratio, max=1.0)

        # Calcular N para cada muestra en el batch
        N = (dynamic_compression_ratio * seq_length).long()
        N = torch.clamp(N, min=1, max=seq_length)  # Asegurar que N no sea mayor que seq_length

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
        x_ifft = x_ifft.type_as(x)  # Mantener el dtype original

        recon_target = torch.fft.ifft(x_fft_compressed, n=seq_length, dim=1).real
        recon_target = self.proj(recon_target).type_as(x)
        
        # Expandir la máscara para que coincida con la forma de x_ifft y recon_target
        mask_expanded = mask.unsqueeze(-1).expand_as(x_ifft)
        
        # Calcular la pérdida de reconstrucción usando la máscara
        diff = (x_ifft - recon_target).abs() * mask_expanded.float()  # Multiplicar por la máscara
        loss_recon = diff.sum() / mask_expanded.sum() if mask_expanded.sum() > 0 else torch.tensor(0.0, device=x.device) # Calcular la media enmascarada

        return x_ifft, loss_recon

class EnhancedLocalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=256, bidirectional=True, dropout=0.12):
        super(EnhancedLocalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.bidirectional = bidirectional
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        B, L, C = x.shape
        
        # Añadir padding para asegurar que L es múltiplo de window_size
        pad_l = (self.window_size - L % self.window_size) % self.window_size
        x = nn.functional.pad(x, (0, 0, 0, pad_l))  # Padding en la dimensión de la longitud
        _, L_padded, _ = x.shape

        # Calcular qkv y reorganizar
        qkv = self.qkv(x).reshape(B, L_padded, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Configurar causalidad
        causal = not self.bidirectional

        # Verificación de dimensión antes de utilizar unfold
        if L_padded < self.window_size:
            raise ValueError("La longitud de la secuencia debe ser al menos igual a window_size.")

        # Manejar ventanas solo si las dimensiones son válidas
        num_windows = (L_padded - self.window_size) // (self.window_size // 2) + 1  # Calcular el número de ventanas

        # Prepárate para almacenar las salidas de atención
        attn_outputs = []

        for i in range(num_windows):
            start_idx = i * (self.window_size // 2)  # Deslizamiento
            end_idx = start_idx + self.window_size
            
            # Asegurarse de que no se sale de límites
            if end_idx <= L_padded:
                q_window = q[..., start_idx:end_idx, :]
                k_window = k[..., start_idx:end_idx, :]
                v_window = v[..., start_idx:end_idx, :]
                
                # Calcular atención para la ventana seleccionada
                attn_output = flash_attn_func(q_window, k_window, v_window, dropout_p=self.dropout, causal=causal)
                attn_outputs.append(attn_output)

        # Concatenar resultados de atención
        attn_output = torch.cat(attn_outputs, dim=2)  # Concatenar por la dimensión donde se acumula la longitud de la ventana
        attn_output = attn_output.reshape(B, L_padded, C)  # Asegurar la forma adecuada
        
        # Aplicar la capa de salida
        attn_output = self.out(attn_output)

        return attn_output[:, :L, :]   
class DeformableConv1d(nn.Module):
    """
    Implementación de una capa de convolución deformable 1D.
    
    Esta capa permite una deformación adaptativa del campo receptivo,
    lo que puede mejorar la capacidad del modelo para capturar características
    en diferentes escalas y posiciones.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DeformableConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding  # Debe ser un entero
        self.stride = stride
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convolución para generar los offsets
        self.offset_conv = nn.Conv1d(
            in_channels,
            2 * kernel_size,  # Para desplazamientos en la dimensión de longitud
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True
        )
        
        # Convolución principal ajustada para recibir C * kernel_size canales
        self.conv = nn.Conv1d(
            in_channels * kernel_size,  # Cambiado de in_channels a in_channels * kernel_size
            out_channels,
            kernel_size=1,  # Kernel size ajustado para operar sobre los canales
            stride=1,       # Stride ajustado
            padding=0,      # Padding ajustado
            dilation=1,     # Dilation ajustado
            bias=bias
        )
        
    def forward(self, x):
        """
        x: [N, C, L]
        """
        # Implementamos autocast para habilitar la precisión mixta
        with autocast(enabled=True):
            # Calcular los offsets
            offsets = self.offset_conv(x)  # [N, 2 * kernel_size, L_out]
            N, _, L_out = offsets.size()
            offsets = offsets.view(N, self.kernel_size, 2, L_out)  # [N, kernel_size, 2, L_out]
            offsets = offsets.permute(0, 3, 1, 2)  # [N, L_out, kernel_size, 2]

            # Preparar la entrada para la interpolación
            x_padded = F.pad(x, (self.padding, self.padding))  # [N, C, L + 2 * padding]

            # Crear mallas para las posiciones
            device = x.device
            dtype = x.dtype

            # Crear una malla de posiciones base
            base_grid = torch.arange(0, x_padded.size(2), device=device, dtype=dtype).unsqueeze(0).unsqueeze(2)  # [1, L_padded, 1]
            base_grid = base_grid.repeat(N, 1, self.kernel_size)  # [N, L_padded, kernel_size]

            # Aplicar los offsets
            grid = base_grid[:, self.padding:x_padded.size(2)-self.padding, :] + offsets[..., 0]  # [N, L_out, kernel_size]

            # Limitar los valores del grid para evitar índices fuera de rango
            grid = grid.clamp(0, x_padded.size(2) - 1)

            # Obtener índices de izquierda y derecha para la interpolación
            left = grid.floor().long()  # [N, L_out, kernel_size]
            right = (left + 1).clamp(max=x_padded.size(2) - 1)  # [N, L_out, kernel_size]
            alpha = grid - left.float()  # [N, L_out, kernel_size]

            # Reshape para gather
            left = left.view(N, -1).unsqueeze(1).expand(-1, self.in_channels, -1)  # [N, C, L_out * kernel_size]
            right = right.view(N, -1).unsqueeze(1).expand(-1, self.in_channels, -1)  # [N, C, L_out * kernel_size]

            # Recoger los valores a la izquierda y a la derecha
            x_left = torch.gather(x_padded, 2, left)  # [N, C, L_out * kernel_size]
            x_right = torch.gather(x_padded, 2, right)  # [N, C, L_out * kernel_size]

            # Reorganizar para obtener [N, C, L_out, kernel_size]
            x_left = x_left.view(N, self.in_channels, L_out, self.kernel_size)
            x_right = x_right.view(N, self.in_channels, L_out, self.kernel_size)

            # Interpolación lineal
            alpha = alpha.view(N, 1, L_out, self.kernel_size)  # [N, 1, L_out, kernel_size]
            x_deform = (1 - alpha) * x_left + alpha * x_right  # [N, C, L_out, kernel_size]

            # Reorganizar para la convolución principal
            x_deform = x_deform.permute(0, 3, 2, 1).contiguous().view(N, self.in_channels * self.kernel_size, L_out)  # [N, C * kernel_size, L_out]

            # Aplicar la convolución principal ajustada
            out = self.conv(x_deform)  # [N, out_channels, L_out]

        return out
class OptimizedGatedConvolution(nn.Module):
    """
    Implementación optimizada de una capa de convolución con puerta (gated convolution).
    
    Esta capa combina una convolución deformable con un mecanismo de puerta,
    permitiendo al modelo aprender a enfocar dinámicamente diferentes partes de la entrada.
    Además, implementa normalización de capa y activación GELU para mejorar el rendimiento.
    """
    def __init__(self, channels, kernel_size, dilation):
        super(OptimizedGatedConvolution, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Calcular padding para 'same' padding
        padding = (kernel_size - 1) * dilation // 2
        
        # Reemplazar la convolución estándar por DeformableConv1d
        self.deform_conv = DeformableConv1d(
            in_channels=channels,
            out_channels=channels * 2,  # Para main y gate
            kernel_size=kernel_size,
            padding=padding,  # Debe ser un entero
            dilation=dilation
        )
        
        # Inicialización de los parámetros
        nn.init.kaiming_normal_(self.deform_conv.conv.weight)
        nn.init.zeros_(self.deform_conv.conv.bias)
        nn.init.kaiming_normal_(self.deform_conv.offset_conv.weight)
        nn.init.zeros_(self.deform_conv.offset_conv.bias)
        
    def forward(self, x):
        # Definimos una función interna para usar con checkpoint
        def conv_function(x):
            # Implementamos autocast para habilitar la precisión mixta
            with autocast(enabled=True):
                # Transponer para que los canales sean la segunda dimensión
                x = x.transpose(1, 2)  # [batch, channels, seq_length]
                
                # Aplicar la convolución deformable y dividir el resultado
                conv_out = self.deform_conv(x)  # [batch, channels*2, L_out]
                main, gate = conv_out.chunk(2, dim=1)
                
                # Aplicar las funciones de activación
                main = F.gelu(main)  # GELU para la salida principal
                gate = torch.sigmoid(gate)  # Sigmoide para la puerta
                
                # Aplicar la puerta y normalizar
                gated_out = main * gate
                
                # Normalización de capa
                mean = gated_out.mean(dim=1, keepdim=True)
                var = gated_out.var(dim=1, keepdim=True, unbiased=False)
                gated_out = (gated_out - mean) / (var + 1e-5).sqrt()
                
                # Volver a transponer para mantener la forma original
                return gated_out.transpose(1, 2)  # [batch, L_out, channels]

        # Usamos checkpoint para ahorrar memoria durante el entrenamiento
        return checkpoint(conv_function, x)

def test_deformable_conv1d():
    batch_size = 2
    in_channels = 4
    out_channels = 8
    seq_length = 16
    kernel_size = 3
    dilation = 1
    stride = 1
    padding = (kernel_size - 1) * dilation // 2

    x = torch.randn(batch_size, in_channels, seq_length)  # [N, C, L]
    deform_conv = DeformableConv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
    out = deform_conv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

test_deformable_conv1d()

def test_optimized_gated_convolution():
    batch_size = 2
    channels = 4
    seq_length = 16
    kernel_size = 3
    dilation = 1

    x = torch.randn(batch_size, seq_length, channels)  # [batch, seq_length, channels]
    gated_conv = OptimizedGatedConvolution(channels, kernel_size, dilation)
    out = gated_conv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

test_optimized_gated_convolution()


class EnhancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(EnhancedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM estándar de PyTorch
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Capa de salida adicional con GELU
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x, hidden=None):
        # Pasar a través del LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Aplicar la capa de salida con conexión residual
        output = self.output_layer(lstm_out) + x
        
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(weight.device),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(weight.device))
        return hidden
class ImprovedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, num_experts, expert_dim, window_size=256, bidirectional=True, dropout=0.12, entropy_weight=0.1, top_k=2, dynamic_k=False):
        super(ImprovedTransformerBlock, self).__init__()
        self.attention = EnhancedLocalAttention(embed_dim, num_heads, window_size, bidirectional)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dilated_conv = OptimizedGatedConvolution(embed_dim, kernel_size=3, dilation=2)
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
        # Utilizar autocast para mantener la precisión mixta
        with torch.cuda.amp.autocast(enabled=True):
            x = x + self.dropout1(self.attention(self.norm1(x)))
            x = x + self.dropout2(self.dilated_conv(self.norm2(x)))
            moe_output, entropy_loss = self.moe(self.norm3(x))
            x = x + self.dropout3(moe_output)
            x = x + self.ff_layer(x)
        return x, entropy_loss



# Clase BidirectionalEncoder actualizada para pasar top_k y dynamic_k a los bloques
class BidirectionalEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length=2048, compression_ratio=0.5, num_layers=4, num_heads=8, ff_hidden_dim=1024, window_size=256, num_experts=2, expert_dim=256, entropy_weight=0.1, top_k=2, dynamic_k=False):
        super(BidirectionalEncoder, self).__init__()
        self.embedding = LiquidEmbedding(vocab_size, embed_dim, max_length, base_compression_ratio=compression_ratio)
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
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Utilizar autocast para mantener la precisión mixta
        with autocast(enabled=True):
            x, recon_loss = self.embedding(x)
            total_entropy_loss = 0
            for layer in self.layers:
                x, entropy_loss = layer(x)
                total_entropy_loss += entropy_loss
            x = self.layer_norm(x)
        return x, recon_loss, total_entropy_loss
class LiquidFoundationModelOptimized(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_layers=4, num_heads=8, ff_hidden_dim=1024,
                 num_experts=4, expert_dim=256, max_length=2048, window_size=256, compression_ratio=0.5, 
                 entropy_weight=0.1, top_k=2, dynamic_k=False, lstm_hidden_size=256, lstm_num_layers=2):
        super(LiquidFoundationModelOptimized, self).__init__()
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
        self.decoder_embedding = LiquidEmbedding(
            vocab_size, 
            embed_dim, 
            max_length, 
            base_compression_ratio=0.5, 
            min_compression_ratio=0.1
        )
        self.decoder_layers = nn.ModuleList([
            ImprovedTransformerBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                ff_hidden_dim=ff_hidden_dim, 
                num_experts=num_experts, 
                expert_dim=expert_dim, 
                window_size=window_size, 
                bidirectional=False, 
                dropout=0.12, 
                entropy_weight=entropy_weight, 
                top_k=top_k, 
                dynamic_k=dynamic_k
            )
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.max_length = max_length
        self.compression_ratio = compression_ratio
        
        # Reemplazar xLSTM con EnhancedLSTM
        self.external_memory = EnhancedLSTM(embed_dim, lstm_hidden_size, num_layers=lstm_num_layers)

    def forward(self, encoder_input_ids, decoder_input_ids):
        with autocast(enabled=True):
            encoder_output, recon_loss_enc, entropy_loss_enc = self.encoder(encoder_input_ids)
            decoder_embeddings, recon_loss_dec = self.decoder_embedding(decoder_input_ids)
            
            # Inicializar el estado oculto del EnhancedLSTM
            batch_size = decoder_embeddings.size(0)
            hidden = self.external_memory.init_hidden(batch_size)
            
            total_entropy_loss_dec = 0
            for layer in self.decoder_layers:
                decoder_embeddings, entropy_loss = layer(decoder_embeddings)
                total_entropy_loss_dec += entropy_loss
                
                # Actualizar la memoria externa con EnhancedLSTM
                decoder_embeddings, hidden = self.external_memory(decoder_embeddings, hidden)
            
            decoder_embeddings = self.layer_norm(decoder_embeddings)
            logits = self.output_layer(decoder_embeddings)
        
        return logits[:, :2048, :], recon_loss_enc, recon_loss_dec, entropy_loss_enc, total_entropy_loss_dec

def prepare_data(max_samples=None, val_size=0.1):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    special_tokens = {'pad_token': '[PAD]', 'eos_token': '<EOS>', 'bos_token': '<BOS>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens)
    print(f"Se agregaron {num_added_toks} tokens especiales al tokenizer.")
    
    # Cambiamos el dataset a "wikitext-2-raw-v1"
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    if max_samples is not None and max_samples < len(dataset['train']):
        dataset['train'] = dataset['train'].select(range(max_samples))
    
    def preprocess(examples):
        combined_texts = [
            f"{tokenizer.bos_token} {text}{tokenizer.eos_token}"
            for text in examples['text'] if text.strip()  # Only process non-empty strings
        ]
        tokens = tokenizer(combined_texts, truncation=True, max_length=2048, padding='max_length')
        decoder_input_ids = [[tokenizer.bos_token_id] + ids[:-1] for ids in tokens['input_ids']]
        tokens['decoder_input_ids'] = decoder_input_ids
        tokens['labels'] = [ids.copy() for ids in tokens['input_ids']]
        # Reemplazar los tokens de padding en labels por -100 para ignorarlos en la pérdida
        tokens['labels'] = [
            [(id_ if id_ != tokenizer.pad_token_id else -100) for id_ in label if id_ is not None]
            for label in tokens['labels']
        ]
        return tokens
    
    tokenized_dataset = dataset['train'].map(preprocess, batched=True, batch_size=1000, remove_columns=dataset['train'].column_names)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'decoder_input_ids', 'labels', 'attention_mask'])
    
    train_val_dataset = tokenized_dataset.train_test_split(test_size=val_size)
    
    return tokenizer, train_val_dataset

def calculate_metrics(model, data_loader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_entropy_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in data_loader:
            encoder_input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with autocast(enabled=True):
                outputs, recon_loss_enc, recon_loss_dec, entropy_loss_enc, entropy_loss_dec = model(encoder_input_ids, decoder_input_ids)
                logits = outputs.reshape(-1, outputs.size(-1))
                labels_flat = labels.reshape(-1)
                
                mask = labels_flat != -100
                logits = logits[mask]
                labels_flat = labels_flat[mask]
                
                loss = criterion(logits, labels_flat) + recon_loss_enc + recon_loss_dec + entropy_loss_enc + entropy_loss_dec
                total_loss += loss.item() * labels_flat.numel()
                total_tokens += labels_flat.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

class OptimizedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=-100, label_smoothing=0.1):
        super(OptimizedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        with torch.set_grad_enabled(self.training):
            num_classes = inputs.size(-1)
            
            chunk_size = 1024
            total_loss = 0
            total_count = 0
            
            for i in range(0, inputs.size(0), chunk_size):
                chunk_inputs = inputs[i:i+chunk_size]
                chunk_targets = targets[i:i+chunk_size]
                
                smoothed_targets = torch.zeros_like(chunk_inputs)
                smoothed_targets.scatter_(1, chunk_targets.unsqueeze(1), 1)
                smoothed_targets.mul_(1 - self.label_smoothing).add_(self.label_smoothing / num_classes)
                
                with torch.cuda.amp.autocast(enabled=True):
                    log_probs = F.log_softmax(chunk_inputs, dim=-1)
                    loss = -smoothed_targets * log_probs
                    
                    loss = loss.sum(-1)
                    pt = torch.exp(-loss)
                    focal_loss = self.alpha * (1-pt)**self.gamma * loss
                    
                    if self.ignore_index >= 0:
                        mask = chunk_targets != self.ignore_index
                        focal_loss = focal_loss[mask]
                    
                    total_loss += focal_loss.sum()
                    total_count += focal_loss.numel()
            
            return total_loss / total_count if total_count > 0 else total_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, num_epochs, accumulation_steps=8, evaluator=None, tokenizer=None, monitor=None):
    writer = SummaryWriter()
    best_val_loss = float('inf')
    patience = 3
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_entropy_loss = 0
        total_batches = 0
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Entrenando Epoch {epoch + 1}")
        
        for batch_idx, batch in loop:
            encoder_input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)

            with autocast(enabled=True):
                outputs, recon_loss_enc, recon_loss_dec, entropy_loss_enc, entropy_loss_dec = model(encoder_input_ids, decoder_input_ids)
                logits = outputs.reshape(-1, outputs.size(-1))
                labels_flat = labels.reshape(-1)
                
                mask = labels_flat != -100
                logits = logits[mask]
                labels_flat = labels_flat[mask]
                
                loss = criterion(logits, labels_flat) + recon_loss_enc + recon_loss_dec + entropy_loss_enc + entropy_loss_dec
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            total_recon_loss += (recon_loss_enc + recon_loss_dec).item() * accumulation_steps
            total_entropy_loss += (entropy_loss_enc + entropy_loss_dec).item() * accumulation_steps
            total_batches += 1

            if monitor and (batch_idx + 1) % 800 == 0:
                for name, activation in monitor.activations.items():
                    writer.add_histogram(f'Activations/{name}', activation.cpu().numpy(), epoch)
                for name, gradient in monitor.gradients.items():
                    writer.add_histogram(f'Gradients/{name}', gradient.cpu().numpy(), epoch)

            if (batch_idx + 1) % 800 == 0:
                avg_train_loss = total_loss / total_batches
                avg_recon_loss = total_recon_loss / total_batches
                avg_entropy_loss = total_entropy_loss / total_batches
                val_loss, val_perplexity = calculate_metrics(model, val_loader, criterion, device, tokenizer)
                
                loop.set_postfix(train_loss=avg_train_loss, recon_loss=avg_recon_loss, entropy_loss=avg_entropy_loss, val_loss=val_loss, val_perplexity=val_perplexity)

            if (batch_idx + 1) % 100 == 0:
                for idx, layer in enumerate(model.encoder.layers):
                    usage_stats = layer.moe.get_expert_usage_stats()
                    print(f"Epoch {epoch}, Batch {batch_idx+1}, Layer {idx} Expert Usage: {usage_stats}")

        scheduler.step()

        avg_train_loss = total_loss / total_batches
        avg_recon_loss = total_recon_loss / total_batches
        avg_entropy_loss = total_entropy_loss / total_batches
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/recon', avg_recon_loss, epoch)
        writer.add_scalar('Loss/entropy', avg_entropy_loss, epoch)

        val_loss, val_perplexity = calculate_metrics(model, val_loader, criterion, device, tokenizer)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Perplexity/val', val_perplexity, epoch)

        if evaluator and tokenizer:
            evaluate_model(model, val_loader, evaluator, tokenizer, device, writer, epoch)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Train Recon Loss: {avg_recon_loss:.4f}")
        print(f"Train Entropy Loss: {avg_entropy_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Perplexity: {val_perplexity:.2f}")

        if val_perplexity < best_val_loss:
            best_val_loss = val_perplexity
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping")
            break

        torch.cuda.empty_cache()

    writer.close()

def calculate_evaluation_metrics(evaluator, questions, answers, labels):
    metrics = evaluator.calculate_metrics(questions, answers, labels)
    coherence = np.mean([evaluator.evaluate_coherence(q, [], a) for q, a in zip(questions, answers)])
    metrics['coherence'] = coherence
    return metrics

def print_metrics(metrics):
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

def evaluate_model(model, val_loader, evaluator, tokenizer, device, writer, epoch):
    model.eval()
    all_questions = []
    all_answers = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            encoder_input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            outputs, _, _, _, _ = model(encoder_input_ids, decoder_input_ids)
            preds = torch.argmax(outputs, dim=-1)
            for i in range(encoder_input_ids.size(0)):
                question = tokenizer.decode(encoder_input_ids[i], skip_special_tokens=True)
                answer = tokenizer.decode(preds[i], skip_special_tokens=True)
                
                label_tokens = [token for token in batch['labels'][i] if token != -100 and token is not None]
                label = tokenizer.decode(label_tokens, skip_special_tokens=True)
                
                all_questions.append(question)
                all_answers.append(answer)
                all_labels.append(label)

    print(f"Longitud de all_questions: {len(all_questions)}")
    print(f"Longitud de all_answers: {len(all_answers)}")
    print(f"Longitud de all_labels: {len(all_labels)}")
    
    metrics = calculate_evaluation_metrics(evaluator, all_questions, all_answers, all_labels)
    
    for metric, value in metrics.items():
        writer.add_scalar(f'Metrics/{metric}', value, epoch)
    
    print_metrics(metrics)

def resize_embeddings(model, tokenizer, new_vocab_size, embed_dim):
    old_embedding = model.encoder.embedding.token_embedding
    new_embedding = nn.Embedding(new_vocab_size, embed_dim)
    
    with torch.no_grad():
        new_embedding.weight.data[:old_embedding.num_embeddings, :] = old_embedding.weight.data
        nn.init.normal_(new_embedding.weight.data[old_embedding.num_embeddings:, :], mean=0.0, std=0.02)
    
    model.encoder.embedding.token_embedding = new_embedding.to(device)
    
    model.decoder_embedding.token_embedding = new_embedding.to(device)
    
    model.output_layer = nn.Linear(embed_dim, new_vocab_size).to(device)
    with torch.no_grad():
        nn.init.normal_(model.output_layer.weight, mean=0.0, std=0.02)
        model.output_layer.bias.data.zero_()
    
    print(f"Capa de salida actualizada: {model.output_layer}")

def main(max_samples=10000):
    tokenizer, train_val_dataset = prepare_data(max_samples)
    
    VOCAB_SIZE = len(tokenizer)
    
    print(f"Tamaño del vocabulario: {VOCAB_SIZE}")

    train_loader = DataLoader(train_val_dataset['train'], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(train_val_dataset['test'], batch_size=BATCH_SIZE, shuffle=False)

    model = LiquidFoundationModelOptimized(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ff_hidden_dim=FF_HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        expert_dim=EXPERT_DIM,
        max_length=MAX_LENGTH,
        window_size=WINDOW_SIZE,
        compression_ratio=COMPRESSION_RATIO,
        entropy_weight=0.25,
        top_k=TOP_K,
        dynamic_k=DYNAMIC_K,
        lstm_hidden_size=128,  # Tamaño del estado oculto del EnhancedLSTM
        lstm_num_layers=2      # Número de capas del EnhancedLSTM
    ).to(device)

    resize_embeddings(model, tokenizer, VOCAB_SIZE, EMBED_DIM)
    print("Se actualizó el tamaño del embedding para tokens especiales sin perder los pesos existentes.")

    print(f"Dimensión de output_layer: {model.output_layer.weight.shape}")

    criterion = OptimizedFocalLoss(ignore_index=-100, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler()

    evaluator = TextEvaluator()

    monitor = ActivationMonitor(model)

    train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        scaler, 
        device, 
        NUM_EPOCHS, 
        ACCUMULATION_STEPS, 
        evaluator=evaluator, 
        tokenizer=tokenizer,
        monitor=monitor
    )

    monitor.remove_hooks()

    return model, tokenizer

from collections import OrderedDict
import hashlib
import numpy as np

class DynamicCacheManager:
    def __init__(self, max_size=1000, base_compression_ratio=0.5, min_compression_ratio=0.1):
        self.cache = OrderedDict()  # Utiliza OrderedDict para mantener el orden de uso
        self.max_size = max_size
        self.base_compression_ratio = base_compression_ratio
        self.min_compression_ratio = min_compression_ratio

    def _generate_key(self, prompt, max_length, beam_width, temperature, top_p, repetition_penalty, max_step_tokens, max_answer_tokens, top_k, num_steps):
        key = f"{prompt}_{max_length}_{beam_width}_{temperature}_{top_p}_{repetition_penalty}_{max_step_tokens}_{max_answer_tokens}_{top_k}_{num_steps}"
        return hashlib.md5(key.encode()).hexdigest()

    def _calculate_complexity(self, data_array):
        fft_result = np.fft.fft(data_array)
        magnitude = np.abs(fft_result)
        complexity = (magnitude > 0.1 * magnitude.max()).mean()
        return complexity

    def _compress_data(self, data):
        if isinstance(data, tuple):
            # Convertir la tupla a una lista para procesarla
            data_list = list(data)
            # Convertir cada elemento de la lista a bytes
            data_bytes = [str(item).encode() for item in data_list]
            # Concatenar todos los bytes
            data_array = np.frombuffer(b''.join(data_bytes), dtype=np.uint8)
        elif isinstance(data, str):
            data_array = np.frombuffer(data.encode(), dtype=np.uint8)
        elif isinstance(data, list):
            data_array = np.array(data)
        else:
            raise ValueError(f"Tipo de dato no soportado para compresión: {type(data)}")

        complexity = self._calculate_complexity(data_array)
        dynamic_compression_ratio = self.base_compression_ratio * (1 - complexity)
        dynamic_compression_ratio = max(self.min_compression_ratio, dynamic_compression_ratio)

        fft_result = np.fft.fft(data_array)
        compressed_size = int(len(fft_result) * dynamic_compression_ratio)
        compressed_fft = fft_result[:compressed_size]
        
        return compressed_fft, dynamic_compression_ratio

    def _decompress_data(self, compressed_data, compression_ratio, original_type):
        full_size = int(len(compressed_data) / compression_ratio)
        reconstructed_fft = np.zeros(full_size, dtype=complex)
        reconstructed_fft[:len(compressed_data)] = compressed_data

        decompressed_array = np.fft.ifft(reconstructed_fft).real.astype(np.uint8)
        decompressed_bytes = decompressed_array.tobytes()

        if original_type == str:
            return decompressed_bytes.decode()
        elif original_type == list:
            return decompressed_array.tolist()
        elif original_type == tuple:
            # Dividir los bytes decomprimidos en sus componentes originales
            components = decompressed_bytes.split(b'\x00')  # Asumimos que '\x00' separa los componentes
            return tuple(eval(comp.decode()) for comp in components if comp)
        else:
            raise ValueError(f"Tipo de dato no soportado para descompresión: {original_type}")

    def get(self, key):
        if key in self.cache:
            # Mover la clave al final para marcarla como recientemente usada
            self.cache.move_to_end(key)
            compressed_data, compression_ratio, original_type = self.cache[key]
            return self._decompress_data(compressed_data, compression_ratio, original_type)
        return None

    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            removed_key, _ = self.cache.popitem(last=False)
            print(f"Eliminada la entrada menos recientemente usada: {removed_key}")
        
        original_type = type(value)
        compressed_data, compression_ratio = self._compress_data(value)
        self.cache[key] = (compressed_data, compression_ratio, original_type)


# Inicializa el DynamicCacheManager con política LRU
dynamic_cache_manager = DynamicCacheManager(max_size=1000)


def unified_generate(model, tokenizer, prompt, device, reasoning=True, max_length=512, beam_width=5, temperature=1.0, top_p=0.9, repetition_penalty=1.2, max_step_tokens=70, max_answer_tokens=30, top_k=50, num_steps=4, max_attempts=4, num_iterations=3, evaluator=None):
    cache_key = dynamic_cache_manager._generate_key(
        prompt, max_length, beam_width, temperature, top_p, repetition_penalty,
        max_step_tokens, max_answer_tokens, top_k, num_steps
    )

    cached_result = dynamic_cache_manager.get(cache_key)
    if cached_result is not None:
        print("Resultado recuperado del caché")
        return cached_result

    model.to(device)
    model.eval()

    best_overall_response = ""
    best_overall_cot_steps = []
    best_overall_coherence = float('-inf')

    for iteration in range(num_iterations):
        print(f"Iteración {iteration + 1}/{num_iterations}")
        
        response, cot_steps, coherence_score = generate_cot(
            model, tokenizer, prompt, device, 
            max_step_tokens=max_step_tokens, 
            max_answer_tokens=max_answer_tokens, 
            temperature=temperature, 
            top_k=top_k, 
            beam_width=beam_width,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_steps=num_steps,
            max_attempts=max_attempts
        )

        print(f"cot_steps recibidos: {cot_steps}")
        if not all(isinstance(step, dict) and "text" in step for step in cot_steps):
            print("Error: cot_steps no contiene diccionarios con la clave 'text'.")
            print(f"cot_steps actual: {cot_steps}")
            continue
        
        print(f"Coherencia de la generación {iteration + 1}: {coherence_score:.4f}")

        if coherence_score > best_overall_coherence:
            best_overall_response = response
            best_overall_cot_steps = cot_steps
            best_overall_coherence = coherence_score

        prompt = f"{prompt}\nRefinamiento {iteration + 1}: {response}\n"

    if evaluator:
        coherence = evaluator.evaluate_coherence(prompt, best_overall_cot_steps, best_overall_response)
        print(f"Puntuación de coherencia final: {coherence:.4f}")

    result = (best_overall_response, best_overall_cot_steps, best_overall_coherence)
    
    # Almacena el resultado comprimido en el caché
    dynamic_cache_manager.set(cache_key, result)

    return result
def generate_cot(model, tokenizer, prompt, device, max_step_tokens=70, max_answer_tokens=30, temperature=0.7, top_k=50, beam_width=5, top_p=0.9, repetition_penalty=1.2, num_steps=4, max_attempts=4):
    model.to(device)
    model.eval()
    best_response = ""
    best_cot_steps = []
    best_coherence_score = float('-inf')

    for attempt in range(max_attempts):
        try:
            with torch.no_grad(), autocast(enabled=True):
                encoder_input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                decoder_input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
                cot_steps = []

                for step in range(num_steps):
                    step_prompt = f"Paso {step + 1}: "
                    step_output = generate_step(model, tokenizer, encoder_input_ids, decoder_input_ids, max_step_tokens, temperature, top_k, beam_width, top_p, repetition_penalty, step_prompt)
                    
                    if step_output.strip():
                        step_analysis = analyze_step(step_output)
                        
                        cot_steps.append({
                            "step": f"Paso {step + 1}",
                            "text": step_output,
                            "analysis": step_analysis
                        })
                        
                        temperature, top_k, top_p = adjust_parameters(step_analysis, temperature, top_k, top_p)
                    else:
                        print(f"El paso {step + 1} generó un texto vacío. Reintentando...")
                        continue

                    decoder_input_ids = tokenizer.encode(" ".join([step["text"] for step in cot_steps]), return_tensors="pt", add_special_tokens=False).to(device)

                final_prompt = "Respuesta final: "
                response = generate_step(model, tokenizer, encoder_input_ids, decoder_input_ids, max_answer_tokens, temperature, top_k, beam_width, top_p, repetition_penalty, final_prompt)

                coherence_score = calculate_coherence(prompt, cot_steps, response)

                if coherence_score > best_coherence_score:
                    best_response = response
                    best_cot_steps = cot_steps
                    best_coherence_score = coherence_score

        except RuntimeError as e:
            print(f"Error durante la generación (intento {attempt + 1}): {e}")
            torch.cuda.empty_cache()
            continue

    print(f"GPU utilizada: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU usada: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    return best_response, best_cot_steps, best_coherence_score

def generate_step(model, tokenizer, encoder_input_ids, decoder_input_ids, max_tokens, temperature, top_k, beam_width, top_p, repetition_penalty, step_prompt):
    step_input_ids = tokenizer.encode(step_prompt, add_special_tokens=False, return_tensors="pt").to(device)
    decoder_input_ids = torch.cat([decoder_input_ids, step_input_ids], dim=-1)
    
    sequences = decoder_input_ids.unsqueeze(1).repeat(1, beam_width, 1)  # [batch, beam_width, seq_len]
    scores = torch.zeros(sequences.size(0), beam_width).to(device)  # [batch, beam_width]
    
    for _ in range(max_tokens):
        # Aplanar las secuencias para procesarlas en batch
        flat_sequences = sequences.view(-1, sequences.size(-1))  # [batch * beam_width, seq_len]
        
        with autocast(enabled=True):
            outputs, _, _, _, _ = model(encoder_input_ids, flat_sequences)
            logits = outputs[:, -1, :] / temperature  # [batch * beam_width, vocab_size]
            logits = top_p_sampling(logits, p=top_p)
            logits = apply_repetition_penalty(logits, flat_sequences, repetition_penalty)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, beam_width, dim=-1)  # [batch * beam_width, beam_width]
        
        # Calcular nuevas puntuaciones
        new_scores = scores.unsqueeze(2) - torch.log(topk_probs)
        new_scores = new_scores.view(-1, beam_width * beam_width)
        
        # Seleccionar las mejores secuencias
        top_scores, top_indices = torch.topk(new_scores, beam_width, dim=-1, largest=False)
        beam_indices = top_indices // beam_width
        token_indices = top_indices % beam_width
        
        # Actualizar secuencias
        sequences = sequences[torch.arange(sequences.size(0)).unsqueeze(1), beam_indices]
        sequences = torch.cat([sequences, topk_indices.view(-1, beam_width)[torch.arange(sequences.size(0)).unsqueeze(1), token_indices].unsqueeze(-1)], dim=-1)
        
        scores = top_scores
        
        # Terminar si todas las secuencias han terminado
        if torch.all(sequences[:, :, -1] == tokenizer.eos_token_id):
            break
    
    # Seleccionar la secuencia con la mejor puntuación
    best_sequences = sequences[torch.arange(sequences.size(0)), scores.argmin(dim=-1)]
    return tokenizer.decode(best_sequences[0], skip_special_tokens=True)

def analyze_step(step_text):
    math_operations = len(re.findall(r'[\+\-\*/\^]', step_text))
    numbers = len(re.findall(r'\d+', step_text))
    words = len(step_text.split())
    
    operation_density = math_operations / words if words > 0 else 0
    number_density = numbers / words if words > 0 else 0
    
    return {
        "math_operations": math_operations,
        "numbers": numbers,
        "words": words,
        "operation_density": operation_density,
        "number_density": number_density
    }

def adjust_parameters(step_analysis, temperature, top_k, top_p):
    if step_analysis["operation_density"] < 0.1:
        temperature *= 0.9
        top_k = max(top_k - 5, 10)
    elif step_analysis["number_density"] < 0.2:
        top_p *= 0.95
    
    return temperature, top_k, top_p

def top_p_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits

def apply_repetition_penalty(logits, sequence, penalty=1.2):
    for token_id in set(sequence[0].tolist()):
        logits[:, token_id] /= penalty
    return logits

def calculate_coherence(question, cot_steps, response):
    try:
        full_text = question + " " + " ".join([step["text"] for step in cot_steps]) + " " + response

        if not full_text.strip():
            return float('-inf')

        coherence_score = 0.0
        # Puedes implementar una evaluación más sofisticada si lo deseas
        return coherence_score

    except (TypeError, KeyError) as e:
        print(f"Error en calculate_coherence: {e}")
        print(f"question: {question}")
        print(f"cot_steps: {cot_steps}")
        print(f"response: {response}")
        return float('-inf')

def get_model_device(model):
    return next(model.parameters()).device

def analyze_token_transformations(model, tokenizer, prompt):
    print("Analizando transformaciones de tokens en LiquidFoundationModelOptimized")
    print(f"Prompt: {prompt}")

    device = get_model_device(model)
    print(f"Dispositivo del modelo: {device}")

    # Tokenización de entrada
    encoder_input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"\n1. Tokenización de entrada:")
    print(f"   Forma de entrada codificada: {encoder_input_ids.shape}")
    print(f"   Entrada codificada: {encoder_input_ids}")
    print(f"   Entrada decodificada: {tokenizer.decode(encoder_input_ids[0])}")

    # Preparar entradas del modelo
    encoder_input_ids = encoder_input_ids.to(device)
    decoder_input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)

    model.eval()
    with torch.no_grad():
        # Embedding de entrada
        encoder_output, recon_loss_enc, entropy_loss_enc = model.encoder(encoder_input_ids)
        print(f"\n2. Embedding de entrada:")
        print(f"   Forma del embedding: {encoder_output.shape}")
        print(f"   Estadísticas del embedding - Media: {encoder_output.mean().item():.4f}, Desv. Est.: {encoder_output.std().item():.4f}")

        # Generar tokens y analizar transformaciones intermedias
        for i in range(5):  # Generar 5 tokens
            # Embedding del decoder
            decoder_embeds, recon_loss_dec = model.decoder_embedding(decoder_input_ids)
            print(f"\n3. Embedding del decoder (paso {i+1}):")
            print(f"   Forma del embedding del decoder: {decoder_embeds.shape}")
            print(f"   Estadísticas del embedding del decoder - Media: {decoder_embeds.mean().item():.4f}, Desv. Est.: {decoder_embeds.std().item():.4f}")

            # Salida del modelo
            outputs, _, _, _, _ = model(encoder_input_ids, decoder_input_ids)
            print(f"\n4. Salida del modelo (paso {i+1}):")
            print(f"   Forma de la salida: {outputs.shape}")
            print(f"   Estadísticas de la salida - Media: {outputs.mean().item():.4f}, Desv. Est.: {outputs.std().item():.4f}")

            # Análisis de la distribución de probabilidad
            probs = F.softmax(outputs[:, -1, :], dim=-1)
            top_probs, top_indices = torch.topk(probs, 5)
            print(f"\n5. Top 5 tokens probables (paso {i+1}):")
            for prob, idx in zip(top_probs[0], top_indices[0]):
                print(f"   Token: {tokenizer.decode([idx.item()])}, Probabilidad: {prob.item():.4f}")

            # Seleccionar el siguiente token
            next_token = outputs[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)

            print(f"\n6. Token seleccionado (paso {i+1}):")
            print(f"   Token: {tokenizer.decode(next_token[0])}")

    # Tokenización de salida
    generated_output = decoder_input_ids[0][1:]  # Eliminar token BOS
    print(f"\n7. Tokenización de salida:")
    print(f"   Forma de la salida generada: {generated_output.shape}")
    print(f"   Salida generada: {generated_output}")
    print(f"   Salida decodificada: {tokenizer.decode(generated_output)}")

    print("\nAnálisis de transformación de tokens completado.")

if __name__ == "__main__":
    model, tokenizer = main(max_samples=100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device)
    print(f"Total parameters: {total_params}")
    
    # Ejemplo de uso
    prompt = "Escribe un cuento corto sobre una aventura en el bosque."
    analyze_token_transformations(model, tokenizer, prompt)
    cot_prompts = [
        "Instrucción: Genera una historia sobre un héroe que salva su aldea.\nEntrada: \nRazonamiento:",
        "Instrucción: Describe los pasos para preparar una taza de té.\nEntrada: \nRazonamiento:",
    ]

    evaluator = TextEvaluator()

    print("Generando soluciones con Chain of Thought:\n")
    
    for question in cot_prompts:
        response, cot_steps, coherence_score = unified_generate(
            model, tokenizer, question, device, 
            reasoning=True,
            max_step_tokens=70, 
            max_answer_tokens=30, 
            temperature=0.7, 
            top_k=50, 
            num_steps=3, 
            max_attempts=4,
            beam_width=5,
            top_p=0.9,
            repetition_penalty=0.8,
            num_iterations=2,
            evaluator=evaluator
        )
        print(f"Pregunta:\n{question}\nRespuesta:\n{response}")
        print("Pasos de razonamiento:")
        for step in cot_steps:
            print(f"{step['step']}: {step['text']}")
        print(f"Puntuación de coherencia: {coherence_score:.4f}")
        print(f"Total de tokens generados: {sum(len(step['text'].split()) for step in cot_steps) + len(response.split())}\n{'-'*50}\n")
