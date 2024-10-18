"""
LiquidFoundationModel: Un modelo de lenguaje avanzado con compresión dinámica y atención local mejorada.

Este script implementa un modelo de lenguaje que utiliza técnicas avanzadas como:
- Embeddings líquidos con compresión dinámica
- Atención local mejorada
- Convoluciones deformables
- Mixture of Experts (MoE)
- LSTM mejorado para memoria externa

El modelo está diseñado para manejar eficientemente secuencias largas y adaptar
su comportamiento basándose en la complejidad de la entrada.

Autor: [Kitsunp]
Fecha: [17/10/2024]
Versión: 1.0
"""

# Importaciones
from collections import OrderedDict
import hashlib
import hashlib
import math
import os
import re

from datasets import load_dataset
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2Tokenizer,
)  # Toolkit para procesamiento de lenguaje natural
nltk.download('all')  # Descarga todos los recursos de NLTK
  # Función para checkpointing en redes neuronales

# Configuración global
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Determina si usar GPU o CPU
print(f"Usando dispositivo: {device}")  # Imprime el dispositivo que se está utilizando

# Constantes globales
EMBED_DIM = 512          # Dimensión de los embeddings
NUM_LAYERS = 12          # Número de capas en el modelo
NUM_HEADS = 8            # Número de cabezas de atención
FF_HIDDEN_DIM = 1024     # Dimensión oculta de la capa feed-forward
NUM_EXPERTS = 8          # Número de expertos en la capa MoE
EXPERT_DIM = 512         # Dimensión de cada experto
MAX_LENGTH = 8192        # Longitud máxima de secuencia
WINDOW_SIZE = 2048       # Tamaño de la ventana para atención local
COMPRESSION_RATIO = 0.5  # Ratio de compresión para embeddings líquidos
BATCH_SIZE = 4           # Tamaño del batch para entrenamiento
NUM_EPOCHS = 20          # Número de épocas de entrenamiento
ACCUMULATION_STEPS = 6   # Pasos de acumulación de gradientes
TOP_K = 4                # Número inicial de expertos a preseleccionar
DYNAMIC_K = True         # Activar el ajuste dinámico de K

# Clases y Funciones

class TextEvaluator:
    """
    Clase para evaluar la calidad del texto generado utilizando métricas como BLEU, ROUGE y F1.
    
    Esta clase proporciona métodos para calcular varias métricas de evaluación de texto,
    incluyendo BLEU, ROUGE-1, ROUGE-2, ROUGE-L, F1 y una medida simple de coherencia.
    """

    def __init__(self):
        """
        Inicializa el TextEvaluator con un scorer ROUGE y una función de suavizado para BLEU.
        """
        # Inicializar el scorer ROUGE para calcular métricas ROUGE
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        # Inicializar la función de suavizado para el cálculo de BLEU
        self.smoothing = SmoothingFunction()

    def calculate_metrics(self, questions, answers, labels):
        """
        Calcula múltiples métricas de evaluación para un conjunto de preguntas, respuestas y etiquetas.

        Args:
            questions (list): Lista de preguntas o prompts.
            answers (list): Lista de respuestas generadas por el modelo.
            labels (list): Lista de respuestas correctas o esperadas.

        Returns:
            dict: Un diccionario con las métricas calculadas (BLEU, ROUGE-1, ROUGE-2, ROUGE-L, F1).
        """
        bleu_scores = []  # Lista para almacenar los puntajes BLEU
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}  # Diccionario para almacenar puntajes ROUGE
        f1_scores_list = []  # Lista para almacenar puntajes F1
        
        # Asegurar que todas las listas tengan la misma longitud
        min_length = min(len(questions), len(answers), len(labels))  # Encuentra la longitud mínima entre las listas
        questions = questions[:min_length]  # Recorta la lista de preguntas si es necesario
        answers = answers[:min_length]  # Recorta la lista de respuestas si es necesario
        labels = labels[:min_length]  # Recorta la lista de etiquetas si es necesario
        
        # Iterar sobre cada trío de pregunta, respuesta y etiqueta
        for q, a, l in zip(questions, answers, labels):
            # Tokenización consistente
            a_tokens = word_tokenize(a.lower())  # Tokeniza y convierte a minúsculas la respuesta
            l_tokens = word_tokenize(l.lower())  # Tokeniza y convierte a minúsculas la etiqueta
            
            # Calcular BLEU
            bleu = sentence_bleu([l_tokens], a_tokens, smoothing_function=self.smoothing.method1)
            bleu_scores.append(bleu)  # Añade el puntaje BLEU a la lista
            
            # Calcular ROUGE
            rouge_result = self.scorer.score(l, a)  # Calcula los puntajes ROUGE
            rouge_scores['rouge1'].append(rouge_result['rouge1'].fmeasure)  # Añade el puntaje ROUGE-1
            rouge_scores['rouge2'].append(rouge_result['rouge2'].fmeasure)  # Añade el puntaje ROUGE-2
            rouge_scores['rougeL'].append(rouge_result['rougeL'].fmeasure)  # Añade el puntaje ROUGE-L
            
            # Calcular F1 (usando conjuntos para manejar longitudes diferentes)
            f1 = f1_score(
                [1 if token in l_tokens else 0 for token in set(a_tokens + l_tokens)],  # Crea una lista binaria para tokens en la etiqueta
                [1 if token in a_tokens else 0 for token in set(a_tokens + l_tokens)],  # Crea una lista binaria para tokens en la respuesta
                average='binary',  # Usa promedio binario para F1
                zero_division=1  # Maneja la división por cero
            )
            f1_scores_list.append(f1)  # Añade el puntaje F1 a la lista
        
        # Calcular promedios de las métricas
        metrics = {
            'bleu': np.mean(bleu_scores),  # Calcula el promedio de los puntajes BLEU
            'rouge1': np.mean(rouge_scores['rouge1']),  # Calcula el promedio de los puntajes ROUGE-1
            'rouge2': np.mean(rouge_scores['rouge2']),  # Calcula el promedio de los puntajes ROUGE-2
            'rougeL': np.mean(rouge_scores['rougeL']),  # Calcula el promedio de los puntajes ROUGE-L
            'f1': np.mean(f1_scores_list)  # Calcula el promedio de los puntajes F1
        }
        return metrics  # Devuelve el diccionario con todas las métricas calculadas

    def evaluate_coherence(self, question, cot_steps, answer):
        """
        Evalúa la coherencia entre la pregunta, los pasos de razonamiento y la respuesta.

        Esta es una implementación simple basada en la similitud de palabras. Para una
        evaluación más sofisticada, se podrían implementar técnicas de NLP más avanzadas.

        Args:
            question (str): La pregunta o prompt inicial.
            cot_steps (list): Lista de pasos de razonamiento (Chain of Thought).
            answer (str): La respuesta final generada.

        Returns:
            float: Puntuación de coherencia entre 0 y 1.
        """
        # Combinar todo el texto para análisis
        all_text = question + " " + " ".join([step["text"] for step in cot_steps]) + " " + answer
        words = set(word_tokenize(all_text.lower()))  # Tokeniza y convierte a conjunto todas las palabras únicas
        
        question_words = set(word_tokenize(question.lower()))  # Tokeniza y convierte a conjunto las palabras de la pregunta
        answer_words = set(word_tokenize(answer.lower()))  # Tokeniza y convierte a conjunto las palabras de la respuesta
        
        # Calcular superposición de palabras entre pregunta y respuesta
        overlap = len(question_words.intersection(answer_words))  # Cuenta las palabras comunes entre pregunta y respuesta
        total_words = len(words)  # Cuenta el total de palabras únicas
        
        # Calcular puntuación de coherencia
        coherence_score = overlap / total_words if total_words > 0 else 0  # Calcula la proporción de palabras comunes
        return coherence_score  # Devuelve la puntuación de coherencia

class ActivationMonitor:
    """
    Clase para monitorear activaciones y gradientes en un modelo neural.

    Esta clase implementa hooks para registrar y almacenar las activaciones
    y gradientes de capas específicas del modelo durante el entrenamiento o la inferencia.
    """

    def __init__(self, model):
        """
        Inicializa el ActivationMonitor y registra los hooks en el modelo.

        Args:
            model (nn.Module): El modelo de PyTorch a monitorear.
        """
        self.handles = []  # Lista para almacenar los manejadores de los hooks
        self.activations = {}  # Diccionario para almacenar las activaciones
        self.gradients = {}  # Diccionario para almacenar los gradientes
        self.register_hooks(model)  # Registra los hooks en el modelo

    def register_hooks(self, model):
        """
        Registra hooks de forward y backward en las capas lineales y convolucionales del modelo.

        Args:
            model (nn.Module): El modelo de PyTorch en el que se registrarán los hooks.
        """
        for name, module in model.named_modules():
            # Verifica si el módulo es una capa lineal o convolucional
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
                # Registra un hook para guardar las activaciones durante el forward pass
                handle = module.register_forward_hook(self.save_activation(name))
                # Registra un hook para guardar los gradientes durante el backward pass
                handle_grad = module.register_backward_hook(self.save_gradient(name))
                # Almacena los manejadores de los hooks
                self.handles.append(handle)
                self.handles.append(handle_grad)

    def save_activation(self, name):
        """
        Crea una función de hook para guardar las activaciones de una capa.

        Args:
            name (str): Nombre de la capa.

        Returns:
            function: Función de hook para guardar activaciones.
        """
        def hook(module, input, output):
            # Guarda una copia desvinculada de las activaciones
            self.activations[name] = output.detach()
            # Verifica si hay valores no finitos en las activaciones
            if not torch.isfinite(output).all():
                print(f"Activaciones no finitas en {name}")
        return hook

    def save_gradient(self, name):
        """
        Crea una función de hook para guardar los gradientes de una capa.

        Args:
            name (str): Nombre de la capa.

        Returns:
            function: Función de hook para guardar gradientes.
        """
        def hook(module, grad_input, grad_output):
            # Guarda una copia desvinculada de los gradientes
            self.gradients[name] = grad_output[0].detach()
            # Verifica si hay valores no finitos en los gradientes
            if not torch.isfinite(grad_output[0]).all():
                print(f"Gradientes no finitos en {name}")
        return hook

    def remove_hooks(self):
        """
        Elimina todos los hooks registrados para liberar recursos.
        """
        for handle in self.handles:
            handle.remove()  # Remueve cada hook registrado
class MoELayer(nn.Module):
    """
    Implementa una capa de Mixture of Experts (MoE) con selección dinámica de expertos.

    Esta capa permite el uso eficiente de múltiples "expertos" (sub-redes) especializados,
    seleccionando dinámicamente los expertos más relevantes para cada entrada.
    """

    def __init__(self, input_dim, hidden_dim, num_experts, expert_dim, dropout=0.15, 
                 entropy_weight=0.1, top_k=2, dynamic_k=False, max_usage_ratio=0.3):
        """
        Inicializa la capa MoE.

        Args:
            input_dim (int): Dimensión de entrada.
            hidden_dim (int): Dimensión oculta de los expertos.
            num_experts (int): Número total de expertos.
            expert_dim (int): Dimensión de salida de cada experto.
            dropout (float): Tasa de dropout para regularización.
            entropy_weight (float): Peso para la regularización de entropía.
            top_k (int): Número inicial de expertos a seleccionar.
            dynamic_k (bool): Si se debe ajustar dinámicamente el número de expertos.
            max_usage_ratio (float): Ratio máximo de uso para cada experto.
        """
        super(MoELayer, self).__init__()
        self.num_experts = num_experts  # Almacena el número total de expertos
        self.top_k = top_k  # Almacena el número inicial de expertos a seleccionar
        self.dynamic_k = dynamic_k  # Indica si se debe ajustar dinámicamente K
        
        # Crea una lista de expertos, cada uno es una capa lineal
        self.experts = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_experts)])
        
        # Capa de puerta para seleccionar expertos
        self.gate = nn.Linear(input_dim, num_experts)
        
        # Capa de dropout para regularización
        self.dropout = nn.Dropout(dropout)
        
        # Parámetros adicionales para el control de la capa
        self.entropy_weight = entropy_weight  # Peso para la regularización de entropía
        self.max_usage_ratio = max_usage_ratio  # Ratio máximo de uso para cada experto
        self.expert_usage_counter = None  # Contador de uso de expertos, se inicializará en el forward pass

    def forward(self, x):
        """
        Realiza la pasada hacia adelante de la capa MoE.

        Args:
            x (Tensor): Tensor de entrada de forma [batch_size, seq_length, input_dim].

        Returns:
            Tuple[Tensor, Tensor]: 
                - Salida procesada de forma [batch_size, seq_length, hidden_dim].
                - Pérdida de entropía y penalización por uso excesivo.
        """
        batch_size, seq_length, input_dim = x.size()
        x_flat = x.view(-1, input_dim)  # Aplana la entrada para procesamiento
        
        # Calcula las probabilidades de la puerta
        gate_logits = self.gate(x_flat)  # Aplica la capa de puerta
        gate_probs = F.softmax(gate_logits, dim=-1)  # Aplica softmax para obtener probabilidades

        # Calcula la pérdida de entropía para regularización
        entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-10), dim=-1).mean()
        entropy_loss = self.entropy_weight * entropy

        # Ajusta dinámicamente K si está activado
        if self.dynamic_k:
            complexity = entropy.detach().item()  # Usa la entropía como medida de complejidad
            K = max(1, min(self.num_experts, int(self.top_k * (1 + complexity))))  # Ajusta K basado en la complejidad
        else:
            K = self.top_k  # Usa el K fijo si no es dinámico

        # Selecciona los top-K expertos
        topk_probs, topk_indices = torch.topk(gate_probs, K, dim=-1)

        # Inicializa o reinicia el contador de uso de expertos
        if self.expert_usage_counter is None:
            self.expert_usage_counter = torch.zeros(self.num_experts, device=x.device)
        else:
            self.expert_usage_counter = self.expert_usage_counter.to(x.device)

        # Prepara tensor para almacenar las salidas de los expertos
        expert_outputs = torch.zeros(batch_size * seq_length, self.experts[0].out_features, device=x.device, dtype=x.dtype)

        # Procesa entradas con los expertos seleccionados
        for k in range(K):
            expert_idx = topk_indices[:, k]
            mask = torch.arange(x_flat.size(0), device=x.device).unsqueeze(1) == expert_idx.unsqueeze(1)
            mask = mask.any(dim=1)
            selected_x = x_flat[mask]

            if selected_x.size(0) > 0:
                # Selecciona el experto de manera eficiente
                unique_experts = expert_idx[mask].unique()
                for expert in unique_experts:
                    expert_mask = expert_idx[mask] == expert
                    inputs = selected_x[expert_mask]
                    output = self.dropout(self.experts[expert](inputs))
                    expert_outputs[mask][expert_mask] += output * topk_probs[:, k][mask][expert_mask].unsqueeze(1)
                    # Actualiza el contador de uso de expertos
                    self.expert_usage_counter[expert] += inputs.size(0)

        # Calcula la penalización por uso excesivo
        usage_ratios = self.expert_usage_counter / (batch_size * seq_length)
        overuse_penalty = torch.sum(F.relu(usage_ratios - self.max_usage_ratio))

        # Reshape la salida a la forma original
        output = expert_outputs.view(batch_size, seq_length, -1)

        return output, entropy_loss + overuse_penalty

    def get_expert_usage_stats(self):
        """
        Obtiene estadísticas de uso de los expertos.

        Returns:
            list: Lista de porcentajes de uso para cada experto.
        """
        if self.expert_usage_counter is None:
            return None
        total_usage = self.expert_usage_counter.sum().item()
        if total_usage == 0:
            return [0.0] * self.num_experts
        usage_percentages = (self.expert_usage_counter / total_usage * 100).tolist()
        return usage_percentages

def next_power_of_two(x):
    """
    Calcula la siguiente potencia de 2 mayor o igual a x.

    Esta función es útil para ajustar dimensiones a potencias de 2,
    lo cual es beneficioso para ciertas operaciones como FFT.

    Args:
        x (int): Número para el cual se calculará la siguiente potencia de 2.

    Returns:
        int: La siguiente potencia de 2 mayor o igual a x.
    """
    return 1 if x == 0 else 2**(x - 1).bit_length()
class LiquidEmbedding(nn.Module):
    """
    Implementa un esquema de embedding adaptativo que combina embeddings de tokens y posiciones,
    y aplica una compresión dinámica basada en la complejidad de la secuencia de entrada.

    Esta clase utiliza convoluciones y la Transformada Rápida de Fourier (FFT) para procesar
    y comprimir los embeddings, permitiendo una representación más eficiente de secuencias largas.
    """

    def __init__(self, vocab_size, embed_dim, max_length=2048, base_compression_ratio=0.5, min_compression_ratio=0.1):
        """
        Inicializa el módulo LiquidEmbedding.

        Args:
            vocab_size (int): Tamaño del vocabulario.
            embed_dim (int): Dimensión de los embeddings.
            max_length (int): Longitud máxima de la secuencia.
            base_compression_ratio (float): Ratio de compresión base para la FFT.
            min_compression_ratio (float): Ratio de compresión mínimo para evitar una compresión excesiva.
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
        
        # Capa de proyección lineal
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

        # Genera embeddings de posición
        positions = torch.arange(0, seq_length, device=device).unsqueeze(0).expand(batch_size, seq_length)
        
        # Combina embeddings de token y posición
        x = self.token_embedding(x) + self.position_embedding(positions)

        # Aplica capas convolucionales
        x = self.conv1(x.permute(0, 2, 1))
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)

        # Añade padding para que seq_length sea potencia de 2 (requerido para FFT eficiente)
        padded_seq_length = next_power_of_two(seq_length)
        padding = padded_seq_length - seq_length
        if padding > 0:
            x = F.pad(x, (0, 0, 0, padding))  # Agrega padding al final de la secuencia

        # Aplica FFT
        x_fft = torch.fft.fft(x, dim=1)

        # Calcula la complejidad de la secuencia
        magnitude = torch.abs(x_fft)
        complexity = (magnitude > 0.1 * magnitude.max(dim=1, keepdim=True).values).float().mean(dim=(1, 2))
        
        # Calcula el ratio de compresión dinámico
        dynamic_compression_ratio = self.base_compression_ratio * (1 - complexity)
        dynamic_compression_ratio = torch.clamp(dynamic_compression_ratio, min=self.min_compression_ratio, max=1.0)

        # Calcula N para cada muestra en el batch
        N = (dynamic_compression_ratio * seq_length).long()
        N = torch.clamp(N, min=1, max=seq_length)  # Asegura que N no sea mayor que seq_length

        max_N = N.max().item()

        # Comprime x_fft para cada muestra
        x_fft_compressed = torch.zeros((batch_size, max_N, x_fft.size(-1)), dtype=torch.complex64, device=device)
        mask = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=device)
        
        for i, n in enumerate(N):
            n = n.item()
            x_fft_compressed[i, :n, :] = x_fft[i, :n, :]
            mask[i, :n] = 1  # Marca las posiciones válidas

        # Reconstrucción usando IFFT
        x_ifft = torch.fft.ifft(x_fft_compressed, n=seq_length, dim=1).real
        x_ifft = self.proj(x_ifft)
        x_ifft = x_ifft.type_as(x)  # Mantiene el dtype original

        recon_target = torch.fft.ifft(x_fft_compressed, n=seq_length, dim=1).real
        recon_target = self.proj(recon_target).type_as(x)
        
        # Expande la máscara para que coincida con la forma de x_ifft y recon_target
        mask_expanded = mask.unsqueeze(-1).expand_as(x_ifft)
        
        # Calcula la pérdida de reconstrucción usando la máscara
        diff = (x_ifft - recon_target).abs() * mask_expanded.float()  # Multiplica por la máscara
        loss_recon = diff.sum() / mask_expanded.sum() if mask_expanded.sum() > 0 else torch.tensor(0.0, device=x.device)

        return x_ifft, loss_recon
class EnhancedLocalAttention(nn.Module):
    """
    Implementa un mecanismo de atención local mejorado con soporte para atención bidireccional y unidireccional.
    """

    def __init__(self, embed_dim, num_heads, window_size=256, bidirectional=True, dropout=0.12):
        """
        Inicializa el módulo de atención local mejorada.

        Args:
            embed_dim (int): Dimensión del embedding.
            num_heads (int): Número de cabezas de atención.
            window_size (int): Tamaño de la ventana de atención local.
            bidirectional (bool): Si es True, usa atención bidireccional; si es False, usa atención unidireccional.
            dropout (float): Tasa de dropout para regularización.
        """
        super(EnhancedLocalAttention, self).__init__()
        self.embed_dim = embed_dim  # Almacena la dimensión del embedding
        self.num_heads = num_heads  # Almacena el número de cabezas de atención
        self.window_size = window_size  # Almacena el tamaño de la ventana de atención
        self.bidirectional = bidirectional  # Indica si la atención es bidireccional o no
        self.head_dim = embed_dim // num_heads  # Calcula la dimensión de cada cabeza de atención
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"  # Verifica que embed_dim sea divisible por num_heads

        # Capa lineal para proyectar las entradas a queries, keys y values
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        # Capa lineal de salida
        self.out = nn.Linear(embed_dim, embed_dim)
        # Tasa de dropout
        self.dropout = dropout

    def forward(self, x):
        """
        Realiza la pasada hacia adelante del módulo de atención local mejorada.

        Args:
            x (Tensor): Tensor de entrada de forma [batch_size, seq_length, embed_dim].

        Returns:
            Tensor: Salida procesada de forma [batch_size, seq_length, embed_dim].
        """
        B, L, C = x.shape  # B: tamaño del batch, L: longitud de la secuencia, C: dimensión del embedding
        
        # Añade padding para asegurar que L es múltiplo de window_size
        pad_l = (self.window_size - L % self.window_size) % self.window_size
        x = nn.functional.pad(x, (0, 0, 0, pad_l))  # Padding en la dimensión de la longitud
        _, L_padded, _ = x.shape  # Obtiene la nueva longitud después del padding

        # Calcula qkv y reorganiza
        qkv = self.qkv(x).reshape(B, L_padded, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Separa queries, keys y values

        # Configura la causalidad (atención unidireccional o bidireccional)
        causal = not self.bidirectional

        # Verifica que la longitud de la secuencia sea al menos igual a window_size
        if L_padded < self.window_size:
            raise ValueError("La longitud de la secuencia debe ser al menos igual a window_size.")

        # Calcula el número de ventanas
        num_windows = (L_padded - self.window_size) // (self.window_size // 2) + 1

        # Prepara lista para almacenar las salidas de atención
        attn_outputs = []

        # Procesa cada ventana
        for i in range(num_windows):
            start_idx = i * (self.window_size // 2)  # Índice de inicio de la ventana
            end_idx = start_idx + self.window_size  # Índice de fin de la ventana
            
            # Asegura que no se salga de los límites
            if end_idx <= L_padded:
                q_window = q[..., start_idx:end_idx, :]  # Extrae queries para la ventana actual
                k_window = k[..., start_idx:end_idx, :]  # Extrae keys para la ventana actual
                v_window = v[..., start_idx:end_idx, :]  # Extrae values para la ventana actual
                
                # Calcula atención para la ventana seleccionada usando flash attention
                attn_output = flash_attn_func(q_window, k_window, v_window, dropout_p=self.dropout, causal=causal)
                attn_outputs.append(attn_output)

        # Concatena los resultados de atención de todas las ventanas
        attn_output = torch.cat(attn_outputs, dim=2)  # Concatena por la dimensión de la longitud de la ventana
        attn_output = attn_output.reshape(B, L_padded, C)  # Reshapea para asegurar la forma adecuada
        
        # Aplica la capa de salida
        attn_output = self.out(attn_output)

        # Devuelve la salida, eliminando el padding añadido al principio
        return attn_output[:, :L, :]  
class DeformableConv1d(nn.Module):
    """
    Implementa una capa de convolución deformable 1D.
    
    Esta capa permite una deformación adaptativa del campo receptivo,
    lo que puede mejorar la capacidad del modelo para capturar características
    en diferentes escalas y posiciones.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        """
        Inicializa la capa de convolución deformable 1D.

        Args:
            in_channels (int): Número de canales de entrada.
            out_channels (int): Número de canales de salida.
            kernel_size (int): Tamaño del kernel de convolución.
            stride (int): Paso de la convolución.
            padding (int): Padding aplicado a la entrada.
            dilation (int): Factor de dilatación de la convolución.
            bias (bool): Si se debe incluir un término de sesgo.
        """
        super(DeformableConv1d, self).__init__()
        self.kernel_size = kernel_size  # Almacena el tamaño del kernel
        self.padding = padding  # Almacena el padding (debe ser un entero)
        self.stride = stride  # Almacena el paso de la convolución
        self.dilation = dilation  # Almacena el factor de dilatación
        self.in_channels = in_channels  # Almacena el número de canales de entrada
        self.out_channels = out_channels  # Almacena el número de canales de salida

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
            in_channels * kernel_size,  # Ajustado para operar sobre los canales deformados
            out_channels,
            kernel_size=1,  # Kernel size ajustado a 1 para operar sobre los canales
            stride=1,       # Stride ajustado a 1
            padding=0,      # No se necesita padding adicional
            dilation=1,     # No se necesita dilatación adicional
            bias=bias
        )
        
    def forward(self, x):
        """
        Realiza la pasada hacia adelante de la capa de convolución deformable 1D.

        Args:
            x (Tensor): Tensor de entrada de forma [N, C, L].

        Returns:
            Tensor: Salida procesada de forma [N, out_channels, L_out].
        """
        # Implementamos autocast para habilitar la precisión mixta
        with autocast(enabled=True):
            # Calcula los offsets
            offsets = self.offset_conv(x)  # [N, 2 * kernel_size, L_out]
            N, _, L_out = offsets.size()
            offsets = offsets.view(N, self.kernel_size, 2, L_out)  # [N, kernel_size, 2, L_out]
            offsets = offsets.permute(0, 3, 1, 2)  # [N, L_out, kernel_size, 2]

            # Prepara la entrada para la interpolación
            x_padded = F.pad(x, (self.padding, self.padding))  # [N, C, L + 2 * padding]

            # Obtiene el dispositivo y tipo de datos del tensor de entrada
            device = x.device
            dtype = x.dtype

            # Crea una malla de posiciones base
            base_grid = torch.arange(0, x_padded.size(2), device=device, dtype=dtype).unsqueeze(0).unsqueeze(2)  # [1, L_padded, 1]
            base_grid = base_grid.repeat(N, 1, self.kernel_size)  # [N, L_padded, kernel_size]

            # Aplica los offsets a la malla base
            grid = base_grid[:, self.padding:x_padded.size(2)-self.padding, :] + offsets[..., 0]  # [N, L_out, kernel_size]

            # Limita los valores del grid para evitar índices fuera de rango
            grid = grid.clamp(0, x_padded.size(2) - 1)

            # Obtiene índices de izquierda y derecha para la interpolación
            left = grid.floor().long()  # [N, L_out, kernel_size]
            right = (left + 1).clamp(max=x_padded.size(2) - 1)  # [N, L_out, kernel_size]
            alpha = grid - left.float()  # [N, L_out, kernel_size]

            # Reshape para gather
            left = left.view(N, -1).unsqueeze(1).expand(-1, self.in_channels, -1)  # [N, C, L_out * kernel_size]
            right = right.view(N, -1).unsqueeze(1).expand(-1, self.in_channels, -1)  # [N, C, L_out * kernel_size]

            # Recoge los valores a la izquierda y a la derecha
            x_left = torch.gather(x_padded, 2, left)  # [N, C, L_out * kernel_size]
            x_right = torch.gather(x_padded, 2, right)  # [N, C, L_out * kernel_size]

            # Reorganiza para obtener [N, C, L_out, kernel_size]
            x_left = x_left.view(N, self.in_channels, L_out, self.kernel_size)
            x_right = x_right.view(N, self.in_channels, L_out, self.kernel_size)

            # Realiza interpolación lineal
            alpha = alpha.view(N, 1, L_out, self.kernel_size)  # [N, 1, L_out, kernel_size]
            x_deform = (1 - alpha) * x_left + alpha * x_right  # [N, C, L_out, kernel_size]

            # Reorganiza para la convolución principal
            x_deform = x_deform.permute(0, 3, 2, 1).contiguous().view(N, self.in_channels * self.kernel_size, L_out)  # [N, C * kernel_size, L_out]

            # Aplica la convolución principal ajustada
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
        """
        Inicializa la capa de convolución con puerta optimizada.

        Args:
            channels (int): Número de canales de entrada y salida.
            kernel_size (int): Tamaño del kernel de convolución.
            dilation (int): Factor de dilatación de la convolución.
        """
        super(OptimizedGatedConvolution, self).__init__()
        self.channels = channels  # Almacena el número de canales
        self.kernel_size = kernel_size  # Almacena el tamaño del kernel
        self.dilation = dilation  # Almacena el factor de dilatación
        
        # Calcula el padding para 'same' padding
        padding = (kernel_size - 1) * dilation // 2
        
        # Reemplaza la convolución estándar por DeformableConv1d
        self.deform_conv = DeformableConv1d(
            in_channels=channels,
            out_channels=channels * 2,  # Duplica los canales para main y gate
            kernel_size=kernel_size,
            padding=padding,  # Debe ser un entero
            dilation=dilation
        )
        
        # Inicialización de los parámetros
        nn.init.kaiming_normal_(self.deform_conv.conv.weight)  # Inicializa los pesos de la convolución principal
        nn.init.zeros_(self.deform_conv.conv.bias)  # Inicializa los sesgos de la convolución principal a cero
        nn.init.kaiming_normal_(self.deform_conv.offset_conv.weight)  # Inicializa los pesos de la convolución de offset
        nn.init.zeros_(self.deform_conv.offset_conv.bias)  # Inicializa los sesgos de la convolución de offset a cero
        
    def forward(self, x):
        """
        Realiza la pasada hacia adelante de la capa de convolución con puerta optimizada.

        Args:
            x (Tensor): Tensor de entrada de forma [batch, seq_length, channels].

        Returns:
            Tensor: Salida procesada de forma [batch, seq_length, channels].
        """
        # Definimos una función interna para usar con checkpoint
        def conv_function(x):
            # Implementamos autocast para habilitar la precisión mixta
            with autocast(enabled=True):
                # Transpone para que los canales sean la segunda dimensión
                x = x.transpose(1, 2)  # [batch, channels, seq_length]
                
                # Aplica la convolución deformable y divide el resultado
                conv_out = self.deform_conv(x)  # [batch, channels*2, L_out]
                main, gate = conv_out.chunk(2, dim=1)  # Divide en dos partes iguales
                
                # Aplica las funciones de activación
                main = F.gelu(main)  # GELU para la salida principal
                gate = torch.sigmoid(gate)  # Sigmoide para la puerta
                
                # Aplica la puerta y normaliza
                gated_out = main * gate  # Multiplicación elemento a elemento
                
                # Normalización de capa
                mean = gated_out.mean(dim=1, keepdim=True)  # Calcula la media por canal
                var = gated_out.var(dim=1, keepdim=True, unbiased=False)  # Calcula la varianza por canal
                gated_out = (gated_out - mean) / (var + 1e-5).sqrt()  # Normaliza
                
                # Vuelve a transponer para mantener la forma original
                return gated_out.transpose(1, 2)  # [batch, L_out, channels]

        # Usamos checkpoint para ahorrar memoria durante el entrenamiento
        return checkpoint(conv_function, x)
def test_deformable_conv1d():
    """
    Función de prueba para la clase DeformableConv1d.
    
    Esta función crea una instancia de DeformableConv1d, le pasa datos de prueba
    y verifica que la salida tenga la forma esperada.
    """
    batch_size = 2  # Tamaño del batch para la prueba
    in_channels = 4  # Número de canales de entrada
    out_channels = 8  # Número de canales de salida
    seq_length = 16  # Longitud de la secuencia de entrada
    kernel_size = 3  # Tamaño del kernel de convolución
    dilation = 1  # Factor de dilatación
    stride = 1  # Paso de la convolución
    padding = (kernel_size - 1) * dilation // 2  # Padding para mantener la longitud de la secuencia

    # Crea un tensor de entrada aleatorio
    x = torch.randn(batch_size, in_channels, seq_length)  # [N, C, L]
    
    # Crea una instancia de DeformableConv1d
    deform_conv = DeformableConv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
    
    # Realiza la pasada hacia adelante
    out = deform_conv(x)
    
    # Imprime las formas de entrada y salida
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

# Ejecuta la función de prueba
test_deformable_conv1d()

def test_optimized_gated_convolution():
    """
    Función de prueba para la clase OptimizedGatedConvolution.
    
    Esta función crea una instancia de OptimizedGatedConvolution, le pasa datos de prueba
    y verifica que la salida tenga la forma esperada.
    """
    batch_size = 2  # Tamaño del batch para la prueba
    channels = 4  # Número de canales
    seq_length = 16  # Longitud de la secuencia
    kernel_size = 3  # Tamaño del kernel
    dilation = 1  # Factor de dilatación

    # Crea un tensor de entrada aleatorio
    x = torch.randn(batch_size, seq_length, channels)  # [batch, seq_length, channels]
    
    # Crea una instancia de OptimizedGatedConvolution
    gated_conv = OptimizedGatedConvolution(channels, kernel_size, dilation)
    
    # Realiza la pasada hacia adelante
    out = gated_conv(x)
    
    # Imprime las formas de entrada y salida
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

# Ejecuta la función de prueba
test_optimized_gated_convolution()


class EnhancedLSTM(nn.Module):
    """
    Implementa una versión mejorada de LSTM (Long Short-Term Memory).
    
    Esta clase extiende la funcionalidad de un LSTM estándar añadiendo
    una capa de salida adicional con activación GELU y una conexión residual.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        """
        Inicializa el módulo EnhancedLSTM.

        Args:
            input_size (int): Tamaño de la entrada en cada paso de tiempo.
            hidden_size (int): Número de features en el estado oculto.
            num_layers (int): Número de capas LSTM apiladas.
            dropout (float): Probabilidad de dropout entre las capas LSTM (excepto la última).
        """
        super(EnhancedLSTM, self).__init__()
        self.input_size = input_size  # Almacena el tamaño de entrada
        self.hidden_size = hidden_size  # Almacena el tamaño del estado oculto
        self.num_layers = num_layers  # Almacena el número de capas
        
        # LSTM estándar de PyTorch
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Capa de salida adicional con GELU
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # Capa lineal
            nn.GELU(),  # Activación GELU
            nn.Linear(hidden_size, input_size)  # Capa lineal final
        )

    def forward(self, x, hidden=None):
        """
        Realiza la pasada hacia adelante del EnhancedLSTM.

        Args:
            x (Tensor): Entrada de forma [batch, seq_len, input_size].
            hidden (Tuple[Tensor, Tensor], optional): Estado oculto inicial (h_0, c_0).

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: 
                - Salida procesada de forma [batch, seq_len, input_size].
                - Tuple conteniendo el estado oculto final y el estado de celda final.
        """
        # Pasa a través del LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Aplica la capa de salida con conexión residual
        output = self.output_layer(lstm_out) + x
        
        return output, hidden

    def init_hidden(self, batch_size):
        """
        Inicializa el estado oculto del LSTM.

        Args:
            batch_size (int): Tamaño del batch.

        Returns:
            Tuple[Tensor, Tensor]: Tuple conteniendo el estado oculto inicial y el estado de celda inicial.
        """
        weight = next(self.parameters()).data  # Obtiene un tensor de parámetros para usar su tipo y dispositivo
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(weight.device),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(weight.device))
        return hidden
class ImprovedTransformerBlock(nn.Module):
    """
    Implementa un bloque de Transformer mejorado con atención local, convolución con puerta,
    y una capa de Mixture of Experts (MoE).
    """

    def __init__(self, embed_dim, num_heads, ff_hidden_dim, num_experts, expert_dim, 
                 window_size=256, bidirectional=True, dropout=0.12, entropy_weight=0.1, 
                 top_k=2, dynamic_k=False):
        """
        Inicializa el bloque de Transformer mejorado.

        Args:
            embed_dim (int): Dimensión del embedding.
            num_heads (int): Número de cabezas de atención.
            ff_hidden_dim (int): Dimensión oculta de la capa feed-forward.
            num_experts (int): Número de expertos en la capa MoE.
            expert_dim (int): Dimensión de cada experto.
            window_size (int): Tamaño de la ventana para la atención local.
            bidirectional (bool): Si es True, usa atención bidireccional.
            dropout (float): Tasa de dropout.
            entropy_weight (float): Peso para la regularización de entropía en MoE.
            top_k (int): Número de expertos top-k a seleccionar en MoE.
            dynamic_k (bool): Si es True, ajusta dinámicamente k en MoE.
        """
        super(ImprovedTransformerBlock, self).__init__()
        
        # Capa de atención local mejorada
        self.attention = EnhancedLocalAttention(embed_dim, num_heads, window_size, bidirectional)
        self.norm1 = nn.LayerNorm(embed_dim)  # Normalización de capa después de la atención
        self.dropout1 = nn.Dropout(dropout)  # Dropout después de la atención
        
        # Capa de convolución con puerta dilatada
        self.dilated_conv = OptimizedGatedConvolution(embed_dim, kernel_size=3, dilation=2)
        self.norm2 = nn.LayerNorm(embed_dim)  # Normalización de capa después de la convolución
        self.dropout2 = nn.Dropout(dropout)  # Dropout después de la convolución
        
        # Capa de Mixture of Experts
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
        self.norm3 = nn.LayerNorm(embed_dim)  # Normalización de capa después de MoE
        self.dropout3 = nn.Dropout(dropout)  # Dropout después de MoE
        
        # Capa feed-forward
        self.ff_layer = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),  # Primera capa lineal
            nn.GELU(),  # Activación GELU
            nn.Dropout(dropout),  # Dropout
            nn.Linear(ff_hidden_dim, embed_dim)  # Segunda capa lineal
        )
        
    def forward(self, x):
        """
        Realiza la pasada hacia adelante del bloque de Transformer mejorado.

        Args:
            x (Tensor): Tensor de entrada de forma [batch_size, seq_length, embed_dim].

        Returns:
            Tuple[Tensor, Tensor]: 
                - Salida procesada de forma [batch_size, seq_length, embed_dim].
                - Pérdida de entropía de la capa MoE.
        """
        # Utilizar autocast para mantener la precisión mixta
        with torch.cuda.amp.autocast(enabled=True):
            # Capa de atención
            x = x + self.dropout1(self.attention(self.norm1(x)))
            
            # Capa de convolución con puerta
            x = x + self.dropout2(self.dilated_conv(self.norm2(x)))
            
            # Capa MoE
            moe_output, entropy_loss = self.moe(self.norm3(x))
            x = x + self.dropout3(moe_output)
            
            # Capa feed-forward
            x = x + self.ff_layer(x)
        
        return x, entropy_loss
class BidirectionalEncoder(nn.Module):
    """
    Implementa un codificador bidireccional utilizando capas de Transformer mejoradas.
    """

    def __init__(self, vocab_size, embed_dim, max_length=2048, compression_ratio=0.5, 
                 num_layers=4, num_heads=8, ff_hidden_dim=1024, window_size=256, 
                 num_experts=2, expert_dim=256, entropy_weight=0.1, top_k=2, dynamic_k=False):
        """
        Inicializa el codificador bidireccional.

        Args:
            vocab_size (int): Tamaño del vocabulario.
            embed_dim (int): Dimensión del embedding.
            max_length (int): Longitud máxima de la secuencia.
            compression_ratio (float): Ratio de compresión para el embedding líquido.
            num_layers (int): Número de capas del Transformer.
            num_heads (int): Número de cabezas de atención en cada capa.
            ff_hidden_dim (int): Dimensión oculta de la capa feed-forward.
            window_size (int): Tamaño de la ventana para la atención local.
            num_experts (int): Número de expertos en la capa MoE.
            expert_dim (int): Dimensión de cada experto.
            entropy_weight (float): Peso para la regularización de entropía en MoE.
            top_k (int): Número de expertos top-k a seleccionar en MoE.
            dynamic_k (bool): Si es True, ajusta dinámicamente k en MoE.
        """
        super(BidirectionalEncoder, self).__init__()
        
        # Capa de embedding líquido
        self.embedding = LiquidEmbedding(vocab_size, embed_dim, max_length, base_compression_ratio=compression_ratio)
        
        # Lista de capas de Transformer mejoradas
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
        
        # Normalización de capa final
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Realiza la pasada hacia adelante del codificador bidireccional.

        Args:
            x (Tensor): Tensor de entrada de forma [batch_size, seq_length].

        Returns:
            Tuple[Tensor, Tensor, Tensor]: 
                - Salida procesada de forma [batch_size, seq_length, embed_dim].
                - Pérdida de reconstrucción del embedding líquido.
                - Suma de las pérdidas de entropía de todas las capas MoE.
        """
        # Utilizar autocast para mantener la precisión mixta
        with autocast(enabled=True):
            # Aplicar embedding líquido
            x, recon_loss = self.embedding(x)
            
            # Inicializar la pérdida de entropía total
            total_entropy_loss = 0
            
            # Pasar por cada capa de Transformer
            for layer in self.layers:
                x, entropy_loss = layer(x)
                total_entropy_loss += entropy_loss
            
            # Aplicar normalización de capa final
            x = self.layer_norm(x)
        
        return x, recon_loss, total_entropy_loss
class LiquidFoundationModelOptimized(nn.Module):
    """
    Implementa el modelo LiquidFoundation optimizado, que combina un codificador
    bidireccional, un decodificador, y una memoria externa mejorada.
    """

    def __init__(self, vocab_size, embed_dim=256, num_layers=4, num_heads=8, ff_hidden_dim=1024,
                 num_experts=4, expert_dim=256, max_length=2048, window_size=256, compression_ratio=0.5, 
                 entropy_weight=0.1, top_k=2, dynamic_k=False, lstm_hidden_size=256, lstm_num_layers=2):
        """
        Inicializa el modelo LiquidFoundation optimizado.

        Args:
            vocab_size (int): Tamaño del vocabulario.
            embed_dim (int): Dimensión del embedding.
            num_layers (int): Número de capas del Transformer.
            num_heads (int): Número de cabezas de atención en cada capa.
            ff_hidden_dim (int): Dimensión oculta de la capa feed-forward.
            num_experts (int): Número de expertos en la capa MoE.
            expert_dim (int): Dimensión de cada experto.
            max_length (int): Longitud máxima de la secuencia.
            window_size (int): Tamaño de la ventana para la atención local.
            compression_ratio (float): Ratio de compresión para el embedding líquido.
            entropy_weight (float): Peso para la regularización de entropía en MoE.
            top_k (int): Número de expertos top-k a seleccionar en MoE.
            dynamic_k (bool): Si es True, ajusta dinámicamente k en MoE.
            lstm_hidden_size (int): Tamaño del estado oculto del LSTM mejorado.
            lstm_num_layers (int): Número de capas del LSTM mejorado.
        """
        super(LiquidFoundationModelOptimized, self).__init__()
        
        # Codificador bidireccional
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
        
        # Embedding del decodificador
        self.decoder_embedding = LiquidEmbedding(
            vocab_size, 
            embed_dim, 
            max_length, 
            base_compression_ratio=0.5, 
            min_compression_ratio=0.1
        )
        
        # Capas del decodificador
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
        
        # Normalización de capa final
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Capa de salida
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
        # Parámetros adicionales
        self.max_length = max_length
        self.compression_ratio = compression_ratio
        
        # LSTM mejorado para memoria externa
        self.external_memory = EnhancedLSTM(embed_dim, lstm_hidden_size, num_layers=lstm_num_layers)

    def forward(self, encoder_input_ids, decoder_input_ids):
        """
        Realiza la pasada hacia adelante del modelo LiquidFoundation optimizado.

        Args:
            encoder_input_ids (Tensor): IDs de entrada del codificador de forma [batch_size, seq_length].
            decoder_input_ids (Tensor): IDs de entrada del decodificador de forma [batch_size, seq_length].

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: 
                - Logits de salida de forma [batch_size, seq_length, vocab_size].
                - Pérdida de reconstrucción del codificador.
                - Pérdida de reconstrucción del decodificador.
                - Pérdida de entropía del codificador.
                - Pérdida de entropía total del decodificador.
        """
        with autocast(enabled=True):
            # Codificación
            encoder_output, recon_loss_enc, entropy_loss_enc = self.encoder(encoder_input_ids)
            
            # Embedding del decodificador
            decoder_embeddings, recon_loss_dec = self.decoder_embedding(decoder_input_ids)
            
            # Inicializar el estado oculto del EnhancedLSTM
            batch_size = decoder_embeddings.size(0)
            hidden = self.external_memory.init_hidden(batch_size)
            
            # Inicializar la pérdida de entropía total del decodificador
            total_entropy_loss_dec = 0
            
            # Pasar por cada capa del decodificador
            for layer in self.decoder_layers:
                decoder_embeddings, entropy_loss = layer(decoder_embeddings)
                total_entropy_loss_dec += entropy_loss
                
                # Actualizar la memoria externa con EnhancedLSTM
                decoder_embeddings, hidden = self.external_memory(decoder_embeddings, hidden)
            
            # Aplicar normalización de capa final
            decoder_embeddings = self.layer_norm(decoder_embeddings)
            
            # Generar logits de salida
            logits = self.output_layer(decoder_embeddings)
        
        # Limitar la salida a max_length
        return logits[:, :self.max_length, :], recon_loss_enc, recon_loss_dec, entropy_loss_enc, total_entropy_loss_dec

def prepare_data(max_samples=None, val_size=0.1):
    """
    Prepara los datos para el entrenamiento y la validación.

    Args:
        max_samples (int, optional): Número máximo de muestras a usar. Si es None, usa todas las muestras disponibles.
        val_size (float): Proporción de datos a usar para validación.

    Returns:
        Tuple[GPT2Tokenizer, datasets.DatasetDict]: Tokenizador y conjunto de datos dividido en entrenamiento y validación.
    """
    # Inicializar el tokenizador GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Agregar tokens especiales al tokenizador
    special_tokens = {'pad_token': '[PAD]', 'eos_token': '<EOS>', 'bos_token': '<BOS>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens)
    print(f"Se agregaron {num_added_toks} tokens especiales al tokenizer.")
    
    # Cargar el dataset "wikitext-2-raw-v1"
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Limitar el número de muestras si se especifica
    if max_samples is not None and max_samples < len(dataset['train']):
        dataset['train'] = dataset['train'].select(range(max_samples))
    
    def preprocess(examples):
        """
        Función de preprocesamiento para tokenizar y preparar los datos.

        Args:
            examples (dict): Diccionario con los ejemplos a preprocesar.

        Returns:
            dict: Diccionario con los datos preprocesados.
        """
        # Combinar los textos con tokens especiales
        combined_texts = [
            f"{tokenizer.bos_token} {text}{tokenizer.eos_token}"
            for text in examples['text'] if text.strip()  # Solo procesa strings no vacías
        ]
        
        # Tokenizar los textos
        tokens = tokenizer(combined_texts, truncation=True, max_length=2048, padding='max_length')
        
        # Preparar los IDs de entrada del decodificador
        decoder_input_ids = [[tokenizer.bos_token_id] + ids[:-1] for ids in tokens['input_ids']]
        tokens['decoder_input_ids'] = decoder_input_ids
        
        # Preparar las etiquetas
        tokens['labels'] = [ids.copy() for ids in tokens['input_ids']]
        
        # Reemplazar los tokens de padding en las etiquetas por -100 para ignorarlos en la pérdida
        tokens['labels'] = [
            [(id_ if id_ != tokenizer.pad_token_id else -100) for id_ in label if id_ is not None]
            for label in tokens['labels']
        ]
        
        return tokens
    
    # Aplicar el preprocesamiento al dataset
    tokenized_dataset = dataset['train'].map(preprocess, batched=True, batch_size=1000, remove_columns=dataset['train'].column_names)
    
    # Configurar el formato de los datos para PyTorch
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'decoder_input_ids', 'labels', 'attention_mask'])
    
    # Dividir el dataset en entrenamiento y validación
    train_val_dataset = tokenized_dataset.train_test_split(test_size=val_size)
    
    return tokenizer, train_val_dataset

def calculate_metrics(model, data_loader, criterion, device, tokenizer):
    """
    Calcula las métricas de evaluación para el modelo.

    Args:
        model (nn.Module): El modelo a evaluar.
        data_loader (DataLoader): Cargador de datos de validación.
        criterion (nn.Module): Función de pérdida.
        device (torch.device): Dispositivo en el que realizar los cálculos.
        tokenizer (Tokenizer): Tokenizador usado para procesar los datos.

    Returns:
        Tuple[float, float]: Pérdida promedio y perplejidad.
    """
    model.eval()  # Establece el modelo en modo de evaluación
    total_loss = 0  # Inicializa la pérdida total
    total_recon_loss = 0  # Inicializa la pérdida de reconstrucción total
    total_entropy_loss = 0  # Inicializa la pérdida de entropía total
    total_tokens = 0  # Inicializa el contador total de tokens

    with torch.no_grad():  # Desactiva el cálculo de gradientes
        for batch in data_loader:  # Itera sobre los lotes de datos
            # Mueve los datos al dispositivo apropiado
            encoder_input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with autocast(enabled=True):  # Habilita la precisión mixta automática
                # Realiza la pasada hacia adelante del modelo
                outputs, recon_loss_enc, recon_loss_dec, entropy_loss_enc, entropy_loss_dec = model(encoder_input_ids, decoder_input_ids)
                logits = outputs.reshape(-1, outputs.size(-1))  # Reshape los logits
                labels_flat = labels.reshape(-1)  # Aplana las etiquetas
                
                # Crea una máscara para ignorar los tokens de padding
                mask = labels_flat != -100
                logits = logits[mask]
                labels_flat = labels_flat[mask]
                
                # Calcula la pérdida total
                loss = criterion(logits, labels_flat) + recon_loss_enc + recon_loss_dec + entropy_loss_enc + entropy_loss_dec
                total_loss += loss.item() * labels_flat.numel()  # Acumula la pérdida
                total_tokens += labels_flat.numel()  # Cuenta los tokens
    
    avg_loss = total_loss / total_tokens  # Calcula la pérdida promedio
    perplexity = math.exp(avg_loss)  # Calcula la perplejidad

    return avg_loss, perplexity
class OptimizedFocalLoss(nn.Module):
    """
    Implementa una versión optimizada de la Focal Loss con suavizado de etiquetas.
    """

    def __init__(self, alpha=1, gamma=2, ignore_index=-100, label_smoothing=0.1):
        """
        Inicializa la función de pérdida Focal Loss optimizada.

        Args:
            alpha (float): Factor de balanceo.
            gamma (float): Factor de enfoque.
            ignore_index (int): Índice a ignorar en el cálculo de la pérdida.
            label_smoothing (float): Factor de suavizado de etiquetas.
        """
        super(OptimizedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """
        Calcula la Focal Loss.

        Args:
            inputs (Tensor): Predicciones del modelo.
            targets (Tensor): Etiquetas verdaderas.

        Returns:
            Tensor: Valor de la pérdida calculada.
        """
        with torch.set_grad_enabled(self.training):  # Habilita/deshabilita el cálculo de gradientes según el modo
            num_classes = inputs.size(-1)  # Obtiene el número de clases
            
            chunk_size = 1024  # Tamaño del chunk para procesamiento por lotes
            total_loss = 0  # Inicializa la pérdida total
            total_count = 0  # Inicializa el contador total
            
            for i in range(0, inputs.size(0), chunk_size):  # Procesa los datos en chunks
                chunk_inputs = inputs[i:i+chunk_size]  # Obtiene un chunk de entradas
                chunk_targets = targets[i:i+chunk_size]  # Obtiene un chunk de objetivos
                
                # Aplica suavizado de etiquetas
                smoothed_targets = torch.zeros_like(chunk_inputs)
                smoothed_targets.scatter_(1, chunk_targets.unsqueeze(1), 1)
                smoothed_targets.mul_(1 - self.label_smoothing).add_(self.label_smoothing / num_classes)
                
                with torch.cuda.amp.autocast(enabled=True):  # Habilita la precisión mixta automática
                    log_probs = F.log_softmax(chunk_inputs, dim=-1)  # Calcula log-probabilidades
                    loss = -smoothed_targets * log_probs  # Calcula la pérdida
                    
                    loss = loss.sum(-1)  # Suma sobre todas las clases
                    pt = torch.exp(-loss)  # Calcula la probabilidad de la clase correcta
                    focal_loss = self.alpha * (1-pt)**self.gamma * loss  # Aplica Focal Loss
                    
                    if self.ignore_index >= 0:  # Si hay un índice a ignorar
                        mask = chunk_targets != self.ignore_index
                        focal_loss = focal_loss[mask]
                    
                    total_loss += focal_loss.sum()  # Acumula la pérdida
                    total_count += focal_loss.numel()  # Cuenta el número de elementos
            
            return total_loss / total_count if total_count > 0 else total_loss  # Devuelve la pérdida promedio
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, num_epochs, accumulation_steps=8, evaluator=None, tokenizer=None, monitor=None):
    """
    Entrena el modelo LiquidFoundation.

    Args:
        model (nn.Module): El modelo a entrenar.
        train_loader (DataLoader): Cargador de datos de entrenamiento.
        val_loader (DataLoader): Cargador de datos de validación.
        criterion (nn.Module): Función de pérdida.
        optimizer (Optimizer): Optimizador.
        scheduler (LRScheduler): Planificador de tasa de aprendizaje.
        scaler (GradScaler): Escalador de gradientes para precisión mixta.
        device (torch.device): Dispositivo en el que realizar el entrenamiento.
        num_epochs (int): Número de épocas de entrenamiento.
        accumulation_steps (int): Pasos de acumulación de gradientes.
        evaluator (TextEvaluator, optional): Evaluador de texto.
        tokenizer (Tokenizer, optional): Tokenizador.
        monitor (ActivationMonitor, optional): Monitor de activaciones.

    Returns:
        None
    """
    writer = SummaryWriter()  # Inicializa el escritor de TensorBoard
    best_val_loss = float('inf')  # Inicializa la mejor pérdida de validación
    patience = 3  # Paciencia para early stopping
    no_improve = 0  # Contador para early stopping

    for epoch in range(num_epochs):  # Itera sobre las épocas
        model.train()  # Establece el modelo en modo de entrenamiento
        total_loss = 0  # Inicializa la pérdida total
        total_recon_loss = 0  # Inicializa la pérdida de reconstrucción total
        total_entropy_loss = 0  # Inicializa la pérdida de entropía total
        total_batches = 0  # Inicializa el contador de lotes
        
        # Crea una barra de progreso
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Entrenando Epoch {epoch + 1}")
        
        for batch_idx, batch in loop:  # Itera sobre los lotes
            # Mueve los datos al dispositivo apropiado
            encoder_input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)

            with autocast(enabled=True):  # Habilita la precisión mixta automática
                # Realiza la pasada hacia adelante
                outputs, recon_loss_enc, recon_loss_dec, entropy_loss_enc, entropy_loss_dec = model(encoder_input_ids, decoder_input_ids)
                logits = outputs.reshape(-1, outputs.size(-1))  # Reshape los logits
                labels_flat = labels.reshape(-1)  # Aplana las etiquetas
                
                # Crea una máscara para ignorar los tokens de padding
                mask = labels_flat != -100
                logits = logits[mask]
                labels_flat = labels_flat[mask]
                
                # Calcula la pérdida
                loss = criterion(logits, labels_flat) + recon_loss_enc + recon_loss_dec + entropy_loss_enc + entropy_loss_dec
                loss = loss / accumulation_steps  # Normaliza la pérdida

            # Realiza la retropropagación
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:  # Actualiza los pesos cada accumulation_steps
                scaler.unscale_(optimizer)  # Deshace la escala de los gradientes
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Recorta los gradientes
                scaler.step(optimizer)  # Realiza un paso de optimización
                scaler.update()  # Actualiza el factor de escala
                optimizer.zero_grad()  # Reinicia los gradientes

            # Acumula las pérdidas
            total_loss += loss.item() * accumulation_steps
            total_recon_loss += (recon_loss_enc + recon_loss_dec).item() * accumulation_steps
            total_entropy_loss += (entropy_loss_enc + entropy_loss_dec).item() * accumulation_steps
            total_batches += 1

            if monitor and (batch_idx + 1) % 800 == 0:  # Registra activaciones y gradientes periódicamente
                for name, activation in monitor.activations.items():
                    writer.add_histogram(f'Activations/{name}', activation.cpu().numpy(), epoch)
                for name, gradient in monitor.gradients.items():
                    writer.add_histogram(f'Gradients/{name}', gradient.cpu().numpy(), epoch)

            if (batch_idx + 1) % 800 == 0:  # Evalúa y registra métricas periódicamente
                avg_train_loss = total_loss / total_batches
                avg_recon_loss = total_recon_loss / total_batches
                avg_entropy_loss = total_entropy_loss / total_batches
                val_loss, val_perplexity = calculate_metrics(model, val_loader, criterion, device, tokenizer)
                
                loop.set_postfix(train_loss=avg_train_loss, recon_loss=avg_recon_loss, entropy_loss=avg_entropy_loss, val_loss=val_loss, val_perplexity=val_perplexity)

            if (batch_idx + 1) % 100 == 0:  # Registra el uso de expertos periódicamente
                for idx, layer in enumerate(model.encoder.layers):
                    usage_stats = layer.moe.get_expert_usage_stats()
                    print(f"Epoch {epoch}, Batch {batch_idx+1}, Layer {idx} Expert Usage: {usage_stats}")

        scheduler.step()  # Actualiza la tasa de aprendizaje

        # Calcula y registra las métricas de entrenamiento
        avg_train_loss = total_loss / total_batches
        avg_recon_loss = total_recon_loss / total_batches
        avg_entropy_loss = total_entropy_loss / total_batches
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/recon', avg_recon_loss, epoch)
        writer.add_scalar('Loss/entropy', avg_entropy_loss, epoch)

        # Calcula y registra las métricas de validación
        val_loss, val_perplexity = calculate_metrics(model, val_loader, criterion, device, tokenizer)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Perplexity/val', val_perplexity, epoch)

        if evaluator and tokenizer:  # Realiza evaluación adicional si se proporciona un evaluador
            evaluate_model(model, val_loader, evaluator, tokenizer, device, writer, epoch)
        
        # Imprime las métricas de la época
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Train Recon Loss: {avg_recon_loss:.4f}")
        print(f"Train Entropy Loss: {avg_entropy_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Perplexity: {val_perplexity:.2f}")

        # Guarda el mejor modelo y verifica early stopping
        if val_perplexity < best_val_loss:
            best_val_loss = val_perplexity
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping")
            break

        torch.cuda.empty_cache()  # Libera la memoria de la GPU

    writer.close()  # Cierra el escritor de TensorBoard
def calculate_evaluation_metrics(evaluator, questions, answers, labels):
    """
    Calcula métricas de evaluación utilizando el evaluador de texto.

    Args:
        evaluator (TextEvaluator): Instancia del evaluador de texto.
        questions (list): Lista de preguntas o prompts.
        answers (list): Lista de respuestas generadas.
        labels (list): Lista de etiquetas o respuestas correctas.

    Returns:
        dict: Diccionario con las métricas calculadas.
    """
    metrics = evaluator.calculate_metrics(questions, answers, labels)
    coherence = np.mean([evaluator.evaluate_coherence(q, [], a) for q, a in zip(questions, answers)])
    metrics['coherence'] = coherence
    return metrics

def print_metrics(metrics):
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

def evaluate_model(model, val_loader, evaluator, tokenizer, device, writer, epoch):
    """
    Evalúa el modelo utilizando métricas personalizadas.

    Args:
        model (nn.Module): Modelo a evaluar.
        val_loader (DataLoader): Cargador de datos de validación.
        evaluator (TextEvaluator): Instancia del evaluador de texto.
        tokenizer (Tokenizer): Tokenizador utilizado.
        device (torch.device): Dispositivo en el que realizar la evaluación.
        writer (SummaryWriter): Escritor de TensorBoard.
        epoch (int): Época actual.
    """
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
    """
    Redimensiona los embeddings del modelo para acomodar nuevos tokens.

    Args:
        model (nn.Module): Modelo a modificar.
        tokenizer (Tokenizer): Tokenizador actualizado.
        new_vocab_size (int): Nuevo tamaño del vocabulario.
        embed_dim (int): Dimensión de los embeddings.
    """
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
    """
    Función principal para entrenar y evaluar el modelo LiquidFoundation.

    Args:
        max_samples (int): Número máximo de muestras a utilizar para el entrenamiento.

    Returns:
        Tuple[nn.Module, Tokenizer]: El modelo entrenado y el tokenizador.
    """
    # Prepara los datos
    tokenizer, train_val_dataset = prepare_data(max_samples)
    
    VOCAB_SIZE = len(tokenizer)  # Obtiene el tamaño del vocabulario
    
    print(f"Tamaño del vocabulario: {VOCAB_SIZE}")

    # Crea los cargadores de datos
    train_loader = DataLoader(train_val_dataset['train'], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(train_val_dataset['test'], batch_size=BATCH_SIZE, shuffle=False)

    # Inicializa el modelo
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

    # Redimensiona los embeddings para tokens especiales
    resize_embeddings(model, tokenizer, VOCAB_SIZE, EMBED_DIM)
    print("Se actualizó el tamaño del embedding para tokens especiales sin perder los pesos existentes.")

    print(f"Dimensión de output_layer: {model.output_layer.weight.shape}")

    # Inicializa el criterio, optimizador, planificador y escalador
    criterion = OptimizedFocalLoss(ignore_index=-100, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler()

    evaluator = TextEvaluator()
    monitor = ActivationMonitor(model)

    # Entrena el modelo
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

class DynamicCacheManager:
    """
    DynamicCacheManager: Un sistema de caché avanzado con compresión dinámica.

    Esta clase implementa un sistema de caché que utiliza compresión dinámica basada en la
    complejidad de los datos para optimizar el almacenamiento. Utiliza la Transformada Rápida
    de Fourier (FFT) para comprimir y descomprimir datos, y emplea una política de reemplazo
    Least Recently Used (LRU) para gestionar el tamaño del caché.

    Atributos:
        cache (OrderedDict): Estructura de datos ordenada que almacena los items cacheados.
                             La ordenación se utiliza para implementar la política LRU.
        max_size (int): Número máximo de elementos que puede contener el caché.
        base_compression_ratio (float): Ratio de compresión base utilizado en el algoritmo de compresión.
        min_compression_ratio (float): Ratio de compresión mínimo para evitar una compresión excesiva.

    Métodos:
        __init__: Inicializa el DynamicCacheManager.
        _generate_key: Genera una clave única para cada conjunto de parámetros de generación.
        _calculate_complexity: Calcula la complejidad de un array de datos.
        _compress_data: Comprime los datos utilizando FFT y compresión dinámica.
        _decompress_data: Descomprime los datos previamente comprimidos.
        get: Recupera un item del caché.
        set: Almacena un item en el caché.
    """

    def __init__(self, max_size=1000, base_compression_ratio=0.5, min_compression_ratio=0.1):
        """
        Inicializa el DynamicCacheManager.

        Args:
            max_size (int): Tamaño máximo del caché. Por defecto es 1000.
            base_compression_ratio (float): Ratio de compresión base. Por defecto es 0.5.
            min_compression_ratio (float): Ratio de compresión mínimo. Por defecto es 0.1.

        El constructor inicializa el caché como un OrderedDict vacío y establece los
        parámetros de compresión. El uso de OrderedDict permite mantener un orden de
        inserción/acceso, crucial para implementar la política LRU.
        """
        self.cache = OrderedDict()  # Inicializa un diccionario ordenado vacío para el caché
        self.max_size = max_size  # Establece el tamaño máximo del caché
        self.base_compression_ratio = base_compression_ratio  # Establece el ratio de compresión base
        self.min_compression_ratio = min_compression_ratio  # Establece el ratio de compresión mínimo

    def _generate_key(self, prompt, max_length, beam_width, temperature, top_p, repetition_penalty, max_step_tokens, max_answer_tokens, top_k, num_steps):
        """
        Genera una clave única para un conjunto específico de parámetros de generación.

        Args:
            prompt (str): El prompt de entrada.
            max_length (int): Longitud máxima de la secuencia generada.
            beam_width (int): Ancho del beam search.
            temperature (float): Temperatura para el muestreo.
            top_p (float): Valor para el muestreo nucleus.
            repetition_penalty (float): Penalización por repetición.
            max_step_tokens (int): Número máximo de tokens por paso.
            max_answer_tokens (int): Número máximo de tokens en la respuesta.
            top_k (int): Valor para el muestreo top-k.
            num_steps (int): Número de pasos de generación.

        Returns:
            str: Una clave hash MD5 única generada a partir de los parámetros.

        Este método crea una representación de string de todos los parámetros y luego
        genera un hash MD5 de este string. Esto asegura que cada conjunto único de
        parámetros tenga una clave única en el caché, permitiendo una recuperación
        precisa de resultados previamente generados.
        """
        # Crea un string que concatena todos los parámetros
        key = f"{prompt}_{max_length}_{beam_width}_{temperature}_{top_p}_{repetition_penalty}_{max_step_tokens}_{max_answer_tokens}_{top_k}_{num_steps}"
        # Genera y devuelve un hash MD5 del string
        return hashlib.md5(key.encode()).hexdigest()

    def _calculate_complexity(self, data_array):
        """
        Calcula la complejidad de un array de datos basándose en su espectro de frecuencia.

        Args:
            data_array (numpy.ndarray): Array de datos para el cual se calculará la complejidad.

        Returns:
            float: Un valor de complejidad entre 0 y 1.

        Este método utiliza la Transformada Rápida de Fourier (FFT) para analizar el
        contenido de frecuencia de los datos. La complejidad se determina por la
        proporción de componentes de frecuencia significativas (por encima del 10%
        de la magnitud máxima) en relación con el total de componentes.

        Un valor de complejidad cercano a 1 indica datos con un espectro de frecuencia
        rico (más complejos), mientras que un valor cercano a 0 indica datos con un
        espectro de frecuencia más simple.
        """
        # Calcula la FFT del array de datos
        fft_result = np.fft.fft(data_array)
        # Calcula la magnitud del espectro de frecuencia
        magnitude = np.abs(fft_result)
        # Determina qué componentes de frecuencia son significativas (> 10% del máximo)
        complexity = (magnitude > 0.1 * magnitude.max()).mean()
        return complexity

    def _compress_data(self, data):
        """
        Comprime los datos utilizando FFT y compresión dinámica basada en la complejidad.

        Args:
            data (Union[tuple, str, list]): Datos a comprimir.

        Returns:
            Tuple[numpy.ndarray, float]: 
                - Array comprimido de coeficientes FFT.
                - Ratio de compresión dinámico utilizado.

        Raises:
            ValueError: Si el tipo de dato no es soportado para compresión.

        Este método realiza los siguientes pasos:
        1. Convierte los datos de entrada en un array numpy de bytes.
        2. Calcula la complejidad de los datos.
        3. Determina un ratio de compresión dinámico basado en la complejidad.
        4. Aplica FFT a los datos.
        5. Comprime los datos reteniendo solo una porción de los coeficientes FFT.

        La compresión es más agresiva para datos menos complejos y más conservadora
        para datos más complejos, optimizando así el equilibrio entre el tamaño de
        almacenamiento y la fidelidad de los datos.
        """
        # Convierte los datos a un array numpy de bytes
        if isinstance(data, tuple):
            data_list = list(data)
            data_bytes = [str(item).encode() for item in data_list]
            data_array = np.frombuffer(b''.join(data_bytes), dtype=np.uint8)
        elif isinstance(data, str):
            data_array = np.frombuffer(data.encode(), dtype=np.uint8)
        elif isinstance(data, list):
            data_array = np.array(data)
        else:
            raise ValueError(f"Tipo de dato no soportado para compresión: {type(data)}")

        # Calcula la complejidad de los datos
        complexity = self._calculate_complexity(data_array)
        # Determina el ratio de compresión dinámico
        dynamic_compression_ratio = self.base_compression_ratio * (1 - complexity)
        dynamic_compression_ratio = max(self.min_compression_ratio, dynamic_compression_ratio)

        # Aplica FFT y comprime reteniendo solo una porción de los coeficientes
        fft_result = np.fft.fft(data_array)
        compressed_size = int(len(fft_result) * dynamic_compression_ratio)
        compressed_fft = fft_result[:compressed_size]
        
        return compressed_fft, dynamic_compression_ratio

    def _decompress_data(self, compressed_data, compression_ratio, original_type):
        """
        Descomprime los datos previamente comprimidos.

        Args:
            compressed_data (numpy.ndarray): Datos comprimidos (coeficientes FFT).
            compression_ratio (float): Ratio de compresión utilizado.
            original_type (type): Tipo de dato original antes de la compresión.

        Returns:
            Union[str, list, tuple]: Datos descomprimidos en su tipo original.

        Raises:
            ValueError: Si el tipo de dato original no es soportado para descompresión.

        Este método realiza la operación inversa de _compress_data:
        1. Reconstruye el array FFT completo rellenando con ceros.
        2. Aplica la FFT inversa para obtener los datos originales.
        3. Convierte los datos de vuelta a su tipo original.

        La descompresión intenta preservar la fidelidad de los datos originales,
        pero puede haber alguna pérdida debido a la naturaleza de la compresión FFT.
        """
        # Reconstruye el array FFT completo
        full_size = int(len(compressed_data) / compression_ratio)
        reconstructed_fft = np.zeros(full_size, dtype=complex)
        reconstructed_fft[:len(compressed_data)] = compressed_data

        # Aplica la FFT inversa
        decompressed_array = np.fft.ifft(reconstructed_fft).real.astype(np.uint8)
        decompressed_bytes = decompressed_array.tobytes()

        # Convierte los datos de vuelta a su tipo original
        if original_type == str:
            return decompressed_bytes.decode()
        elif original_type == list:
            return decompressed_array.tolist()
        elif original_type == tuple:
            # Divide los bytes descomprimidos en sus componentes originales
            components = decompressed_bytes.split(b'\x00')  # Asumimos que '\x00' separa los componentes
            return tuple(eval(comp.decode()) for comp in components if comp)
        else:
            raise ValueError(f"Tipo de dato no soportado para descompresión: {original_type}")

    def get(self, key):
        """
        Recupera un item del caché.

        Args:
            key (str): La clave del item a recuperar.

        Returns:
            Any: El item descomprimido si está en el caché, None si no está.

        Este método busca un item en el caché usando la clave proporcionada.
        Si el item está en el caché:
        1. Mueve el item al final del OrderedDict (marcándolo como recientemente usado).
        2. Descomprime los datos.
        3. Devuelve los datos descomprimidos.

        Si el item no está en el caché, devuelve None.

        Este método implementa parte de la política LRU al mover los items accedidos
        al final del OrderedDict.
        """
        if key in self.cache:
            # Mover la clave al final para marcarla como recientemente usada
            self.cache.move_to_end(key)
            compressed_data, compression_ratio, original_type = self.cache[key]
            return self._decompress_data(compressed_data, compression_ratio, original_type)
        return None

    def set(self, key, value):
        """
        Almacena un item en el caché.

        Args:
            key (str): La clave bajo la cual almacenar el item.
            value (Any): El valor a almacenar.

        Este método realiza las siguientes operaciones:
        1. Si la clave ya existe, mueve el item al final del OrderedDict.
        2. Si el caché está lleno, elimina el item menos recientemente usado.
        3. Comprime el valor.
        4. Almacena el valor comprimido en el caché.

        Este método implementa la política LRU completa:
        - Mueve los items existentes al final cuando son actualizados.
        - Elimina el item del principio (menos recientemente usado) cuando el caché está lleno.

        La compresión se realiza para optimizar el uso de memoria, permitiendo
        almacenar más items en el caché.
        """
        if key in self.cache:
            # Si la clave ya existe, moverla al final (recientemente usada)
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            # Si el caché está lleno, eliminar el item menos recientemente usado
            removed_key, _ = self.cache.popitem(last=False)
            print(f"Eliminada la entrada menos recientemente usada: {removed_key}")
        
        # Comprimir y almacenar el valor
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
    model, tokenizer = main(max_samples=400)
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
            num_steps=1, 
            max_attempts=1,
            beam_width=5,
            top_p=0.9,
            repetition_penalty=0.8,
            num_iterations=1,
            evaluator=evaluator
        )
        print(f"Pregunta:\n{question}\nRespuesta:\n{response}")
        print("Pasos de razonamiento:")
        for step in cot_steps:
            print(f"{step['step']}: {step['text']}")
        print(f"Puntuación de coherencia: {coherence_score:.4f}")
        print(f"Total de tokens generados: {sum(len(step['text'].split()) for step in cot_steps) + len(response.split())}\n{'-'*50}\n")
