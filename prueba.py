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
import sympy
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# Configuración global
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Constantes globales
EMBED_DIM = 256
# Dimensión de embedding. Valores más altos pueden capturar más información pero aumentan la complejidad.
# Rango recomendado: 128-512. Valores mayores pueden llevar a sobreajuste en datasets pequeños.

NUM_LAYERS = 8
# Número de capas en el codificador/decodificador. Más capas pueden capturar relaciones más complejas.
# Rango recomendado: 4-12. Aumentar con precaución, ya que puede llevar a problemas de gradientes desvanecientes.

NUM_HEADS = 8
# Número de cabezas de atención. Permite al modelo atender a diferentes aspectos simultáneamente.
# Rango recomendado: 4-16. Debe ser un divisor de EMBED_DIM.

FF_HIDDEN_DIM = 1024
# Dimensión oculta de la capa feed-forward. Afecta la capacidad de procesamiento no lineal.
# Rango recomendado: 2-4 veces EMBED_DIM.

NUM_EXPERTS = 4
# Número de expertos en MoE. Más expertos pueden manejar tareas más diversas, pero aumentan la complejidad.
# Rango recomendado: 2-8. Aumentar con cuidado, ya que puede llevar a problemas de entrenamiento.

EXPERT_DIM = 256
# Dimensión de salida de cada experto. Similar a EMBED_DIM en sus implicaciones.
# Rango recomendado: Igual o cercano a EMBED_DIM.

MAX_LENGTH = 2048
# Longitud máxima de secuencia. Afecta directamente el uso de memoria y tiempo de procesamiento.
# Ajustar según las necesidades específicas del dataset y recursos computacionales disponibles.

WINDOW_SIZE = 256
# Tamaño de la ventana para atención local. Balancea eficiencia y capacidad de capturar dependencias de largo alcance.
# Rango recomendado: 128-512. Valores mayores capturan más contexto pero aumentan la complejidad.

COMPRESSION_RATIO = 0.5
# Ratio de compresión para embedding líquido. Menor valor = más compresión.
# Rango recomendado: 0.3-0.7. Ajustar con cuidado, ya que afecta significativamente la representación de la entrada.

BATCH_SIZE = 4
# Tamaño del batch. Afecta la estabilidad del entrenamiento y el uso de memoria.
# Ajustar según la memoria GPU disponible. Valores típicos: 4-32 para GPUs de consumo, más para GPUs de datacenter.

NUM_EPOCHS = 3
# Número de épocas de entrenamiento. Más épocas pueden mejorar el rendimiento pero aumentan el riesgo de sobreajuste.
# Ajustar según el tamaño del dataset y monitorear la pérdida de validación para evitar sobreajuste.

ACCUMULATION_STEPS = 4
# Pasos de acumulación de gradientes. Permite simular batches más grandes en GPUs con memoria limitada.
# Aumentar si se necesita un batch efectivo más grande pero la memoria es limitada.

TOP_K = 2
# Número inicial de expertos a preseleccionar en MoE. Afecta el balance entre especialización y generalización.
# Rango recomendado: 1-3. Valores más altos pueden diluir la especialización de los expertos.

DYNAMIC_K = True
# Activar el ajuste dinámico de K en MoE. Permite adaptabilidad durante el entrenamiento.
# Recomendado mantener en True para mayor flexibilidad, pero puede desactivarse para comportamiento más consistente.
# Clase MathEvaluator
class MathEvaluator:
    def __init__(self):
        self.math_lm = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").to(device)
        self.math_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    def evaluate_coherence(self, question, cot_steps, response):
        full_text = question + " ".join([step["text"] for step in cot_steps]) + response
        
        math_elements = re.findall(r'\d+|[\+\-\*/\^]', full_text)
        math_score = len(math_elements) / len(full_text.split()) if full_text.split() else 0
        
        step_scores = [len(step["text"].split()) for step in cot_steps]
        step_score = sum(step_scores) / len(step_scores) if step_scores else 0
        
        response_score = 1 if response.strip().isdigit() else 0
        
        coherence_score = (math_score + step_score + response_score) / 3
        
        return coherence_score

    def evaluate_math_precision(self, expected, generated):
        try:
            expected_expr = sympy.sympify(expected)
            generated_expr = sympy.sympify(generated)
            return sympy.simplify(expected_expr - generated_expr) == 0
        except:
            return False

    def evaluate_step_relevance(self, question, steps):
        if not steps:
            return 0.0
        
        question_embedding = torch.tensor(self.get_embedding(question)).to(device)  # Shape: [embed_dim]
        step_embeddings = torch.tensor([self.get_embedding(step["text"]) for step in steps]).to(device)  # Shape: [num_steps, embed_dim]
        
        # Normalizar embeddings
        question_norm = question_embedding / question_embedding.norm(dim=-1, keepdim=True)
        step_norms = step_embeddings / step_embeddings.norm(dim=-1, keepdim=True)
        
        # Calcular similitud de coseno en batch
        similarities = torch.mm(step_norms, question_norm.unsqueeze(-1)).squeeze(-1)  # Shape: [num_steps]
        
        return similarities.mean().item()


    def evaluate_reasoning_complexity(self, steps):
        total_operations = sum(len(re.findall(r'[\+\-\*/\^]', step["text"])) for step in steps)
        return total_operations / len(steps) if steps else 0

    def evaluate_numerical_consistency(self, steps, answer):
        # Extraer todos los números de una sola pasada
        numbers = re.findall(r'\d+', ' '.join([step["text"] for step in steps]) + ' ' + answer)
        unique_numbers = set(numbers)
        return len(unique_numbers) / len(numbers) if numbers else 1.0


    def evaluate_concept_coverage(self, question, answer):
        question_concepts = set(re.findall(r'\b\w+\b', question.lower()))
        answer_concepts = set(re.findall(r'\b\w+\b', answer.lower()))
        return len(question_concepts.intersection(answer_concepts)) / len(question_concepts) if question_concepts else 0

    def evaluate_explainability(self, question, steps, answer):
        full_explanation = " ".join([step["text"] for step in steps]) + " " + answer
        question_words = set(word_tokenize(question.lower()))
        explanation_words = set(word_tokenize(full_explanation.lower()))
        return len(question_words.intersection(explanation_words)) / len(question_words) if question_words else 0

    def evaluate_solution_efficiency(self, question, steps):
        ideal_steps = self.estimate_ideal_steps(question)
        return max(0, 1 - abs(ideal_steps - len(steps)) / ideal_steps)

    def evaluate_reasoning_adaptability(self, questions, answers):
        complexities = [self.calculate_problem_complexity(q) for q in questions]
        answer_qualities = [self.evaluate_answer_quality(q, a) for q, a in zip(questions, answers)]
        return np.mean([q / c for q, c in zip(answer_qualities, complexities) if c != 0])

    def evaluate_f1_score(self, pred_text, label_text, tolerance=1e-6, with_steps=False):
        if with_steps:
            pred_steps = self.extract_step_numbers(pred_text)
            label_steps = self.extract_step_numbers(label_text)
            return np.mean([self.calculate_f1(str(p), str(l), tolerance) for p, l in zip(pred_steps, label_steps)])
        else:
            return self.calculate_f1(pred_text, label_text, tolerance)

    def calculate_f1(self, pred_text, label_text, tolerance):
        pred_nums = torch.tensor(self.extract_numbers(pred_text), dtype=torch.float32, device=device)
        label_nums = torch.tensor(self.extract_numbers(label_text), dtype=torch.float32, device=device)

        if not pred_nums.numel() or not label_nums.numel():
            return 0.0

        # Calcular diferencias absolutas en GPU
        diffs = torch.abs(pred_nums.unsqueeze(1) - label_nums)
        true_positives = (diffs <= tolerance).any(dim=1).sum().item()

        precision = true_positives / pred_nums.numel() if pred_nums.numel() else 0
        recall = true_positives / label_nums.numel() if label_nums.numel() else 0

        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    def evaluate_f1_with_complexity(self, pred_text, label_text):
        complexity = self.calculate_problem_complexity(label_text)
        base_f1 = self.evaluate_f1_score(pred_text, label_text)
        return base_f1 * (1 - 1 / (np.log(complexity + 1) + 1))

    def evaluate_f1_with_explanation(self, pred_text, label_text):
        numeric_f1 = self.evaluate_f1_score(pred_text, label_text)
        
        pred_words = pred_text.lower().split()
        label_words = label_text.lower().split()
        bleu_score = sentence_bleu([label_words], pred_words)
        
        return (numeric_f1 + bleu_score) / 2

    def get_embedding(self, text):
        inputs = self.math_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = self.math_lm(**inputs)
        return outputs.logits  # Retorna tensor en GPU
    def estimate_ideal_steps(self, question):
        return max(2, min(5, self.calculate_problem_complexity(question)))

    def calculate_problem_complexity(self, text):
        operations = re.findall(r'[\+\-\*/\^]', text)
        return len(operations) + 1

    def evaluate_answer_quality(self, question, answer):
        return self.evaluate_concept_coverage(question, answer)

    def extract_numbers(self, text):
        return [float(num) for num in re.findall(r'-?\d+\.?\d*', text)]

    def extract_step_numbers(self, text):
        steps = text.split('Step')
        return [self.extract_numbers(step) for step in steps if step.strip()]

    def comprehensive_evaluation(self, question, steps, answer, label_text=None):
        results = {
            "coherence": self.evaluate_coherence(question, steps, answer),
            "step_relevance": self.evaluate_step_relevance(question, steps),
            "reasoning_complexity": self.evaluate_reasoning_complexity(steps),
            "numerical_consistency": self.evaluate_numerical_consistency(steps, answer),
            "concept_coverage": self.evaluate_concept_coverage(question, answer),
            "explainability": self.evaluate_explainability(question, steps, answer),
            "solution_efficiency": self.evaluate_solution_efficiency(question, steps),
        }
        
        if label_text:
            results.update({
                "f1_score": self.evaluate_f1_score(answer, label_text),
                "f1_with_steps": self.evaluate_f1_score(answer, label_text, with_steps=True),
                "f1_with_complexity": self.evaluate_f1_with_complexity(answer, label_text),
                "f1_with_explanation": self.evaluate_f1_with_explanation(answer, label_text),
            })
        
        return results

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
    """
    Implementa una capa de Mixture of Experts (MoE) que permite el enrutamiento dinámico de la entrada a múltiples expertos.
    """
    def __init__(self, input_dim, hidden_dim, num_experts, expert_dim, dropout=0.1, entropy_weight=0.1, top_k=2, dynamic_k=False, max_usage_ratio=0.3):
        """
        Inicializa la capa MoE.

        Args:
            input_dim (int): Dimensión de entrada.
            hidden_dim (int): Dimensión oculta de los expertos.
            num_experts (int): Número total de expertos.
            expert_dim (int): Dimensión de salida de cada experto.
            dropout (float): Tasa de dropout.
            entropy_weight (float): Peso para la regularización de entropía.
            top_k (int): Número inicial de expertos a seleccionar.
            dynamic_k (bool): Si se debe ajustar dinámicamente el número de expertos.
            max_usage_ratio (float): Ratio máximo de uso permitido para cada experto.
        """
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dynamic_k = dynamic_k
        # Lista de expertos (cada uno es una capa lineal)
        self.experts = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_experts)])
        # Capa de enrutamiento (gate)
        self.gate = nn.Linear(input_dim, num_experts)
        self.dropout = nn.Dropout(dropout)
        self.entropy_weight = entropy_weight
        self.max_usage_ratio = max_usage_ratio
        self.expert_usage_counter = None

    def forward(self, x):
        """
        Realiza la pasada hacia adelante de la capa MoE.

        Args:
            x (Tensor): Tensor de entrada de forma [batch_size, seq_length, input_dim].

        Returns:
            Tuple[Tensor, Tensor]: Salida procesada y pérdida combinada (entropía + penalización por uso excesivo).
        """
        batch_size, seq_length, input_dim = x.size()
        x_flat = x.view(-1, input_dim)
        
        # Calcular probabilidades de enrutamiento
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Regularización de entropía
        entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-10), dim=-1).mean()
        entropy_loss = self.entropy_weight * entropy

        # Ajuste dinámico de K (número de expertos a seleccionar)
        if self.dynamic_k:
            complexity = entropy.detach().item()
            K = max(1, min(self.num_experts, int(self.top_k * (1 + complexity))))
        else:
            K = self.top_k

        # Seleccionar los top-K expertos
        topk_probs, topk_indices = torch.topk(gate_probs, K, dim=-1)

        # Inicializar o reiniciar el contador de uso de expertos
        self.expert_usage_counter = torch.zeros(self.num_experts, device=x.device)

        # Preparar el tensor de salida
        expert_outputs = torch.zeros(batch_size * seq_length, self.experts[0].out_features, device=x.device)

        # Procesar la entrada con los expertos seleccionados
        for k in range(K):
            expert_idx = topk_indices[:, k]
            mask = torch.arange(x_flat.size(0), device=x.device).unsqueeze(1) == expert_idx.unsqueeze(1)
            mask = mask.any(dim=1)
            selected_x = x_flat[mask]

            if selected_x.size(0) > 0:
                expert = self.experts[expert_idx[mask][0]]
                expert_output = self.dropout(expert(selected_x))
                expert_outputs[mask] += expert_output * topk_probs[:, k][mask].unsqueeze(1)

                # Actualizar el contador de uso de expertos
                self.expert_usage_counter[expert_idx[mask][0]] += selected_x.size(0)

        # Calcular la penalización por uso excesivo
        usage_ratios = self.expert_usage_counter / (batch_size * seq_length)
        overuse_penalty = torch.sum(F.relu(usage_ratios - self.max_usage_ratio))

        # Reformatear la salida
        output = expert_outputs.view(batch_size, seq_length, -1)

        return output, entropy_loss + overuse_penalty

    def get_expert_usage_stats(self):
        """
        Obtiene estadísticas de uso de los expertos.

        Returns:
            List[float] or None: Porcentajes de uso de cada experto, o None si no hay datos disponibles.
        """
        if self.expert_usage_counter is None:
            return None
        total_usage = self.expert_usage_counter.sum().item()
        usage_percentages = (self.expert_usage_counter / total_usage * 100).tolist()
        return usage_percentages
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
class BidirectionalEncoder(nn.Module):
    """
    Implementa un codificador bidireccional con múltiples capas de transformador mejorado.
    """
    def __init__(self, vocab_size, embed_dim, max_length=2048, compression_ratio=0.5, num_layers=4, num_heads=8, ff_hidden_dim=1024, window_size=256, num_experts=2, expert_dim=256, entropy_weight=0.1, top_k=2, dynamic_k=False):
        super(BidirectionalEncoder, self).__init__()
        # Capa de embedding líquido
        self.embedding = LiquidEmbedding(vocab_size, embed_dim, max_length, base_compression_ratio=compression_ratio)
        # Lista de capas de transformador mejorado
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
        # Normalización final
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Realiza la pasada hacia adelante del codificador bidireccional.

        Args:
            x (Tensor): Tensor de entrada de forma [batch_size, seq_length].

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Salida procesada, pérdida de reconstrucción y pérdida de entropía total.
        """
        # Aplicar embedding líquido
        x, recon_loss = self.embedding(x)
        total_entropy_loss = 0
        # Pasar por cada capa de transformador
        for layer in self.layers:
            x, entropy_loss = layer(x)
            total_entropy_loss += entropy_loss
        # Normalización final
        x = self.layer_norm(x)
        return x, recon_loss, total_entropy_loss
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
# Funciones auxiliares
def prepare_data(max_samples=None, val_size=0.1):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    special_tokens = {'pad_token': '[PAD]', 'eos_token': '<EOS>', 'bos_token': '<BOS>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens)
    print(f"Se agregaron {num_added_toks} tokens especiales al tokenizer.")
    
    dataset = load_dataset("TIGER-Lab/MathInstruct")
    
    if max_samples is not None and max_samples < len(dataset['train']):
        dataset['train'] = dataset['train'].select(range(max_samples))
    
    def preprocess(examples):
        combined_texts = [
            f"{tokenizer.bos_token} Fuente: {source}\nInstrucción: {instruction}\nRazonamiento: {output} {tokenizer.eos_token}"
            for source, instruction, output in zip(examples['source'], examples['instruction'], examples['output'])
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
    tokenized_dataset = dataset['train'].map(preprocess, batched=True, batch_size=1000)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'decoder_input_ids', 'labels', 'attention_mask'])
    
    train_val_dataset = tokenized_dataset.train_test_split(test_size=val_size)
    
    return tokenizer, train_val_dataset

def calculate_metrics(model, data_loader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_entropy_loss = 0
    total_gradients = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in data_loader:
            encoder_input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, num_epochs, accumulation_steps=4, evaluator=None, tokenizer=None, monitor=None):
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
            attention_mask = batch['attention_mask'].to(device)

            with torch.cuda.amp.autocast(enabled=True):
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

            # Monitoreo de activaciones y gradientes
            if monitor and (batch_idx + 1) % 800 == 0:
                for name, activation in monitor.activations.items():
                    writer.add_histogram(f'Activations/{name}', activation.cpu().numpy(), epoch)
                for name, gradient in monitor.gradients.items():
                    writer.add_histogram(f'Gradients/{name}', gradient.cpu().numpy(), epoch)

            # Validación periódica dentro del epoch
            if (batch_idx + 1) % 800 == 0:
                avg_train_loss = total_loss / total_batches
                avg_recon_loss = total_recon_loss / total_batches
                avg_entropy_loss = total_entropy_loss / total_batches
                val_loss, val_perplexity = calculate_metrics(model, val_loader, criterion, device, tokenizer)
                
                loop.set_postfix(train_loss=avg_train_loss, recon_loss=avg_recon_loss, entropy_loss=avg_entropy_loss, val_loss=val_loss, val_perplexity=val_perplexity)
            if (batch_idx + 1) % 100 == 0:  # Cada 100 lotes
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
                
                # Manejar tokens nulos en las etiquetas
                label_tokens = [token for token in batch['labels'][i] if token != -100]  # Filtrar tokens de relleno
                label_tokens = [token for token in label_tokens if token is not None]  # Filtrar tokens nulos
                label = tokenizer.decode(label_tokens, skip_special_tokens=True)
                
                all_questions.append(question)
                all_answers.append(answer)
                all_labels.append(label)
    
    metrics = calculate_evaluation_metrics(evaluator, all_questions, all_answers, all_labels)
    
    for metric, value in metrics.items():
        writer.add_scalar(f'Metrics/{metric}', value, epoch)
    
    print_metrics(metrics)

def calculate_evaluation_metrics(evaluator, questions, answers, labels):
    coherence_scores = []
    f1_scores = []
    bleu_scores = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for q, a, l in zip(questions, answers, labels):
        coherence = evaluator.evaluate_coherence(q, [], a)
        coherence_scores.append(coherence)
        
        f1 = evaluator.evaluate_f1_score(a, l)
        f1_scores.append(f1)

        chencherry = SmoothingFunction()
        bleu = sentence_bleu([word_tokenize(l.lower())], word_tokenize(a.lower()), smoothing_function=chencherry.method1)
        bleu_scores.append(bleu)
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_result = rouge.score(l, a)
        rouge_scores['rouge1'].append(rouge_result['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(rouge_result['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(rouge_result['rougeL'].fmeasure)
    
    metrics = {
        'coherence': np.mean(coherence_scores),
        'f1': np.mean(f1_scores),
        'bleu': np.mean(bleu_scores),
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL'])
    }
    
    return metrics

def print_metrics(metrics):
    print(f"Coherence: {metrics['coherence']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"BLEU: {metrics['bleu']:.4f}")
    print(f"ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {metrics['rougeL']:.4f}")

def resize_embeddings(model, tokenizer, new_vocab_size, embed_dim):
    old_embedding = model.encoder.embedding.token_embedding
    new_embedding = nn.Embedding(new_vocab_size, embed_dim)
    
    new_embedding.weight.data[:old_embedding.num_embeddings, :] = old_embedding.weight.data
    nn.init.normal_(new_embedding.weight.data[old_embedding.num_embeddings:, :], mean=0.0, std=0.02)
    
    model.encoder.embedding.token_embedding = new_embedding.to(device)
    
    model.decoder_embedding.token_embedding = new_embedding.to(device)
    
    model.output_layer = nn.Linear(embed_dim, new_vocab_size).to(device)
    nn.init.normal_(model.output_layer.weight, mean=0.0, std=0.02)
    model.output_layer.bias.data.zero_()
    
    print(f"Capa de salida actualizada: {model.output_layer}")

# Función principal actualizada para pasar top_k y dynamic_k
def main(max_samples=1000):
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
        entropy_weight=0.1,  # Añadir peso para regularización de entropía
        top_k=TOP_K,
        dynamic_k=DYNAMIC_K
    ).to(device)

    resize_embeddings(model, tokenizer, VOCAB_SIZE, EMBED_DIM)
    print("Se actualizó el tamaño del embedding para tokens especiales sin perder los pesos existentes.")

    print(f"Dimensión de output_layer: {model.output_layer.weight.shape}")

    criterion = OptimizedFocalLoss(ignore_index=-100, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler()

    evaluator = MathEvaluator()

    # Inicializar monitor de activaciones y gradientes
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
def unified_generate(model, tokenizer, prompt, device, reasoning=True, 
                     max_length=512, beam_width=5, temperature=1.0, 
                     top_p=0.9, repetition_penalty=1.2, max_step_tokens=70, 
                     max_answer_tokens=30, top_k=50, num_steps=4, 
                     max_attempts=4, num_iterations=3, evaluator=None):
    """
    Genera texto utilizando el modelo entrenado.

    Parámetros:
    - max_length (int, default=512): 
      Longitud máxima de la secuencia generada. 
      Rango recomendado: 64-1024. Ajustar según las necesidades específicas y recursos disponibles.

    - beam_width (int, default=5): 
      Número de beams en la búsqueda de beam. Valores más altos pueden mejorar la calidad pero aumentan el tiempo de generación.
      Rango recomendado: 1-10. Valores mayores a 5 suelen tener rendimientos decrecientes.

    - temperature (float, default=1.0): 
      Controla la aleatoriedad de la generación. Valores más bajos hacen la salida más determinista.
      Rango recomendado: 0.5-1.5. Valores cercanos a 0 pueden llevar a repeticiones, mientras que valores muy altos pueden producir incoherencias.

    - top_p (float, default=0.9): 
      Umbral de probabilidad acumulativa para muestreo nucleico. Controla la diversidad de la salida.
      Rango recomendado: 0.7-1.0. Valores más bajos aumentan la coherencia pero pueden limitar la creatividad.

    - repetition_penalty (float, default=1.2): 
      Penalización para la repetición de tokens. Valores más altos desalientan las repeticiones.
      Rango recomendado: 1.0-1.5. Ajustar con cuidado, ya que valores muy altos pueden afectar la coherencia.

    - max_step_tokens (int, default=70): 
      Número máximo de tokens por paso de razonamiento.
      Ajustar según la complejidad deseada de cada paso. Rango típico: 50-100.

    - max_answer_tokens (int, default=30): 
      Número máximo de tokens para la respuesta final.
      Ajustar según la longitud deseada de la respuesta. Rango típico: 20-50.

    - top_k (int, default=50): 
      Número de tokens de mayor probabilidad a considerar en cada paso de generación.
      Rango recomendado: 20-100. Valores más bajos aumentan la coherencia pero pueden limitar la diversidad.

    - num_steps (int, default=4): 
      Número de pasos de razonamiento.
      Ajustar según la complejidad del problema. Rango típico: 2-6.

    - max_attempts (int, default=4): 
      Número máximo de intentos de generación.
      Aumentar si se encuentran frecuentes fallos de generación. Rango típico: 1-5.

    - num_iterations (int, default=3): 
      Número de iteraciones de refinamiento.
      Más iteraciones pueden mejorar la calidad pero aumentan el tiempo de generación. Rango típico: 1-5.
    """    
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

    return best_overall_response, best_overall_cot_steps, best_overall_coherence

def generate_cot(model, tokenizer, prompt, device, max_step_tokens=70, max_answer_tokens=30, temperature=0.7, top_k=50, beam_width=5, top_p=0.9, repetition_penalty=1.2, num_steps=4, max_attempts=4):
    model.to(device)
    model.eval()
    best_response = ""
    best_cot_steps = []
    best_coherence_score = float('-inf')

    for attempt in range(max_attempts):
        try:
            with torch.no_grad(), torch.cuda.amp.autocast():
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
        
        with torch.cuda.amp.autocast(enabled=True):
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

        math_elements = re.findall(r'\d+|[\+\-\*/\^]', full_text)
        math_score = len(math_elements) / len(full_text.split()) if full_text.split() else 0

        step_scores = [len(step["text"].split()) for step in cot_steps]
        step_score = sum(step_scores) / len(step_scores) if step_scores else 0

        progression_score = evaluate_progression(cot_steps)

        response_score = evaluate_response(cot_steps, response)

        coherence_score = (0.3 * math_score + 0.2 * step_score + 0.3 * progression_score + 0.2 * response_score)

        return coherence_score

    except (TypeError, KeyError) as e:
        print(f"Error en calculate_coherence: {e}")
        print(f"question: {question}")
        print(f"cot_steps: {cot_steps}")
        print(f"response: {response}")
        return float('-inf')

def evaluate_progression(cot_steps):
    if len(cot_steps) < 2:
        return 0
    
    progression_scores = []
    for i in range(1, len(cot_steps)):
        prev_step = cot_steps[i-1]["text"]
        curr_step = cot_steps[i]["text"]
        
        prev_numbers = set(re.findall(r'\d+', prev_step))
        curr_numbers = set(re.findall(r'\d+', curr_step))
        number_overlap = len(prev_numbers.intersection(curr_numbers))
        
        progression_keywords = ["por lo tanto", "entonces", "siguiente", "ahora", "utilizando"]
        keyword_score = sum(1 for keyword in progression_keywords if keyword in curr_step.lower())
        
        progression_scores.append((number_overlap + keyword_score) / (len(prev_numbers) + len(progression_keywords)))
    
    return sum(progression_scores) / len(progression_scores)

def evaluate_response(cot_steps, response):
    if not cot_steps:
        return 0
    
    last_step = cot_steps[-1]["text"]
    last_step_numbers = set(re.findall(r'\d+', last_step))
    response_numbers = set(re.findall(r'\d+', response))
    
    number_overlap = len(last_step_numbers.intersection(response_numbers))
    
    conclusion_keywords = ["resultado", "respuesta", "solución", "por lo tanto", "en conclusión"]
    keyword_score = sum(1 for keyword in conclusion_keywords if keyword in response.lower())
    
    return (number_overlap + keyword_score) / (len(last_step_numbers) + len(conclusion_keywords))

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
    # Número máximo de muestras a utilizar para el entrenamiento
    # Implicaciones: Controla el tamaño del dataset de entrenamiento
    # Ajuste: Aumentar para mejorar el rendimiento, pero también aumenta el tiempo de entrenamiento y los requisitos de memoria
    # Rango recomendado: 1000-100000, dependiendo de los recursos disponibles
    max_samples = 10000
    model, tokenizer = main(max_samples=max_samples)

    # Selección del dispositivo (GPU si está disponible, CPU en caso contrario)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cálculo del número total de parámetros del modelo
    # Útil para estimar la complejidad del modelo y los requisitos de memoria
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device)
    print(f"Total parameters: {total_params}")

    # Ejemplo de prompt para análisis de transformaciones de tokens
    # Puede modificarse para analizar diferentes tipos de entradas
    prompt = "Resuelve la siguiente ecuación: 2x + 5 = 15"
    analyze_token_transformations(model, tokenizer, prompt)

    # Lista de prompts para generar soluciones matemáticas
    # Puede expandirse o modificarse para probar diferentes tipos de problemas
    cot_prompts = [
        "Instrucción: Resuelve el siguiente problema matemático.\nEntrada: Si una bicicleta cuesta $120 y pago con un billete de $200, ¿cuánto cambio recibiré?\nRazonamiento:",
        "Instrucción: Explica el teorema de Pitágoras.\nEntrada: \nRazonamiento:",
    ]

    # Inicialización del evaluador matemático
    evaluator = MathEvaluator()

    print("Generando soluciones matemáticas con Chain of Thought:\n")
    
    for question in cot_prompts:
        # Generación de respuestas utilizando el modelo
        response, cot_steps, coherence_score = unified_generate(
            model, tokenizer, question, device, 
            reasoning=True,
            max_step_tokens=70,  # Máximo número de tokens por paso de razonamiento
                                 # Ajuste: 50-100, dependiendo de la complejidad deseada de cada paso
            max_answer_tokens=30,  # Máximo número de tokens para la respuesta final
                                   # Ajuste: 20-50, según la longitud deseada de la respuesta
            temperature=0.7,  # Controla la aleatoriedad de la generación
                              # Ajuste: 0.5-1.0, valores más bajos para respuestas más deterministas
            top_k=50,  # Número de tokens más probables a considerar en cada paso
                       # Ajuste: 20-100, valores más bajos para mayor coherencia
            num_steps=3,  # Número de pasos de razonamiento
                          # Ajuste: 2-5, según la complejidad del problema
            max_attempts=4,  # Número máximo de intentos de generación
                             # Ajuste: 1-5, aumentar si hay fallos frecuentes
            beam_width=5,  # Número de beams en la búsqueda de beam
                           # Ajuste: 1-10, valores más altos para potencialmente mejor calidad
            top_p=0.9,  # Umbral de probabilidad acumulativa para muestreo nucleico
                        # Ajuste: 0.7-1.0, valores más bajos para mayor coherencia
            repetition_penalty=0.8,  # Penalización para la repetición de tokens
                                     # Ajuste: 0.8-1.2, valores más altos para desalentar repeticiones
            num_iterations=2,  # Número de iteraciones de refinamiento
                               # Ajuste: 1-5, más iteraciones pueden mejorar la calidad pero aumentan el tiempo
            evaluator=evaluator
        )

        # Impresión de resultados
        print(f"Pregunta:\n{question}\nRespuesta:\n{response}")
        print("Pasos de razonamiento:")
        for step in cot_steps:
            print(f"{step['step']}: {step['text']}")
        print(f"Puntuación de coherencia: {coherence_score:.4f}")
        print(f"Total de tokens generados: {sum(len(step['text'].split()) for step in cot_steps) + len(response.split())}\n{'-'*50}\n")
