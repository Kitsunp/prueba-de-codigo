import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint  # Importar checkpoint
import math
import os
import re

# Asegúrate de que CUDA está disponible y prefiere CUDA para generación
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, expert_dim):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        batch_size, seq_length, input_dim = x.size()
        x_flat = x.view(-1, input_dim)
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
        output = torch.sum(gate_probs.unsqueeze(-1) * expert_outputs, dim=1)
        return output.view(batch_size, seq_length, -1)

class LiquidEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length=2048, compression_ratio=0.5):  # Aumentar compression_ratio a 0.5
        super(LiquidEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.compression_ratio = compression_ratio
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length = x.size()
        device = x.device
        positions = torch.arange(0, seq_length, device=device).unsqueeze(0).expand(batch_size, seq_length)
        x = self.token_embedding(x) + self.position_embedding(positions)  # [batch_size, seq_length, embed_dim]
        x = self.conv1(x.permute(0, 2, 1))  # [batch_size, embed_dim, seq_length]
        x = F.relu(x)
        x = self.conv2(x)  # [batch_size, embed_dim, seq_length]
        x = F.relu(x)
        x = x.permute(0, 2, 1)  # [batch_size, seq_length, embed_dim]
        x = x.to(torch.float32)
        x_fft = torch.fft.fft(x, dim=1)
        N = int(self.compression_ratio * (seq_length // 2 + 1))
        x_fft_compressed = x_fft[:, :N, :]  # [batch_size, N, embed_dim]
        x_ifft = torch.fft.ifft(x_fft_compressed, n=None, dim=1).real  # [batch_size, N, embed_dim]
        x_ifft = x_ifft.to(x.dtype)
        x = self.proj(x_ifft)  # [batch_size, N, embed_dim]
        return x

class EnhancedLocalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=256, bidirectional=True):
        super(EnhancedLocalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.bidirectional = bidirectional
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, L, C = x.shape
        pad_l = (self.window_size - L % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_l))  # [B, L_padded, C]
        _, L_padded, _ = x.shape
        
        qkv = self.qkv(x).reshape(B, L_padded, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, L_padded, head_dim]
        
        if self.bidirectional:
            overlapping_size = self.window_size // 2  # 128
            step = overlapping_size  # 128
            window_size = self.window_size  # 256
            # Realizar un solo unfold con step=128 para crear ventanas superpuestas
            q = q.unfold(2, window_size, step).contiguous()  # [B, num_heads, num_windows, window_size, head_dim]
            k = k.unfold(2, window_size, step).contiguous()
            v = v.unfold(2, window_size, step).contiguous()
        else:
            q = q.unfold(2, self.window_size, self.window_size).contiguous()  # [B, num_heads, num_windows, window_size, head_dim]
            k = k.unfold(2, self.window_size, self.window_size).contiguous()
            v = v.unfold(2, self.window_size, self.window_size).contiguous()
        
        # Calcular la atención
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, num_heads, num_windows, window_size, window_size]
        attn = F.softmax(attn, dim=-1)
        
        # Aplicar la atención a los valores
        x = (attn @ v).reshape(B, self.num_heads, -1, self.head_dim).permute(0, 2, 1, 3).reshape(B, -1, C)  # [B, L_padded, C]
        x = self.out(x)  # [B, L_padded, C]
        
        return x[:, :L]  # [B, L, C]

class DilatedConvolution(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(DilatedConvolution, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding='same', dilation=dilation)
        
    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)

class ImprovedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, num_experts, expert_dim, window_size=256, bidirectional=True):
        super(ImprovedTransformerBlock, self).__init__()
        self.attention = EnhancedLocalAttention(embed_dim, num_heads, window_size, bidirectional)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dilated_conv = DilatedConvolution(embed_dim, kernel_size=3, dilation=2)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.moe = MoELayer(embed_dim, embed_dim, num_experts, expert_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ff_layer = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        
    def forward(self, x):
        return checkpoint(self._forward, x)
    
    def _forward(self, x):
        print(f"Input to ImprovedTransformerBlock: {x.shape}")
        x = x + self.attention(self.norm1(x))
        print(f"After attention: {x.shape}")
        x = x + self.dilated_conv(self.norm2(x))
        print(f"After dilated convolution: {x.shape}")
        x = x + self.moe(self.norm3(x))
        print(f"After MoE: {x.shape}")
        x = x + self.ff_layer(x)
        print(f"After feed-forward: {x.shape}")
        return x

class BidirectionalEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length=2048, compression_ratio=0.5, num_layers=4, num_heads=8, ff_hidden_dim=1024, window_size=256):
        super(BidirectionalEncoder, self).__init__()
        self.embedding = LiquidEmbedding(vocab_size, embed_dim, max_length, compression_ratio)
        self.layers = nn.ModuleList([
            ImprovedTransformerBlock(embed_dim, num_heads, ff_hidden_dim, num_experts=2, expert_dim=256, window_size=window_size, bidirectional=True)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.embedding(x)  # [batch_size, N=64, embed_dim=256]
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return x

class LiquidFoundationModelOptimized(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_layers=4, num_heads=8, ff_hidden_dim=1024,
                 num_experts=2, expert_dim=256, max_length=2048, window_size=256, compression_ratio=0.5):
        super(LiquidFoundationModelOptimized, self).__init__()
        self.encoder = BidirectionalEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_length=max_length,
            compression_ratio=compression_ratio,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            window_size=window_size
        )
        self.decoder_embedding = LiquidEmbedding(vocab_size, embed_dim, max_length, compression_ratio)
        self.decoder_layers = nn.ModuleList([
            ImprovedTransformerBlock(embed_dim, num_heads, ff_hidden_dim, num_experts, expert_dim, window_size, bidirectional=False)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.max_length = max_length
        self.compression_ratio = compression_ratio

    def forward(self, encoder_input_ids, decoder_input_ids):
        # Encoder
        encoder_output = self.encoder(encoder_input_ids)  # [batch_size, N=64, embed_dim=256]
        
        # Decoder
        decoder_embeddings = self.decoder_embedding(decoder_input_ids)  # [batch_size, N=64, embed_dim=256]
        for layer in self.decoder_layers:
            decoder_embeddings = layer(decoder_embeddings)
        decoder_embeddings = self.layer_norm(decoder_embeddings)
        logits = self.output_layer(decoder_embeddings)  # [batch_size, N=64, vocab_size]
        logits = F.interpolate(logits.permute(0, 2, 1), size=decoder_input_ids.size(1), mode='linear', align_corners=False)
        return logits.permute(0, 2, 1)  # [batch_size, seq_length, vocab_size]

from sklearn.model_selection import train_test_split

def prepare_data(max_samples=None, val_size=0.1):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    special_tokens = {'pad_token': '[PAD]', 'eos_token': '<EOS>', 'bos_token': '<BOS>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens)
    print(f"Se agregaron {num_added_toks} tokens especiales al tokenizer.")
    
    dataset = load_dataset("gsm8k", "main")

    def preprocess(example):
        question = example['question']
        answer = example['answer']
        combined_text = f"{tokenizer.bos_token} Instrucción: Resuelve el siguiente problema matemático.\nEntrada: {question}\nRazonamiento: {answer} {tokenizer.eos_token}"
        tokens = tokenizer(combined_text, truncation=True, max_length=2048, padding='max_length')
        return tokens

    tokenized_dataset = dataset['train'].map(preprocess, batched=False)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    if max_samples is not None and max_samples < len(tokenized_dataset):
        tokenized_dataset = tokenized_dataset.select(range(max_samples))

    train_val_dataset = tokenized_dataset.train_test_split(test_size=val_size)

    return tokenizer, train_val_dataset

def calculate_perplexity(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in data_loader:
            encoder_input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['input_ids'].to(device)  # Asumiendo que la tarea es generación de texto
            outputs = model(encoder_input_ids, decoder_input_ids)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = decoder_input_ids.reshape(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, num_epochs, accumulation_steps=4):
    writer = SummaryWriter()
    best_val_loss = float('inf')
    patience = 3
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Entrenando Epoch {epoch + 1}")

        for batch_idx, batch in loop:
            encoder_input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['input_ids'].to(device)  # Ajusta según tu tarea

            # Ajuste: Eliminamos 'device_type' de autocast y especificamos solo dtype
            with autocast(dtype=torch.float16 if device.type == 'cuda' else torch.float32):
                outputs = model(encoder_input_ids, decoder_input_ids)
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = decoder_input_ids.reshape(-1)
                loss = criterion(outputs, targets)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

            if (batch_idx + 1) % 100 == 0:
                loop.set_postfix(loss=loss.item() * accumulation_steps)

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        val_perplexity = calculate_perplexity(model, val_loader, criterion, device)
        writer.add_scalar('Perplexity/val', val_perplexity, epoch)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
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

        scheduler.step()
        torch.cuda.empty_cache()

    writer.close()

@torch.no_grad()
def unified_generate(model, tokenizer, prompt, device, reasoning=True, max_length=512, beam_width=5, temperature=1.0, top_p=0.9, repetition_penalty=1.2, max_step_tokens=70, max_answer_tokens=30, top_k=50, num_steps=4, max_attempts=4, num_iterations=3):
    """
    Método unificado para generar respuestas con refinamiento iterativo y memoria.
    
    - reasoning: Si es True, utiliza un enfoque de razonamiento paso a paso (Chain of Thought).
    - num_iterations: Número de iteraciones para refinar la generación.
    """
    if not reasoning:
        raise NotImplementedError("El modo general ha sido eliminado. Utiliza el modo 'cot' para generación.")

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
            num_steps=num_steps, 
            max_attempts=max_attempts,
            beam_width=beam_width,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        print(f"Coherencia de la generación {iteration + 1}: {coherence_score:.4f}")

        if coherence_score > best_overall_coherence:
            best_overall_response = response
            best_overall_cot_steps = cot_steps
            best_overall_coherence = coherence_score

        # Actualizar el prompt para la siguiente iteración incluyendo la respuesta actual
        prompt = f"{prompt}\nRefinamiento {iteration + 1}: {response}\n"

    return best_overall_response, best_overall_cot_steps, best_overall_coherence

def generate_cot(model, tokenizer, prompt, device, max_step_tokens=70, max_answer_tokens=30, temperature=0.7, top_k=50, num_steps=4, max_attempts=4, beam_width=5, top_p=0.9, repetition_penalty=1.2):
    """
    Genera soluciones matemáticas utilizando un enfoque de Chain of Thought (CoT).
    Incluye generación de pasos de razonamiento intermedios y selección basada en coherencia.
    """
    model.eval()
    best_response = ""
    best_cot_steps = []
    best_coherence_score = float('-inf')

    for attempt in range(max_attempts):
        # Tokenizar y preparar la entrada para el encoder
        encoder_input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device).long()
        
        # Inicializar el decoder_input_ids con el BOS token
        decoder_input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device).long()
        
        cot_steps = []
        total_tokens = 0

        try:
            # Generar pasos de Chain of Thought
            for step in range(num_steps):
                sequences = [[decoder_input_ids, 0.0]]  # Empezamos con la secuencia de entrada del decoder

                for _ in range(max_step_tokens):
                    all_candidates = []
                    for seq, score in sequences:
                        if seq[0, -1].item() == tokenizer.eos_token_id:
                            all_candidates.append((seq, score))
                            continue
                        with autocast(dtype=torch.float16 if device.type == 'cuda' else torch.float32):
                            outputs = model(encoder_input_ids, seq)  # [batch_size, seq_length, vocab_size]
                            logits = outputs[:, -1, :] / temperature
                            logits = top_p_sampling(logits, top_p)
                            logits = apply_repetition_penalty(logits, seq, repetition_penalty)
                            probs = F.softmax(logits, dim=-1)
                            topk_probs, topk_indices = torch.topk(probs, beam_width)

                        for k in range(beam_width):
                            next_token = topk_indices[0, k].unsqueeze(0).unsqueeze(0)  # [1, 1]
                            next_score = score - torch.log(topk_probs[0, k]).item()
                            candidate_seq = torch.cat([seq, next_token], dim=1)  # [1, seq_length+1]
                            all_candidates.append((candidate_seq, next_score))

                    # Seleccionar las mejores secuencias
                    ordered = sorted(all_candidates, key=lambda tup: tup[1])
                    sequences = ordered[:beam_width]

                    # Verificar si todas las secuencias han generado el token EOS
                    if all(seq[0, -1].item() == tokenizer.eos_token_id for seq, _ in sequences):
                        break

                # Seleccionar la mejor secuencia generada para el paso actual
                best_seq = sequences[0][0]
                step_text = tokenizer.decode(best_seq[0], skip_special_tokens=True)

                if step_text.strip():
                    cot_steps.append({
                        "step": f"Step {step + 1}",
                        "text": step_text,
                    })

                decoder_input_ids = best_seq  # Actualizar el decoder_input_ids para el siguiente paso

            # Generar respuesta final usando beam search
            sequences = [[decoder_input_ids, 0.0]]
            for _ in range(max_answer_tokens):
                all_candidates = []
                for seq, score in sequences:
                    if seq[0, -1].item() == tokenizer.eos_token_id:
                        all_candidates.append((seq, score))
                        continue
                    with autocast(dtype=torch.float16 if device.type == 'cuda' else torch.float32):
                        outputs = model(encoder_input_ids, seq)
                        logits = outputs[:, -1, :] / temperature
                        logits = top_p_sampling(logits, top_p)
                        logits = apply_repetition_penalty(logits, seq, repetition_penalty)
                        probs = F.softmax(logits, dim=-1)
                        topk_probs, topk_indices = torch.topk(probs, beam_width)

                for k in range(beam_width):
                    next_token = topk_indices[0, k].unsqueeze(0).unsqueeze(0)
                    next_score = score - torch.log(topk_probs[0, k]).item()
                    candidate_seq = torch.cat([seq, next_token], dim=1)
                    all_candidates.append((candidate_seq, next_score))

                # Seleccionar las mejores secuencias
                ordered = sorted(all_candidates, key=lambda tup: tup[1])
                sequences = ordered[:beam_width]

                # Verificar si todas las secuencias han generado el token EOS
                if all(seq[0, -1].item() == tokenizer.eos_token_id for seq, _ in sequences):
                    break

            best_seq = sequences[0][0]
            response = tokenizer.decode(best_seq[0], skip_special_tokens=True)

            # Calcular coherencia entre la pregunta, los pasos de razonamiento y la respuesta final
            coherence_score = calculate_coherence(prompt, [step["text"] for step in cot_steps], response)

            if coherence_score > best_coherence_score:
                best_response = response
                best_cot_steps = cot_steps
                best_coherence_score = coherence_score

        except RuntimeError as e:
            print(f"Error durante la generación (intento {attempt + 1}): {e}")
            continue

    return best_response, best_cot_steps, best_coherence_score

def top_p_sampling(logits, p=0.9):
    """
    Implementa Top-p (nucleus) sampling para filtrar tokens menos probables.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Eliminar tokens con probabilidad acumulada por encima del umbral
    sorted_indices_to_remove = cumulative_probs > p
    # Desplazar hacia la derecha para mantener al menos un token por encima del umbral
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits

def apply_repetition_penalty(logits, sequence, penalty=1.2):
    """
    Aplica una penalización a los tokens que ya han sido generados para evitar repeticiones.
    """
    for token_id in set(sequence[0].tolist()):
        logits[:, token_id] /= penalty
    return logits

def calculate_coherence(question, cot_steps, response):
    """
    Calcula una puntuación de coherencia basada en la presencia de elementos matemáticos,
    la longitud y complejidad de los pasos de razonamiento, y la precisión de la respuesta final.
    """
    full_text = question + " " + " ".join(cot_steps) + " " + response
    
    # Verificar la presencia de números y operaciones matemáticas
    math_elements = re.findall(r'\d+|[\+\-\*/\^]', full_text)
    math_score = len(math_elements) / len(full_text.split()) if full_text.split() else 0
    
    # Verificar la longitud y complejidad de los pasos
    step_scores = [len(step.split()) for step in cot_steps]
    step_score = sum(step_scores) / len(step_scores) if step_scores else 0
    
    # Verificar si la respuesta es un número
    response_score = 1 if response.strip().isdigit() else 0
    
    # Combinar las puntuaciones
    coherence_score = (math_score + step_score + response_score) / 3
    
    return coherence_score

def main(max_samples=1000):
    tokenizer, train_val_dataset = prepare_data(max_samples)
    
    VOCAB_SIZE = len(tokenizer)
    EMBED_DIM = 256
    NUM_LAYERS = 4
    NUM_HEADS = 8
    FF_HIDDEN_DIM = 1024
    NUM_EXPERTS = 2
    EXPERT_DIM = 256
    MAX_LENGTH = 2048
    WINDOW_SIZE = 256
    COMPRESSION_RATIO = 0.5  # Aumentar compression_ratio a 0.5
    BATCH_SIZE = 2
    NUM_EPOCHS = 3
    ACCUMULATION_STEPS = 4

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
        compression_ratio=COMPRESSION_RATIO
    ).to(device)

    # Actualizar el tamaño del embedding si se agregaron tokens especiales
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is not None:
        model.encoder.embedding.token_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM).to(device)
        model.output_layer = nn.Linear(EMBED_DIM, VOCAB_SIZE).to(device)
        print("Se actualizó el tamaño del embedding para tokens especiales.")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Inicializar GradScaler sin argumentos incompatibles
    scaler = GradScaler()

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, NUM_EPOCHS, ACCUMULATION_STEPS)

    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = main(max_samples=4000)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    
    cot_prompts = [
        "Instrucción: Resuelve el siguiente problema matemático.\nEntrada: Si una bicicleta cuesta $120 y pago con un billete de $200, ¿cuánto cambio recibiré?\nRazonamiento:",
        "Instrucción: Explica el teorema de Pitágoras.\nEntrada: \nRazonamiento:",
    ]

    # Generar soluciones matemáticas con Chain of Thought usando la función unificada
    print("Generando soluciones matemáticas con Chain of Thought:\n")
    
    for question in cot_prompts:
        response, cot_steps, coherence_score = unified_generate(
            model, tokenizer, question, device, 
            reasoning=True,
            max_step_tokens=70, 
            max_answer_tokens=30, 
            temperature=0.7, 
            top_k=50, 
            num_steps=4, 
            max_attempts=4,
            beam_width=5,
            top_p=0.9,
            repetition_penalty=1.2,
            num_iterations=3  # Número de iteraciones para refinamiento
        )
        print(f"Pregunta:\n{question}\nRespuesta:\n{response}")
        print("Pasos de razonamiento:")
        for step in cot_steps:
            print(f"{step['step']}: {step['text']}")
        print(f"Puntuación de coherencia: {coherence_score:.4f}")
        print(f"Total de tokens generados: {sum(len(step['text'].split()) for step in cot_steps) + len(response.split())}\n{'-'*50}\n")
