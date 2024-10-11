import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from datasets import load_dataset
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def prepare_data(max_samples=None, val_size=0.1):
    """
    Prepara y tokeniza el dataset para el entrenamiento y validación.

    Args:
        max_samples (int, optional): Número máximo de muestras a utilizar. Si es None, se utilizan todas las muestras.
        val_size (float, optional): Proporción del dataset a utilizar para validación.

    Returns:
        Tuple: tokenizer y dataset dividido en entrenamiento y validación.
    """
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
    """
    Calcula las métricas de evaluación para el modelo.

    Args:
        model (nn.Module): El modelo a evaluar.
        data_loader (DataLoader): DataLoader para el conjunto de datos de evaluación.
        criterion (nn.Module): Función de pérdida.
        device (torch.device): Dispositivo en el que se ejecuta el modelo.
        tokenizer (GPT2Tokenizer): Tokenizador utilizado para el modelo.

    Returns:
        Tuple: Pérdida promedio y perplejidad.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in data_loader:
            encoder_input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            
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
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity

class OptimizedFocalLoss(nn.Module):
    """
    Implementa una versión optimizada de la pérdida focal con suavizado de etiquetas.
    """
    def __init__(self, alpha=1, gamma=2, ignore_index=-100, label_smoothing=0.1):
        super(OptimizedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """
        Calcula la pérdida focal para las entradas y objetivos dados.

        Args:
            inputs (Tensor): Logits de entrada.
            targets (Tensor): Objetivos de las etiquetas.

        Returns:
            Tensor: Pérdida calculada.
        """
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
                    log_probs = nn.functional.log_softmax(chunk_inputs, dim=-1)
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
    """
    Entrena el modelo utilizando los datos proporcionados.

    Args:
        model (nn.Module): El modelo a entrenar.
        train_loader (DataLoader): DataLoader para el conjunto de datos de entrenamiento.
        val_loader (DataLoader): DataLoader para el conjunto de datos de validación.
        criterion (nn.Module): Función de pérdida.
        optimizer (Optimizer): Optimizador para el entrenamiento.
        scheduler (LRScheduler): Planificador de tasa de aprendizaje.
        scaler (GradScaler): Escalador para entrenamiento de precisión mixta.
        device (torch.device): Dispositivo en el que se ejecuta el modelo.
        num_epochs (int): Número de épocas de entrenamiento.
        accumulation_steps (int, optional): Número de pasos de acumulación de gradientes.
        evaluator (MathEvaluator, optional): Evaluador para métricas adicionales.
        tokenizer (GPT2Tokenizer, optional): Tokenizador utilizado para el modelo.
        monitor (ActivationMonitor, optional): Monitor para activaciones y gradientes.
    """
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
    """
    Evalúa el modelo en el conjunto de datos de validación.

    Args:
        model (nn.Module): El modelo a evaluar.
        val_loader (DataLoader): DataLoader para el conjunto de datos de validación.
        evaluator (MathEvaluator): Evaluador para métricas adicionales.
        tokenizer (GPT2Tokenizer): Tokenizador utilizado para el modelo.
        device (torch.device): Dispositivo en el que se ejecuta el modelo.
        writer (SummaryWriter): Escritor para TensorBoard.
        epoch (int): Época actual de entrenamiento.
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
    """
    Calcula métricas de evaluación para las respuestas generadas.

    Args:
        evaluator (MathEvaluator): Evaluador para métricas adicionales.
        questions (list): Lista de preguntas.
        answers (list): Lista de respuestas generadas.
        labels (list): Lista de respuestas correctas.

    Returns:
        dict: Diccionario con las métricas calculadas.
    """
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
    """
    Imprime las métricas calculadas.

    Args:
        metrics (dict): Diccionario con las métricas calculadas.
    """
    print(f"Coherence: {metrics['coherence']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"BLEU: {metrics['bleu']:.4f}")
    print(f"ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {metrics['rougeL']:.4f}")