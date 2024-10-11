# prueba/evaluation/metrics.py

"""
Metrics module for evaluating model performance.
"""

import re
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def calculate_metrics(model, data_loader, criterion, device, tokenizer):
    """
    Calculate evaluation metrics for the model.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to perform calculations on.
        tokenizer (Tokenizer): Tokenizer for encoding/decoding text.

    Returns:
        tuple: Average loss and perplexity.
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

def evaluate_coherence(question, cot_steps, response):
    """
    Calculate coherence score for the generated response.

    Args:
        question (str): The input question.
        cot_steps (list): List of chain-of-thought steps.
        response (str): The generated response.

    Returns:
        float: Coherence score.
    """
    full_text = question + " " + " ".join([step["text"] for step in cot_steps]) + " " + response
    math_elements = re.findall(r'\d+|[\+\-\*/\^]', full_text)
    math_score = len(math_elements) / len(full_text.split()) if full_text.split() else 0

    step_scores = [len(step["text"].split()) for step in cot_steps]
    step_score = sum(step_scores) / len(step_scores) if step_scores else 0

    response_score = 1 if response.strip().isdigit() else 0

    coherence_score = (math_score + step_score + response_score) / 3

    return coherence_score

def calculate_evaluation_metrics(evaluator, questions, answers, labels):
    """
    Calculate various evaluation metrics for the model's predictions.

    Args:
        evaluator (MathEvaluator): Instance of MathEvaluator for evaluation.
        questions (list): List of input questions.
        answers (list): List of generated answers.
        labels (list): List of ground truth labels.

    Returns:
        dict: Dictionary containing various evaluation metrics.
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
    Print evaluation metrics.

    Args:
        metrics (dict): Dictionary containing evaluation metrics.
    """
    print(f"Coherence: {metrics['coherence']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"BLEU: {metrics['bleu']:.4f}")
    print(f"ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {metrics['rougeL']:.4f}")