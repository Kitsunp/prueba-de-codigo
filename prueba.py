# Importaciones
from evaluators.math_evaluator import MathEvaluator
from monitoring.activation_monitor import ActivationMonitor
from models.liquid_foundation_model_optimized import LiquidFoundationModelOptimized
from utils.helpers import prepare_data, calculate_metrics, OptimizedFocalLoss, train_model, evaluate_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
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
