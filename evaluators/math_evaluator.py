from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import torch
import sympy
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

class MathEvaluator:
    """
    Clase para evaluar diferentes aspectos matemáticos de las respuestas generadas.
    """

    def __init__(self):
        self.math_lm = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").to(device)
        self.math_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def evaluate_coherence(self, question, cot_steps, response):
        """
        Evalúa la coherencia de la respuesta generada.

        Args:
            question (str): La pregunta original.
            cot_steps (list): Lista de pasos de razonamiento.
            response (str): La respuesta generada.

        Returns:
            float: Puntuación de coherencia.
        """
        full_text = question + " ".join([step["text"] for step in cot_steps]) + response
        math_elements = re.findall(r'\d+|[\+\-\*/\^]', full_text)
        math_score = len(math_elements) / len(full_text.split()) if full_text.split() else 0
        step_scores = [len(step["text"].split()) for step in cot_steps]
        step_score = sum(step_scores) / len(step_scores) if step_scores else 0
        response_score = 1 if response.strip().isdigit() else 0
        coherence_score = (math_score + step_score + response_score) / 3
        return coherence_score

    def evaluate_math_precision(self, expected, generated):
        """
        Evalúa la precisión matemática comparando la expresión esperada con la generada.

        Args:
            expected (str): Expresión matemática esperada.
            generated (str): Expresión matemática generada.

        Returns:
            bool: True si las expresiones son equivalentes, False en caso contrario.
        """
        try:
            expected_expr = sympy.sympify(expected)
            generated_expr = sympy.sympify(generated)
            return sympy.simplify(expected_expr - generated_expr) == 0
        except:
            return False

    def evaluate_step_relevance(self, question, steps):
        """
        Evalúa la relevancia de los pasos de razonamiento con respecto a la pregunta.

        Args:
            question (str): La pregunta original.
            steps (list): Lista de pasos de razonamiento.

        Returns:
            float: Puntuación de relevancia de los pasos.
        """
        if not steps:
            return 0.0
        question_embedding = torch.tensor(self.get_embedding(question)).to(device)
        step_embeddings = torch.tensor([self.get_embedding(step["text"]) for step in steps]).to(device)
        question_norm = question_embedding / question_embedding.norm(dim=-1, keepdim=True)
        step_norms = step_embeddings / step_embeddings.norm(dim=-1, keepdim=True)
        similarities = torch.mm(step_norms, question_norm.unsqueeze(-1)).squeeze(-1)
        return similarities.mean().item()

    def evaluate_reasoning_complexity(self, steps):
        """
        Evalúa la complejidad del razonamiento basado en los pasos proporcionados.

        Args:
            steps (list): Lista de pasos de razonamiento.

        Returns:
            float: Puntuación de complejidad del razonamiento.
        """
        total_operations = sum(len(re.findall(r'[\+\-\*/\^]', step["text"])) for step in steps)
        return total_operations / len(steps) if steps else 0

    def evaluate_numerical_consistency(self, steps, answer):
        """
        Evalúa la consistencia numérica entre los pasos y la respuesta.

        Args:
            steps (list): Lista de pasos de razonamiento.
            answer (str): Respuesta generada.

        Returns:
            float: Puntuación de consistencia numérica.
        """
        numbers = re.findall(r'\d+', ' '.join([step["text"] for step in steps]) + ' ' + answer)
        unique_numbers = set(numbers)
        return len(unique_numbers) / len(numbers) if numbers else 1.0

    def evaluate_concept_coverage(self, question, answer):
        """
        Evalúa la cobertura de conceptos entre la pregunta y la respuesta.

        Args:
            question (str): La pregunta original.
            answer (str): La respuesta generada.

        Returns:
            float: Puntuación de cobertura de conceptos.
        """
        question_concepts = set(re.findall(r'\b\w+\b', question.lower()))
        answer_concepts = set(re.findall(r'\b\w+\b', answer.lower()))
        return len(question_concepts.intersection(answer_concepts)) / len(question_concepts) if question_concepts else 0

    def evaluate_explainability(self, question, steps, answer):
        """
        Evalúa la explicabilidad de la respuesta generada.

        Args:
            question (str): La pregunta original.
            steps (list): Lista de pasos de razonamiento.
            answer (str): La respuesta generada.

        Returns:
            float: Puntuación de explicabilidad.
        """
        full_explanation = " ".join([step["text"] for step in steps]) + " " + answer
        question_words = set(word_tokenize(question.lower()))
        explanation_words = set(word_tokenize(full_explanation.lower()))
        return len(question_words.intersection(explanation_words)) / len(question_words) if question_words else 0

    def evaluate_solution_efficiency(self, question, steps):
        """
        Evalúa la eficiencia de la solución basada en los pasos proporcionados.

        Args:
            question (str): La pregunta original.
            steps (list): Lista de pasos de razonamiento.

        Returns:
            float: Puntuación de eficiencia de la solución.
        """
        ideal_steps = self.estimate_ideal_steps(question)
        return max(0, 1 - abs(ideal_steps - len(steps)) / ideal_steps)

    def evaluate_reasoning_adaptability(self, questions, answers):
        """
        Evalúa la adaptabilidad del razonamiento a diferentes preguntas.

        Args:
            questions (list): Lista de preguntas.
            answers (list): Lista de respuestas generadas.

        Returns:
            float: Puntuación de adaptabilidad del razonamiento.
        """
        complexities = [self.calculate_problem_complexity(q) for q in questions]
        answer_qualities = [self.evaluate_answer_quality(q, a) for q, a in zip(questions, answers)]
        return np.mean([q / c for q, c in zip(answer_qualities, complexities) if c != 0])

    def evaluate_f1_score(self, pred_text, label_text, tolerance=1e-6, with_steps=False):
        """
        Evalúa la puntuación F1 entre el texto predicho y el texto de etiqueta.

        Args:
            pred_text (str): Texto predicho.
            label_text (str): Texto de etiqueta.
            tolerance (float): Tolerancia para la comparación numérica.
            with_steps (bool): Si se debe considerar los pasos de razonamiento.

        Returns:
            float: Puntuación F1.
        """
        if with_steps:
            pred_steps = self.extract_step_numbers(pred_text)
            label_steps = self.extract_step_numbers(label_text)
            return np.mean([self.calculate_f1(str(p), str(l), tolerance) for p, l in zip(pred_steps, label_steps)])
        else:
            return self.calculate_f1(pred_text, label_text, tolerance)

    def calculate_f1(self, pred_text, label_text, tolerance):
        """
        Calcula la puntuación F1 entre el texto predicho y el texto de etiqueta.

        Args:
            pred_text (str): Texto predicho.
            label_text (str): Texto de etiqueta.
            tolerance (float): Tolerancia para la comparación numérica.

        Returns:
            float: Puntuación F1.
        """
        pred_nums = torch.tensor(self.extract_numbers(pred_text), dtype=torch.float32, device=device)
        label_nums = torch.tensor(self.extract_numbers(label_text), dtype=torch.float32, device=device)

        if not pred_nums.numel() or not label_nums.numel():
            return 0.0

        diffs = torch.abs(pred_nums.unsqueeze(1) - label_nums)
        true_positives = (diffs <= tolerance).any(dim=1).sum().item()

        precision = true_positives / pred_nums.numel() if pred_nums.numel() else 0
        recall = true_positives / label_nums.numel() if label_nums.numel() else 0

        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    def evaluate_f1_with_complexity(self, pred_text, label_text):
        """
        Evalúa la puntuación F1 considerando la complejidad del problema.

        Args:
            pred_text (str): Texto predicho.
            label_text (str): Texto de etiqueta.

        Returns:
            float: Puntuación F1 ajustada por complejidad.
        """
        complexity = self.calculate_problem_complexity(label_text)
        base_f1 = self.evaluate_f1_score(pred_text, label_text)
        return base_f1 * (1 - 1 / (np.log(complexity + 1) + 1))

    def evaluate_f1_with_explanation(self, pred_text, label_text):
        """
        Evalúa la puntuación F1 considerando la explicación proporcionada.

        Args:
            pred_text (str): Texto predicho.
            label_text (str): Texto de etiqueta.

        Returns:
            float: Puntuación F1 ajustada por explicación.
        """
        numeric_f1 = self.evaluate_f1_score(pred_text, label_text)
        pred_words = pred_text.lower().split()
        label_words = label_text.lower().split()
        bleu_score = sentence_bleu([label_words], pred_words)
        return (numeric_f1 + bleu_score) / 2

    def get_embedding(self, text):
        """
        Obtiene el embedding del texto utilizando el modelo de lenguaje matemático.

        Args:
            text (str): Texto para el cual se desea obtener el embedding.

        Returns:
            Tensor: Embedding del texto.
        """
        inputs = self.math_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = self.math_lm(**inputs)
        return outputs.logits

    def estimate_ideal_steps(self, question):
        """
        Estima el número ideal de pasos de razonamiento para una pregunta dada.

        Args:
            question (str): La pregunta original.

        Returns:
            int: Número estimado de pasos ideales.
        """
        return max(2, min(5, self.calculate_problem_complexity(question)))

    def calculate_problem_complexity(self, text):
        """
        Calcula la complejidad de un problema basado en el texto proporcionado.

        Args:
            text (str): Texto del problema.

        Returns:
            int: Complejidad calculada del problema.
        """
        operations = re.findall(r'[\+\-\*/\^]', text)
        return len(operations) + 1

    def evaluate_answer_quality(self, question, answer):
        """
        Evalúa la calidad de la respuesta en función de la cobertura de conceptos.

        Args:
            question (str): La pregunta original.
            answer (str): La respuesta generada.

        Returns:
            float: Puntuación de calidad de la respuesta.
        """
        return self.evaluate_concept_coverage(question, answer)

    def extract_numbers(self, text):
        """
        Extrae números del texto proporcionado.

        Args:
            text (str): Texto del cual extraer números.

        Returns:
            list: Lista de números extraídos.
        """
        return [float(num) for num in re.findall(r'-?\d+\.?\d*', text)]

    def extract_step_numbers(self, text):
        """
        Extrae números de los pasos de razonamiento en el texto proporcionado.

        Args:
            text (str): Texto del cual extraer números de pasos.

        Returns:
            list: Lista de números extraídos de los pasos.
        """
        steps = text.split('Step')
        return [self.extract_numbers(step) for step in steps if step.strip()]

    def comprehensive_evaluation(self, question, steps, answer, label_text=None):
        """
        Realiza una evaluación integral de la respuesta generada.

        Args:
            question (str): La pregunta original.
            steps (list): Lista de pasos de razonamiento.
            answer (str): La respuesta generada.
            label_text (str, optional): Texto de etiqueta para comparación.

        Returns:
            dict: Resultados de la evaluación integral.
        """
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