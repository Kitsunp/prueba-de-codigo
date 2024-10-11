# prueba/models/math_evaluator.py

"""
MathEvaluator module for evaluating mathematical coherence and precision.
"""

import re
import sympy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np

class MathEvaluator:
    def __init__(self):
        """
        Initialize the MathEvaluator with a pre-trained model and tokenizer.
        """
        self.math_lm = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").to(device)
        self.math_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def evaluate_coherence(self, question, cot_steps, response):
        """
        Evaluate the coherence of the generated response.

        Args:
            question (str): The input question.
            cot_steps (list): List of chain-of-thought steps.
            response (str): The generated response.

        Returns:
            float: Coherence score.
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
        Evaluate the precision of the generated mathematical expression.

        Args:
            expected (str): The expected mathematical expression.
            generated (str): The generated mathematical expression.

        Returns:
            bool: True if the expressions are equivalent, False otherwise.
        """
        try:
            expected_expr = sympy.sympify(expected)
            generated_expr = sympy.sympify(generated)
            return sympy.simplify(expected_expr - generated_expr) == 0
        except:
            return False

    def evaluate_step_relevance(self, question, steps):
        """
        Evaluate the relevance of each step in the chain-of-thought.

        Args:
            question (str): The input question.
            steps (list): List of chain-of-thought steps.

        Returns:
            float: Average relevance score of the steps.
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
        Evaluate the complexity of reasoning based on the number of operations.

        Args:
            steps (list): List of chain-of-thought steps.

        Returns:
            float: Average number of operations per step.
        """
        total_operations = sum(len(re.findall(r'[\+\-\*/\^]', step["text"])) for step in steps)
        return total_operations / len(steps) if steps else 0

    def evaluate_numerical_consistency(self, steps, answer):
        """
        Evaluate the numerical consistency between steps and the final answer.

        Args:
            steps (list): List of chain-of-thought steps.
            answer (str): The final answer.

        Returns:
            float: Consistency score.
        """
        numbers = re.findall(r'\d+', ' '.join([step["text"] for step in steps]) + ' ' + answer)
        unique_numbers = set(numbers)
        return len(unique_numbers) / len(numbers) if numbers else 1.0

    def evaluate_concept_coverage(self, question, answer):
        """
        Evaluate the coverage of concepts from the question in the answer.

        Args:
            question (str): The input question.
            answer (str): The generated answer.

        Returns:
            float: Coverage score.
        """
        question_concepts = set(re.findall(r'\b\w+\b', question.lower()))
        answer_concepts = set(re.findall(r'\b\w+\b', answer.lower()))
        return len(question_concepts.intersection(answer_concepts)) / len(question_concepts) if question_concepts else 0

    def evaluate_explainability(self, question, steps, answer):
        """
        Evaluate the explainability of the generated response.

        Args:
            question (str): The input question.
            steps (list): List of chain-of-thought steps.
            answer (str): The generated answer.

        Returns:
            float: Explainability score.
        """
        full_explanation = " ".join([step["text"] for step in steps]) + " " + answer
        question_words = set(word_tokenize(question.lower()))
        explanation_words = set(word_tokenize(full_explanation.lower()))
        return len(question_words.intersection(explanation_words)) / len(question_words) if question_words else 0

    def evaluate_solution_efficiency(self, question, steps):
        """
        Evaluate the efficiency of the solution based on the number of steps.

        Args:
            question (str): The input question.
            steps (list): List of chain-of-thought steps.

        Returns:
            float: Efficiency score.
        """
        ideal_steps = self.estimate_ideal_steps(question)
        return max(0, 1 - abs(ideal_steps - len(steps)) / ideal_steps)

    def evaluate_reasoning_adaptability(self, questions, answers):
        """
        Evaluate the adaptability of reasoning across multiple questions.

        Args:
            questions (list): List of input questions.
            answers (list): List of generated answers.

        Returns:
            float: Adaptability score.
        """
        complexities = [self.calculate_problem_complexity(q) for q in questions]
        answer_qualities = [self.evaluate_answer_quality(q, a) for q, a in zip(questions, answers)]
        return np.mean([q / c for q, c in zip(answer_qualities, complexities) if c != 0])

    def evaluate_f1_score(self, pred_text, label_text, tolerance=1e-6, with_steps=False):
        """
        Evaluate the F1 score between predicted and label text.

        Args:
            pred_text (str): The predicted text.
            label_text (str): The label text.
            tolerance (float): Tolerance for numerical comparison.
            with_steps (bool): Whether to consider steps in evaluation.

        Returns:
            float: F1 score.
        """
        if with_steps:
            pred_steps = self.extract_step_numbers(pred_text)
            label_steps = self.extract_step_numbers(label_text)
            return np.mean([self.calculate_f1(str(p), str(l), tolerance) for p, l in zip(pred_steps, label_steps)])
        else:
            return self.calculate_f1(pred_text, label_text, tolerance)

    def calculate_f1(self, pred_text, label_text, tolerance):
        """
        Calculate the F1 score for numerical values in the text.

        Args:
            pred_text (str): The predicted text.
            label_text (str): The label text.
            tolerance (float): Tolerance for numerical comparison.

        Returns:
            float: F1 score.
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
        Evaluate the F1 score considering problem complexity.

        Args:
            pred_text (str): The predicted text.
            label_text (str): The label text.

        Returns:
            float: F1 score adjusted for complexity.
        """
        complexity = self.calculate_problem_complexity(label_text)
        base_f1 = self.evaluate_f1_score(pred_text, label_text)
        return base_f1 * (1 - 1 / (np.log(complexity + 1) + 1))

    def evaluate_f1_with_explanation(self, pred_text, label_text):
        """
        Evaluate the F1 score considering explanation quality.

        Args:
            pred_text (str): The predicted text.
            label_text (str): The label text.

        Returns:
            float: F1 score adjusted for explanation quality.
        """
        numeric_f1 = self.evaluate_f1_score(pred_text, label_text)
        pred_words = pred_text.lower().split()
        label_words = label_text.lower().split()
        bleu_score = sentence_bleu([label_words], pred_words)
        return (numeric_f1 + bleu_score) / 2

    def get_embedding(self, text):
        """
        Get the embedding for a given text.

        Args:
            text (str): The input text.

        Returns:
            Tensor: Embedding tensor.
        """
        inputs = self.math_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = self.math_lm(**inputs)
        return outputs.logits

    def estimate_ideal_steps(self, question):
        """
        Estimate the ideal number of steps for solving a question.

        Args:
            question (str): The input question.

        Returns:
            int: Estimated number of ideal steps.
        """
        return max(2, min(5, self.calculate_problem_complexity(question)))

    def calculate_problem_complexity(self, text):
        """
        Calculate the complexity of a problem based on operations.

        Args:
            text (str): The input text.

        Returns:
            int: Complexity score.
        """
        operations = re.findall(r'[\+\-\*/\^]', text)
        return len(operations) + 1

    def evaluate_answer_quality(self, question, answer):
        """
        Evaluate the quality of the answer based on concept coverage.

        Args:
            question (str): The input question.
            answer (str): The generated answer.

        Returns:
            float: Quality score.
        """
        return self.evaluate_concept_coverage(question, answer)

    def extract_numbers(self, text):
        """
        Extract numerical values from the text.

        Args:
            text (str): The input text.

        Returns:
            list: List of extracted numbers.
        """
        return [float(num) for num in re.findall(r'-?\d+\.?\d*', text)]

    def extract_step_numbers(self, text):
        """
        Extract numbers from each step in the text.

        Args:
            text (str): The input text.

        Returns:
            list: List of numbers for each step.
        """
        steps = text.split('Step')
        return [self.extract_numbers(step) for step in steps if step.strip()]

    def comprehensive_evaluation(self, question, steps, answer, label_text=None):
        """
        Perform a comprehensive evaluation of the generated response.

        Args:
            question (str): The input question.
            steps (list): List of chain-of-thought steps.
            answer (str): The generated answer.
            label_text (str, optional): The label text for comparison.

        Returns:
            dict: Dictionary of evaluation metrics.
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