import torch.nn as nn
import torch.nn.functional as F
import torch

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