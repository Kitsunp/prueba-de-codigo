# prueba/models/moe_layer.py

"""
MoELayer module implementing a Mixture of Experts layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    """
    Mixture of Experts Layer for dynamic routing.
    """

    def __init__(self, input_dim, hidden_dim, num_experts, expert_dim, dropout=0.1, entropy_weight=0.1, top_k=2, dynamic_k=False, max_usage_ratio=0.3):
        """
        Initialize the MoELayer.

        Args:
            input_dim (int): Dimension of the input.
            hidden_dim (int): Hidden dimension for the experts.
            num_experts (int): Total number of experts.
            expert_dim (int): Output dimension of each expert.
            dropout (float): Dropout rate.
            entropy_weight (float): Weight for entropy regularization.
            top_k (int): Initial number of experts to select.
            dynamic_k (bool): Whether to dynamically adjust the number of experts.
            max_usage_ratio (float): Maximum allowed usage ratio for each expert.
        """
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
        """
        Forward pass for MoELayer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_length, input_dim].

        Returns:
            Tuple[Tensor, Tensor]: Processed output and combined loss (entropy + overuse penalty).
        """
        batch_size, seq_length, input_dim = x.size()
        x_flat = x.view(-1, input_dim)
        
        # Calculate routing probabilities
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Entropy regularization
        entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-10), dim=-1).mean()
        entropy_loss = self.entropy_weight * entropy

        # Dynamic adjustment of K (number of experts to select)
        if self.dynamic_k:
            complexity = entropy.detach().item()
            K = max(1, min(self.num_experts, int(self.top_k * (1 + complexity))))
        else:
            K = self.top_k

        # Select the top-K experts
        topk_probs, topk_indices = torch.topk(gate_probs, K, dim=-1)

        # Initialize or reset the expert usage counter
        self.expert_usage_counter = torch.zeros(self.num_experts, device=x.device)

        # Prepare the output tensor
        expert_outputs = torch.zeros(batch_size * seq_length, self.experts[0].out_features, device=x.device)

        # Process the input with the selected experts
        for k in range(K):
            expert_idx = topk_indices[:, k]
            mask = torch.arange(x_flat.size(0), device=x.device).unsqueeze(1) == expert_idx.unsqueeze(1)
            mask = mask.any(dim=1)
            selected_x = x_flat[mask]

            if selected_x.size(0) > 0:
                expert = self.experts[expert_idx[mask][0]]
                expert_output = self.dropout(expert(selected_x))
                expert_outputs[mask] += expert_output * topk_probs[:, k][mask].unsqueeze(1)

                # Update the expert usage counter
                self.expert_usage_counter[expert_idx[mask][0]] += selected_x.size(0)

        # Calculate the overuse penalty
        usage_ratios = self.expert_usage_counter / (batch_size * seq_length)
        overuse_penalty = torch.sum(F.relu(usage_ratios - self.max_usage_ratio))

        # Reshape the output
        output = expert_outputs.view(batch_size, seq_length, -1)

        return output, entropy_loss + overuse_penalty

    def get_expert_usage_stats(self):
        """
        Get usage statistics of the experts.

        Returns:
            List[float] or None: Usage percentages of each expert, or None if no data is available.
        """
        if self.expert_usage_counter is None:
            return None
        total_usage = self.expert_usage_counter.sum().item()
        usage_percentages = (self.expert_usage_counter / total_usage * 100).tolist()
        return usage_percentages