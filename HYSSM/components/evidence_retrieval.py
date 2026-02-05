from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidenceRetrieval(nn.Module):
    def __init__(self, 
                 obs_dim: int = 512, 
                 plan_dim: int = 512, 
                 query_dim: int = 512, 
                 key_dim: int = 512, 
                 value_dim: int = 512, 
                 top_k: int = 5,
                 temperature: float = 0.07,
                 config: dict = None):
        super().__init__()
        
        if config and 'evidence_retrieval' in config:
            er_config = config['evidence_retrieval']
            self.top_k = er_config.get('top_k', top_k)
            self.temperature = er_config.get('temperature', temperature)
        else:
            self.top_k = top_k
            self.temperature = temperature
        
        self.projection = nn.Linear(obs_dim + plan_dim, query_dim)
        self.activation = nn.ReLU()
        self.kb_size = 1000
        
        self.register_buffer("keys", torch.randn(self.kb_size, key_dim))
        self.register_buffer("values", torch.randn(self.kb_size, value_dim))
        self.register_buffer("semantic_embeddings", torch.randn(self.kb_size, plan_dim))

    def _compute_alignment_cost(self, c_S: torch.Tensor, indices: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        u_i = self.semantic_embeddings[indices]
        c_S_norm = F.normalize(c_S, p=2, dim=-1).unsqueeze(1)
        u_i_norm = F.normalize(u_i, p=2, dim=-1)
        cosine_sim = torch.sum(c_S_norm * u_i_norm, dim=-1)
        cost = torch.sum(alpha * (1.0 - cosine_sim), dim=-1)
        return cost.mean()

    def _project_query(self, u_X: torch.Tensor, c_S: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([u_X, c_S], dim=-1)
        q = self.projection(combined)
        return self.activation(q)

    def _retrieve(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(self.keys, p=2, dim=-1)
        scores = torch.matmul(q_norm, k_norm.t()) / self.temperature
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        alpha = F.softmax(topk_scores, dim=-1)
        retrieved_values = self.values[topk_indices]
        E = torch.sum(alpha.unsqueeze(-1) * retrieved_values, dim=1)
        return E, topk_indices, alpha

    def forward(self, u_X: torch.Tensor, c_S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self._project_query(u_X, c_S)
        E, indices, alpha = self._retrieve(q)
        alignment_cost = self._compute_alignment_cost(c_S, indices, alpha)
        return E, indices, alignment_cost

if __name__ == "__main__":
    # Example usage
    batch_size = 4
    obs_dim = 512
    plan_dim = 512
    
    evidence_retrieval = EvidenceRetrieval(obs_dim=obs_dim, plan_dim=plan_dim)
    
    u_X = torch.randn(batch_size, obs_dim)
    c_S = torch.randn(batch_size, plan_dim)
    
    E, indices, cost = evidence_retrieval(u_X, c_S)
    
    print(f"Evidence shape: {E.shape}")
    print(f"Top-K indices shape: {indices.shape}")
    print(f"Alignment cost: {cost.item():.4f}")