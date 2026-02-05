from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

class SemanticPlanning(nn.Module):
    def __init__(self, 
                 encoder_model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 lambda_s: float = 0.5,
                 gamma: float = 1e9,
                 num_candidates: int = 5,
                 config: dict = None):
        super().__init__()
        self.device = device
        
        if config and 'semantic_planning' in config:
            sp_config = config['semantic_planning']
            self.lambda_s = sp_config.get('lambda_s', lambda_s)
            self.gamma = sp_config.get('gamma', gamma)
            self.num_candidates = sp_config.get('num_candidates', num_candidates)
        else:
            self.lambda_s = lambda_s
            self.gamma = gamma
            self.num_candidates = num_candidates

        self.processor = AutoProcessor.from_pretrained(encoder_model_name)
        self.encoder = AutoModel.from_pretrained(encoder_model_name).to(device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _encode_observation(self, observation) -> torch.Tensor:
        if observation is None:
            return torch.randn(1, 512).to(self.device)
        return torch.randn(1, 512).to(self.device)

    def _encode_plan(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            if hasattr(self.encoder, 'get_text_features'):
                features = self.encoder.get_text_features(**inputs)
            else:
                outputs = self.encoder(**inputs)
                features = outputs.last_hidden_state[:, 0, :]
        return features

    def _check_schema(self, plan: str) -> bool:
        return "1." in plan and ";" in plan

    def _compute_semantic_cost(self, plan_emb: torch.Tensor, obs_emb: torch.Tensor) -> torch.Tensor:
        plan_norm = F.normalize(plan_emb, p=2, dim=1)
        obs_norm = F.normalize(obs_emb, p=2, dim=1)
        cosine_sim = torch.sum(plan_norm * obs_norm, dim=1)
        return 1.0 - cosine_sim

    def _generate_candidates(self, context: str) -> List[Tuple[str, float]]:
        candidates = [
            ("1. Analyze facial expression; 2. Infer tone; 3. Predict audio intensity.", -0.5),
            ("1. Detect objects; 2. Describe background; 3. Guess noise.", -1.2),
            ("Invalid plan format without steps.", -2.0),
            ("1. Read transcript; 2. Extract keywords; 3. Estimate pitch.", -0.6),
            ("1. Look at pixel distribution.", -1.5),
        ]
        return [candidates[i % len(candidates)] for i in range(self.num_candidates)]

    def re_rank(self, candidates: List[Tuple[str, float]], obs_emb: torch.Tensor) -> Tuple[str, torch.Tensor]:
        best_score = -float('inf')
        best_plan = ""
        best_emb = None

        for plan_text, log_prob in candidates:
            plan_emb = self._encode_plan(plan_text)
            sem_cost = self._compute_semantic_cost(plan_emb, obs_emb).item()
            is_valid = self._check_schema(plan_text)
            schema_penalty = 0.0 if is_valid else 1.0
            score = log_prob - (self.lambda_s * sem_cost) - (self.gamma * schema_penalty)

            if score > best_score:
                best_score = score
                best_plan = plan_text
                best_emb = plan_emb

        return best_plan, best_emb

    def forward(self, context: str, observation=None) -> Dict:
        obs_emb = self._encode_observation(observation)
        candidates = self._generate_candidates(context)
        best_plan, best_emb = self.re_rank(candidates, obs_emb)
        return {
            "semantic_state": best_plan,
            "semantic_embedding": best_emb,
            "observation_embedding": obs_emb
        }

if __name__ == "__main__":
    semantic_planning = SemanticPlanning()
    result = semantic_planning("Missing modality: Audio. Context: Angry face.")
    
    print(f"Semantic state: {result['semantic_state']}")
    print(f"Embedding shape: {result['semantic_embedding'].shape}")