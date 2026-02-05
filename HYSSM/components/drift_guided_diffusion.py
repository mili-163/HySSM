import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class ZeroAdapter(nn.Module):
    def __init__(self, channels: int, condition_dim: int):
        super().__init__()
        self.linear = nn.Linear(condition_dim, channels)
        # Initialize parameters to zero
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] or [B, L, C]
        # condition: [B, condition_dim]
        cond_feat = self.linear(condition)
        
        # Reshape for broadcasting
        if x.dim() == 4: # Image-like
            cond_feat = cond_feat.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 3: # Sequence-like
            cond_feat = cond_feat.unsqueeze(1)
            
        return x + cond_feat

class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] (Query)
        # context: [B, 1, D_ctx] or [B, L_ctx, D_ctx] (Key/Value)
        
        B, L, D = x.shape
        if context.dim() == 2:
            context = context.unsqueeze(1)
            
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Split heads
        q = q.reshape(B, -1, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.to_out(out) + x # Residual connection

class DenoisingUNet(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 plan_dim: int, 
                 evidence_dim: int,
                 obs_dim: int,
                 base_dim: int = 64,
                 layers_per_block: int = 2):
        super().__init__()
        
        # Simple U-Net structure simulation
        # Layers 0-1: Shallow (Encoder) -> Zero Adapter (Local Statistics)
        # Layers 2-3: Deep (Bottleneck) -> Cross Attention (Global Consistency)
        # Layers 4-5: Shallow (Decoder) -> Zero Adapter (Local Statistics)
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, base_dim),
            nn.SiLU(),
            nn.Linear(base_dim, base_dim)
        )
        
        # Observation injection
        self.obs_proj = nn.Linear(obs_dim, base_dim)

        # Shallow Layers (Encoder)
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, base_dim, 3, padding=1), nn.GroupNorm(8, base_dim), nn.SiLU())
        self.za1 = ZeroAdapter(base_dim, evidence_dim)
        
        # Deep Layers (Bottleneck)
        self.mid = nn.Sequential(nn.Conv2d(base_dim, base_dim*2, 3, padding=1), nn.GroupNorm(8, base_dim*2), nn.SiLU())
        self.ca = CrossAttention(dim=base_dim*2, context_dim=plan_dim)
        
        # Shallow Layers (Decoder)
        self.dec1 = nn.Sequential(nn.Conv2d(base_dim*2, base_dim, 3, padding=1), nn.GroupNorm(8, base_dim), nn.SiLU())
        self.za2 = ZeroAdapter(base_dim, evidence_dim)
        
        self.final = nn.Conv2d(base_dim, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, u_X: torch.Tensor, c_S: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        # Time and Observation embeddings
        t_emb = self.time_embed(t.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)
        obs_emb = self.obs_proj(u_X).unsqueeze(-1).unsqueeze(-1)
        
        # --- Shallow Layer (Encoder) ---
        h = self.enc1(x)
        h = h + t_emb + obs_emb
        # Local Statistics Injection (Zero-Adapter with E)
        h = self.za1(h, E)
        
        # --- Deep Layer (Bottleneck) ---
        h = self.mid(h)
        # Reshape for Attention: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = h.shape
        h_flat = h.permute(0, 2, 3, 1).reshape(B, -1, C)
        # Global Consistency Injection (Cross-Attention with c_S)
        h_flat = self.ca(h_flat, c_S)
        h = h_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # --- Shallow Layer (Decoder) ---
        h = self.dec1(h)
        # Local Statistics Injection (Zero-Adapter with E)
        h = self.za2(h, E)
        
        return self.final(h)

class DriftGuidedDiffusion(nn.Module):
    def __init__(self, 
                 vae_model, 
                 text_encoder,
                 feature_extractor,
                 latent_dim: int = 4,
                 plan_dim: int = 512,
                 evidence_dim: int = 512,
                 obs_dim: int = 512,
                 num_timesteps: int = 1000):
        super().__init__()
        self.vae = vae_model
        self.text_encoder = text_encoder # g(.)
        self.feature_extractor = feature_extractor # phi(.)
        
        self.num_timesteps = num_timesteps
        self.denoise_net = DenoisingUNet(latent_dim, plan_dim, evidence_dim, obs_dim)
        
        # Evidence Alignment Head A(.)
        self.evidence_align = nn.Linear(evidence_dim, 4096) # Output dim matches phi(Y)
        
        # Load loss weights from config if provided
        if config and 'drift_guidance' in config:
            dg_config = config['drift_guidance']
            self.loss_plan_weight = dg_config.get('loss_plan_weight', 0.1)
            self.loss_evi_weight = dg_config.get('loss_evi_weight', 0.1)
            self.guidance_scale = dg_config.get('guidance_scale', 1.0)
        else:
            self.loss_plan_weight = 0.1
            self.loss_evi_weight = 0.1
            self.guidance_scale = 1.0

        # DDPM Schedule
        beta = torch.linspace(1e-4, 0.02, num_timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, z_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(z_0)
        
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1)
        
        return sqrt_alpha_bar_t * z_0 + sqrt_one_minus_alpha_bar_t * noise

    def p_sample(self, z_t: torch.Tensor, t: torch.Tensor, u_X: torch.Tensor, c_S: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        # Predict noise: epsilon_theta
        t_tensor = torch.full((z_t.shape[0],), t, device=z_t.device, dtype=torch.float32)
        eps_theta = self.denoise_net(z_t, t_tensor, u_X, c_S, E)
        
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        sigma_t = torch.sqrt(1 - alpha_t) # Simplified sigma
        
        # Denoising step equation
        coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        mean = (1 / torch.sqrt(alpha_t)) * (z_t - coef * eps_theta)
        
        if t > 0:
            noise = torch.randn_like(z_t)
            return mean + sigma_t * noise
        else:
            return mean

    @torch.no_grad()
    def generate(self, u_X: torch.Tensor, c_S: torch.Tensor, E: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        device = u_X.device
        B = u_X.shape[0]
        z = torch.randn((B, *shape), device=device)
        
        for t in reversed(range(self.num_timesteps)):
            z = self.p_sample(z, t, u_X, c_S, E)
            
        return self.vae.decode(z)

    def compute_loss(self, Y_target: torch.Tensor, u_X: torch.Tensor, c_S: torch.Tensor, E: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Latent Compression
        with torch.no_grad():
            z_0 = self.vae.encode(Y_target) # z0 = VAE(Y)
        
        # 2. Diffusion Forward
        t = torch.randint(0, self.num_timesteps, (z_0.shape[0],), device=z_0.device)
        noise = torch.randn_like(z_0)
        z_t = self.q_sample(z_0, t, noise)
        
        # 3. Noise Prediction for Diffusion
        eps_pred = self.denoise_net(z_t, t.float(), u_X, c_S, E)
        loss_diff = F.mse_loss(eps_pred, noise)
        
        # 4. Instruction-Following Regularization
        # We need to approximate Y_hat from z_0_pred for regularization
        # z_0_pred = (z_t - sqrt(1-alpha_bar)*eps) / sqrt(alpha_bar)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1)
        z_0_pred = (z_t - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar
        Y_hat = self.vae.decode(z_0_pred)
        
        # L_plan = 1 - cos(g(Y_hat), c_S)
        y_feat = self.text_encoder(Y_hat) # Assuming g(.) maps image to text space
        l_plan = 1 - F.cosine_similarity(y_feat, c_S, dim=-1).mean()
        
        # L_evi = ||phi(Y_hat) - A(E)||_1
        phi_y = self.feature_extractor(Y_hat) # Flattened shallow features
        aligned_E = self.evidence_align(E)
        l_evi = F.l1_loss(phi_y, aligned_E)
        
        # Loss weights from implementation details
        loss_plan_weight = getattr(self, 'loss_plan_weight', 0.1)
        loss_evi_weight = getattr(self, 'loss_evi_weight', 0.1)
        total_loss = loss_diff + loss_plan_weight * l_plan + loss_evi_weight * l_evi
        
        return {
            "loss": total_loss,
            "loss_diff": loss_diff,
            "loss_plan": l_plan,
            "loss_evi": l_evi
        }