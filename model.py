from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Energy-Inspired Landscape Modeling framework.

Author: JunYoung Park and Myung-Kyu Yi
"""


class RelativeEnergyPhysics(nn.Module):
    """Relative physical energy proxy from IMU channels."""
    def __init__(
        self, 
        acc_indices: Sequence[int],
        gyro_indices: Sequence[int],
    ):
        super().__init__()
        self.acc_indices = list(acc_indices)
        self.gyro_indices = list(gyro_indices)
        self.m = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.I = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:  
        m_pos = F.softplus(self.m)
        I_pos = F.softplus(self.I)

        x_t = x.transpose(1, 2) 
        B, T, _ = x_t.shape

        if len(self.acc_indices) > 0:
            acc_data = x_t[:, :, self.acc_indices]
            n_acc = len(self.acc_indices) // 3
            if n_acc > 0:
                acc_reshaped = acc_data.view(B, T, n_acc, 3)
                acc_mag = (acc_reshaped ** 2).sum(dim=-1).mean(dim=-1, keepdim=True)
            else:
                acc_mag = torch.zeros(B, T, 1, device=x.device)
        else:
            acc_mag = torch.zeros(B, T, 1, device=x.device)

        E_kin = 0.5 * m_pos * acc_mag

        if len(self.gyro_indices) > 0:
            gyro_data = x_t[:, :, self.gyro_indices]
            n_gyro = len(self.gyro_indices) // 3
            if n_gyro > 0:
                gyro_reshaped = gyro_data.view(B, T, n_gyro, 3)
                gyro_mag = (gyro_reshaped ** 2).sum(dim=-1).mean(dim=-1, keepdim=True)
            else:
                gyro_mag = torch.zeros(B, T, 1, device=x.device)
        else:
            gyro_mag = torch.zeros(B, T, 1, device=x.device)

        E_rot = 0.5 * I_pos * gyro_mag
        return E_kin + E_rot  


class PotentialEnergyField(nn.Module):
    """Learned potential energy field E(t) conditioned on (x_t, E_rel(t))."""
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int,
    ):
        super().__init__()
        self.energy_net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        E_phys: torch.Tensor,
    ) -> torch.Tensor:  
        x_t = x.transpose(1, 2)           
        x_aug = torch.cat([x_t, E_phys], -1)  
        return self.energy_net(x_aug)         


class EnergyGradientFlow(nn.Module):
    """Learned gradient-like flow g(t) conditioned on (x_t, E_rel(t))."""
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int,
    ):
        super().__init__()
        self.gradient_net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        E_phys: torch.Tensor
    ) -> torch.Tensor:  
        x_t = x.transpose(1, 2)             
        x_aug = torch.cat([x_t, E_phys], -1)   
        return self.gradient_net(x_aug)      


class RateAttention(nn.Module):
    """Computes rate-based attention from first-order energy variation."""
    def __init__(
        self, 
        hidden_dim: int,
    ):
        super().__init__()
        self.rate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rate_gate = nn.Linear(hidden_dim, 1)

    def forward(
        self, 
        features: torch.Tensor, 
        energy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:  
        energy_rate = torch.cat([energy[:, :1, :], energy[:, 1:, :] - energy[:, :-1, :]], dim=1)
        ctx = self.rate_proj(features)
        scores = self.rate_gate(ctx + energy_rate)
        attn = torch.softmax(scores.squeeze(-1), dim=1)
        return attn, features * attn.unsqueeze(-1)


class PhaseAttention(nn.Module):
    """Computes phase-based attention from second-order energy curvature."""
    def __init__(
        self,
        hidden_dim: int,
    ):
        super().__init__()
        self.phase_proj = nn.Linear(hidden_dim, hidden_dim)
        self.phase_gate = nn.Linear(hidden_dim, 1)

    def forward(
        self, 
        features: torch.Tensor, 
        energy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        energy_pad = F.pad(energy.transpose(1, 2), (1, 1), mode='replicate').transpose(1, 2)
        curv = energy_pad[:, 2:, :] - 2 * energy_pad[:, 1:-1, :] + energy_pad[:, :-2, :]
        ctx = self.phase_proj(features)
        scores = self.phase_gate(ctx + curv)
        attn = torch.softmax(scores.squeeze(-1), dim=1)
        return attn, features * attn.unsqueeze(-1)


class AttentionToAttention(nn.Module):
    """Attention-to-Attention (A²) reconciliation with magnitude-aware gating."""
    def __init__(
        self, 
        hidden_dim: int,
    ):
        super().__init__()
        d_rho = hidden_dim // 4

        self.W1 = nn.Linear(3, d_rho)
        self.W2 = nn.Linear(d_rho, 2)

    def forward(
        self,
        rate_attn: torch.Tensor,
        phase_attn: torch.Tensor, 
        rate_feat: torch.Tensor,
        phase_feat: torch.Tensor,
        h_t: torch.Tensor,
    ) -> torch.Tensor:
        h_mag = torch.norm(h_t, p=2, dim=-1) 

        v_t = torch.stack([rate_attn, phase_attn, h_mag], dim=-1)

        hidden = F.gelu(self.W1(v_t))
        rho = torch.softmax(self.W2(hidden), dim=-1)  

        rate_reliability = rho[:, :, 0:1]  
        phase_reliability = rho[:, :, 1:2]  

        reconciled = rate_reliability * rate_feat + phase_reliability * phase_feat
        return reconciled


class LandscapeGeometryEncoder(nn.Module):
    """Landscape geometry encoder with dual attention and A² reconciliation."""
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int,
    ):
        super().__init__()
        combined_dim = input_dim * 2 + 1
        self.input_proj = nn.Conv1d(combined_dim, hidden_dim, kernel_size=1)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.15)
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        self.rate_attention = RateAttention(hidden_dim)
        self.phase_attention = PhaseAttention(hidden_dim)
        self.a2 = AttentionToAttention(hidden_dim)

    def forward(
        self, 
        energy: torch.Tensor, 
        gradient: torch.Tensor,
        x_original: torch.Tensor,
    ) -> torch.Tensor:
        x_t = x_original.transpose(1, 2)                     
        state = torch.cat([x_t, gradient, energy], dim=-1)  

        h = self.input_proj(state.transpose(1, 2))       
        h = self.conv(h)
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(h)
        h = self.dropout(h)

        h_t = h.transpose(1, 2)                           
        h_t, _ = self.temporal_attn(h_t, h_t, h_t)

        rate_attn, rate_feat = self.rate_attention(h_t, energy)
        phase_attn, phase_feat = self.phase_attention(h_t, energy)

        reconciled = self.a2(rate_attn, phase_attn, rate_feat, phase_feat, h_t)
        return reconciled


class MIELHAR_A2(nn.Module):
    """MIEL-HAR with energy-inspired landscape modeling and A² reconciliation."""
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_classes: int,
        sensor_config: Dict[str, Sequence[int]],
        physics_grad_weight: float = 0.25, 
        energy_reg_weight: float = 0.01,
    ):
        super().__init__()
        self.acc_indices = sensor_config["acc_indices"]
        self.gyro_indices = sensor_config["gyro_indices"]

        self.physics = RelativeEnergyPhysics(self.acc_indices, self.gyro_indices)
        self.potential_field = PotentialEnergyField(input_dim, hidden_dim)
        self.gradient_flow   = EnergyGradientFlow(input_dim, hidden_dim)
        self.physics_grad_weight = physics_grad_weight

        self.encoder = LandscapeGeometryEncoder(input_dim, hidden_dim)
        self.energy_reg_weight = energy_reg_weight

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def _physics_grad(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        x_t = x.transpose(1, 2) 
        m_pos = F.softplus(self.physics.m)
        I_pos = F.softplus(self.physics.I)

        pg = torch.zeros_like(x_t)
        if len(self.acc_indices) > 0:
            pg[:, :, self.acc_indices] = x_t[:, :, self.acc_indices] * m_pos
        if len(self.gyro_indices) > 0:
            pg[:, :, self.gyro_indices] = x_t[:, :, self.gyro_indices] * I_pos
        return pg

    def forward(
        self, 
        x: torch.Tensor, 
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
  
        E_phys = self.physics(x)
        energy = self.potential_field(x, E_phys)
        grad = self.gradient_flow(x, E_phys)
        grad = grad + self.physics_grad_weight * self._physics_grad(x)
        
        landscape_features = self.encoder(energy, grad, x)
        
        g = self.pool(landscape_features.transpose(1, 2)).squeeze(-1)
        logits = self.classifier(g)
        
        if not return_aux:
            return logits, None
        
        aux: Dict[str, torch.Tensor] = {
            "h": g,
            "landscape": landscape_features,
            "energy": energy,
            "grad": grad,
            "E_phys": E_phys,
        }
        return logits, aux
