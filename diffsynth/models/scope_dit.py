"""
SCOPE DiT — Diffusion Transformer with Action Module.

This module extends the Wan2.2-TI2V-5B DiT architecture by injecting an ActionModule
into each transformer block. The ActionModule enables action-conditioned video generation
by processing keyboard/button signals (via cross-attention) and mouse/joystick signals
(via temporal self-attention with MLP fusion) at every layer.

Key classes:
    - WanModel: The full SCOPE DiT (30-layer transformer + ActionModule per block)
    - ActionModule: Per-block action conditioning (mouse path + keyboard path)
    - DiTBlock: Transformer block with self-attn, cross-attn, ActionModule, and FFN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from typing import Tuple, Optional
from einops import rearrange, repeat
from .wan_video_camera_controller import SimpleAdapter
from ..core.gradient import gradient_checkpoint_forward

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)

class ActionModule(nn.Module):
    """Fuses video hidden states with mouse and keyboard action sequences."""

    def __init__(
        self,
        mouse_dim_in: int,
        keyboard_dim_in: int,
        dim: int,
        num_heads: int,
        vae_time_compression_ratio: int = 4,
        windows_size: int = 1,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.vae_time_compression_ratio = vae_time_compression_ratio
        self.windows_size = windows_size
        self.mouse_feat_dim = (
            mouse_dim_in * vae_time_compression_ratio * windows_size
        )
        self.num_heads = num_heads

        self.mouse_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.mouse_feat_dim + dim, dim, bias=True),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(dim, dim),
            torch.nn.LayerNorm(dim),
        )
        self.mouse_attn = SelfAttention(dim, num_heads)

        # 1. Embedding: Raw keyboard signals -> Hidden Size.
        # Reference implementation: nn.Linear -> SiLU -> nn.Linear.
        # Map to a smaller space first to match sliding window expectations.
        intermediate_dim = (
            dim // vae_time_compression_ratio // windows_size
        )
        self.keyboard_embed = nn.Sequential(
            nn.Linear(keyboard_dim_in, intermediate_dim, bias=True),
            nn.SiLU(),
            nn.Linear(intermediate_dim, intermediate_dim, bias=True),
        )

        # Query Projection (Video Hidden State -> Keyboard Query Space)
        self.keyboard_q_proj = nn.Linear(dim, dim, bias=False)
        # Key/Value Projection (Keyboard Features -> Keyboard KV Space)
        self.keyboard_kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.keyboard_o_proj = nn.Linear(dim, dim, bias=False)
        self.key_attn_q_norm = RMSNorm(dim, eps=eps)
        self.key_attn_k_norm = RMSNorm(dim, eps=eps)

        self.init_weights()

    def init_weights(self):
        """Initializes weights using standard Transformer and ResNet zero-init strategies."""
        self.apply(self._init_weights)
        if hasattr(self.mouse_attn, "o"):
            nn.init.zeros_(self.mouse_attn.o.weight)
            nn.init.zeros_(self.mouse_attn.o.bias)

        # Keyboard output zero initialization (ResNet style)
        if hasattr(self, "keyboard_o_proj"):
            nn.init.zeros_(self.keyboard_o_proj.weight)
            if self.keyboard_o_proj.bias is not None:
                nn.init.zeros_(self.keyboard_o_proj.bias)

    def _init_weights(self, m: nn.Module):
        """Applies Xavier uniform to linear layers and sets gains/biases."""
        if isinstance(m, nn.Linear):
            # Xavier Uniform initialization is well-suited for Tanh/GELU/SiLU activations.
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init.ones_(m.weight)
            init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        f: int,
        h: int,
        w: int,
        freqs: Tuple[torch.Tensor, torch.Tensor],
        mouse_action: Optional[torch.Tensor] = None,
        keyboard_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuses temporal actions into spatial-temporal video tokens.

        Args:
            x: Video tokens of shape [B, L, C], where L = f * h * w.
            f: Number of latent video frames (temporal dimension).
            h: Latent height.
            w: Latent width.
            freqs: Cosine and sine components for RoPE [cos, sin].
            mouse_action: Mouse sequence of shape [B, N_frames_raw, mouse_dim].
            keyboard_action: Keyboard sequence of shape [B, N_frames_raw,
              key_dim].

        Returns:
            Processed video tokens of shape [B, L, C].
        """
        B, L, C = x.shape
        S = h * w  # Total number of spatial pixels

        assert (
            L == f * S
        ), f"Input shape {L} does not match f*h*w ({f}*{h}*{w})"

        # =========================================================
        # 1. Video Feature Reshaping
        # =========================================================
        # Goal: Merge spatial dimension S into Batch, treating each pixel as an
        # independent time series.
        # Shape change: [B, f*h*w, C] -> [B*h*w, f, C]
        hidden_states = rearrange(x, "b (f s) c -> (b s) f c", f=f, s=S)

        # =========================================================
        # 2. Mouse Action Processing
        # =========================================================
        if mouse_action is not None:
            B_mouse, N_raw, C_mouse = mouse_action.shape

            # --- 2.1 Padding (For Time Windows) ---
            pad_t = self.vae_time_compression_ratio * self.windows_size

            # Replicate padding: Copy the first frame to pad the beginning
            pad = mouse_action[:, 0:1, :].expand(-1, pad_t, -1)
            mouse_padded = torch.cat(
                [pad, mouse_action], dim=1
            )  # [B, N_raw + pad_t, C_mouse]

            # --- 2.2 Sliding Window Extraction ---
            # Goal: For each time-step f in hidden_states, extract its
            # corresponding raw mouse signal window.
            group_mouse = []
            for i in range(f):
                # Calculate the slice range.
                # Logic: The i-th latent frame maps to raw signals from i*R to i*R + pad_t.
                start_idx = i * self.vae_time_compression_ratio
                end_idx = start_idx + pad_t

                # Extract window: [B, pad_t, C_mouse]
                window = mouse_padded[:, start_idx:end_idx, :]
                group_mouse.append(window)

            # Stack along temporal dimension: [B, f, pad_t, C_mouse]
            group_mouse = torch.stack(group_mouse, dim=1)

            # --- 2.3 Spatial Broadcasting ---
            # Mouse signals are globally shared across the entire frame.
            # Broadcast across all S pixels: [B, f, pad_t, C_mouse] -> [B, f, pad_t, C_mouse, S]
            group_mouse = group_mouse.unsqueeze(-1).expand(-1, -1, -1, -1, S)

            # --- 2.4 Rearrange & Flatten ---
            # 1. Move S into the Batch dimension: (b ... s) -> (b s ...)
            # 2. Flatten pad_t and C_mouse dimensions: (p c) -> flat_dim
            # Shape change: [B, f, p, c, S] -> [B*S, f, p*c]
            group_mouse = rearrange(group_mouse, "b f p c s -> (b s) f (p c)")

            # --- 2.5 Feature Fusion ---
            # hidden_states: [B*S, f, C]
            # group_mouse:   [B*S, f, mouse_feat_flat_dim]
            fusion_input = torch.cat([hidden_states, group_mouse], dim=-1)

            # MLP Mapping: [B*S, f, C]
            attn_input = self.mouse_mlp(fusion_input)

            # --- 2.6 Attention (Pixel-wise Temporal Self-Attention) ---
            attn_out = self.mouse_attn(attn_input, freqs)

            # --- 2.7 Residual Connection ---
            hidden_states = hidden_states + attn_out

        # =========================================================
        # 3. Keyboard Action Processing
        # =========================================================
        if keyboard_action is not None:
            # keyboard_action: [B, N_frames_raw, C_key_in]

            # --- 3.1 Embedding ---
            k_emb = self.keyboard_embed(keyboard_action)  # [B, N_raw, dim]

            # --- 3.2 Padding & Sliding Window (Similar to Mouse) ---
            pad_t = self.vae_time_compression_ratio * self.windows_size
            pad_k = k_emb[:, 0:1, :].expand(-1, pad_t, -1)
            k_padded = torch.cat([pad_k, k_emb], dim=1)

            group_keyboard = []
            for i in range(f):
                start_idx = i * self.vae_time_compression_ratio
                end_idx = start_idx + pad_t
                window = k_padded[:, start_idx:end_idx, :]
                group_keyboard.append(window)

            # [B, f, pad_t, dim]
            group_keyboard = torch.stack(group_keyboard, dim=1)

            # Flatten window dim to map to the KV hidden space: [B, f, pad_t * dim]
            group_keyboard = rearrange(group_keyboard, "b f p d -> b f (p d)")

            # --- 3.3 Cross Attention Preparation ---
            # Queries originate from Video (hidden_states)
            # hidden_states: [(B*S), f, C]
            q_video = self.keyboard_q_proj(
                hidden_states
            )  # [(B*S), f, keyboard_hidden_dim]
            q_video = self.key_attn_q_norm(q_video)

            # Keys/Values originate from Keyboard features
            # group_keyboard: [B, f, feat_dim]
            kv_keyboard = self.keyboard_kv_proj(
                group_keyboard
            )  # [B, f, 2 * keyboard_hidden_dim]
            k_keyboard, v_key = kv_keyboard.chunk(2, dim=-1)
            k_key = self.key_attn_k_norm(k_keyboard)

            # --- 3.4 Expand Keyboard KV to match Video Batch (B -> B*S) ---
            # Keyboard inputs are global; repeat them to match the pixel-fused sequence.
            # Shape change: [B, f, d] -> [(B S), f, d]
            k_key = repeat(k_key, "b f d -> (b s) f d", s=S)
            v_key = repeat(v_key, "b f d -> (b s) f d", s=S)

            # --- 3.5 Rotary Position Embedding (RoPE) ---
            # Note: The provided temporal freqs accurately match dimension f.
            # rope_apply expects (b, s, n, d). In our space, 'f' behaves as the sequence 
            # length dimension 's', which is completely aligned.
            q_video = rope_apply(q_video, freqs, self.num_heads)
            k_key = rope_apply(k_key, freqs, self.num_heads)

            # --- 3.6 Attention (Time-aligned Cross Attention) ---
            # Q, K, V shapes: [(B*S), f, h, d]
            # At each temporal index f, every pixel query pays attention to its 
            # corresponding temporal window of keyboard actions.
            attn_val = flash_attention(
                q_video, k_key, v_key, num_heads=self.num_heads
            )

            # --- 3.7 Projection & Residual ---
            attn_out = self.keyboard_o_proj(attn_val)  # [(B*S), f, dim]
            hidden_states = hidden_states + attn_out

        # Restore sequence structure back to global spatial-temporal token space
        x = rearrange(hidden_states, "(b s) f c -> b (f s) c", b=B)
        return x
    
class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6, enable_action: bool = False, action_config: dict = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.enable_action = enable_action
        if enable_action:
            self.action_attn = ActionModule(**action_config)

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, f, h, w, mouse_action=None, keyboard_action=None, freqs_mouse=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        if self.enable_action:
            x = self.action_attn(x, f, h, w, freqs=freqs_mouse, mouse_action=mouse_action, keyboard_action=keyboard_action)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        # ---- Action ----
        enable_action: bool = False,
        action_config: dict = None,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        if enable_action:
            if action_config is None:
                action_config = {
                    'mouse_dim_in': 2,
                    'keyboard_dim_in': 8,
                    'dim': self.dim,
                    'num_heads': num_heads,
                    'windows_size': 4,
                }
            else:
                action_config = action_config.copy()


        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps,
                     enable_action=enable_action, action_config=action_config)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)
        self.freqs_mouse = precompute_freqs_cis(dim=head_dim, end=100)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x, context, t_mod, freqs
                )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
