from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)


class AdaLayerNorm(nn.Module):
    """
    Adaptive LayerNorm (AdaLN) as used in DiT-style architectures.
    Given a global conditioning vector c (time + obs), it produces
    a per-channel scale and shift that modulate the normalized features.
    """
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Maps cond embedding to (shift, scale)
        self.modulation = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, T, D)
        cond: (B, C)  -> broadcast over T
        """
        x_norm = self.norm(x)
        shift, scale = self.modulation(cond).chunk(2, dim=-1)   # (B, D), (B, D)
        # Broadcast to (B, T, D)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        return x_norm * (1 + scale) + shift


class DiTBlock(nn.Module):
    """
    A single DiT block: AdaLN -> self-attention -> AdaLN -> MLP.
    """
    def __init__(self, dim: int, n_head: int, cond_dim: int, p_drop_attn: float = 0.0):
        super().__init__()
        self.ada_ln1 = AdaLayerNorm(dim, cond_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_head,
            batch_first=True,
            dropout=p_drop_attn
        )
        self.ada_ln2 = AdaLayerNorm(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:    (B, T, D)
        cond: (B, C)
        attn_mask: (T, T) or None, additive mask with 0 for allowed, -inf for blocked
        """
        # Self-attention with AdaLN
        h = self.ada_ln1(x, cond)
        h_attn, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + h_attn

        # MLP with AdaLN
        h = self.ada_ln2(x, cond)
        h = self.mlp(h)
        x = x + h
        return x


class TransformerForDiffusion(ModuleAttrMixin):
    """
    DiT-style Transformer for Diffusion over trajectories.

    Interface-compatible with the original TransformerForDiffusion:
    - Same __init__ signature
    - Same forward(sample, timestep, cond=None, **kwargs) API
    - Same get_optim_groups / configure_optimizers
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,  # kept for API compatibility, see below
        n_cond_layers: int = 0,     # unused in this DiT variant
    ) -> None:
        super().__init__()

        if n_obs_steps is None:
            n_obs_steps = horizon

        # In the original implementation, obs_as_cond was overridden by cond_dim > 0.
        # We keep the same behavior for compatibility.
        obs_as_cond = cond_dim > 0

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_emb = n_emb
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = False  # kept for API compatibility

        # Number of tokens in the main trajectory and conditioning sequence.
        self.T = horizon
        self.T_cond = n_obs_steps if obs_as_cond else 0

        # Embedding layers
        self.input_emb = nn.Linear(input_dim, n_emb)
        # Positional embeddings for [cond tokens (if any) + trajectory tokens]
        max_tokens = self.T + self.T_cond
        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens, n_emb))

        self.drop = nn.Dropout(p_drop_emb)

        # Time embedding (Sinusoidal + used directly as cond)
        self.time_emb = SinusoidalPosEmb(n_emb)

        # Optional observation-conditioning projection
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb) if obs_as_cond and cond_dim > 0 else None
        # We don't use a separate cond_pos_emb in this DiT variant, but keep the
        # attribute for compatibility with the old _init_weights logic.
        self.cond_pos_emb = None

        # DiT blocks (use n_emb as cond_dim; time + pooled obs are embedded into this space)
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=n_emb,
                n_head=n_head,
                cond_dim=n_emb,
                p_drop_attn=p_drop_attn,
            )
            for _ in range(n_layer)
        ])

        # Final layer norm + head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # Causal attention masks
        if causal_attn:
            # Mask without observation tokens (just trajectory tokens)
            self.mask = self._build_causal_mask(self.T)
            # Mask with observation tokens concatenated before trajectory tokens
            if self.T_cond > 0:
                self.mask_with_cond = self._build_causal_mask_with_cond(
                    T=self.T, T_cond=self.T_cond
                )
            else:
                self.mask_with_cond = None
        else:
            self.mask = None
            self.mask_with_cond = None

        # Init
        self.apply(self._init_weights)
        logger.info(
            "DiT TransformerForDiffusion - number of parameters: %e",
            sum(p.numel() for p in self.parameters())
        )

    # -------------------------------------------------------------------------
    # Mask construction
    # -------------------------------------------------------------------------
    @staticmethod
    def _build_causal_mask(T: int) -> torch.Tensor:
        """
        Standard causal mask over T tokens: token i cannot attend to j > i.
        Returns (T, T) float mask with 0 for allowed, -inf for blocked.
        """
        mask = torch.triu(torch.ones(T, T), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf")).masked_fill(mask == 0, 0.0)
        return mask

    @staticmethod
    def _build_causal_mask_with_cond(T: int, T_cond: int) -> torch.Tensor:
        """
        Causal mask when we prepend T_cond observation tokens:

        Tokens layout: [cond_0, ..., cond_{T_cond-1}, x_0, ..., x_{T-1}]

        - cond tokens can attend to everything (including future trajectory tokens).
        - trajectory token x_t can:
            - attend to all cond tokens
            - attend to x_0..x_t (no future x_{t'>t}).
        """
        total = T + T_cond
        mask = torch.zeros(total, total)

        # Indices: 0..T_cond-1 are cond; T_cond..T_cond+T-1 are trajectory
        for i in range(total):
            for j in range(total):
                if i >= T_cond and j >= T_cond and j > i:
                    # trajectory token i cannot see future trajectory j
                    mask[i, j] = float("-inf")
                else:
                    mask[i, j] = 0.0
        return mask

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def _init_weights(self, module: nn.Module):
        # Generic, DiT-friendly init (adapted from the original, but more permissive)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            # Initialize all attention weights similarly
            for name, param in module.named_parameters():
                if "weight" in name:
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        # other modules (AdaLayerNorm, DiTBlock, etc.) are composed of the above

    # -------------------------------------------------------------------------
    # Optimizer configuration (unchanged API)
    # -------------------------------------------------------------------------
    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        Same semantics as the original: separate parameters into decay / no-decay
        groups for AdamW.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias") or pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # pos_emb should not be decayed
        no_decay.add("pos_emb")
        # kept for compatibility with original code; ModuleAttrMixin typically
        # defines this parameter.
        no_decay.add("_dummy_variable")

        # Validate partition
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!" %
            (str(param_dict.keys() - union_params),)
        )

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    # -------------------------------------------------------------------------
    # Forward (API-compatible)
    # -------------------------------------------------------------------------
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        sample:   (B, T, input_dim)
        timestep: (B,) or scalar
        cond:     (B, T_obs, cond_dim) or None

        Returns:
        (B, T, output_dim)
        """
        B, T, _ = sample.shape
        assert T == self.T, f"Expected horizon {self.T}, got {T}"

        # 1. Time embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(B)  # (B,)
        time_emb = self.time_emb(timesteps)  # (B, n_emb)

        # 2. Input embedding
        x_tokens = self.input_emb(sample)  # (B, T, n_emb)

        # 3. Optional observation conditioning
        use_obs = self.obs_as_cond and (cond is not None) and (self.cond_obs_emb is not None)
        if use_obs:
            # cond: (B, T_obs, cond_dim)
            cond_tokens = self.cond_obs_emb(cond)  # (B, T_obs, n_emb)
            Bc, T_obs, _ = cond_tokens.shape
            assert Bc == B
            assert T_obs == self.T_cond, f"Expected {self.T_cond} obs tokens, got {T_obs}"

            # Concatenate [obs tokens, trajectory tokens]
            tokens = torch.cat([cond_tokens, x_tokens], dim=1)  # (B, T_obs + T, n_emb)

            # Positional embeddings for [cond + traj]
            pos = self.pos_emb[:, : (self.T_cond + self.T), :]  # (1, total, n_emb)
            tokens = tokens + pos

            # Global conditioning vector: time + mean pooled obs
            cond_pooled = cond_tokens.mean(dim=1)  # (B, n_emb)
            cond_vec = time_emb + cond_pooled      # (B, n_emb)

            # Choose appropriate mask
            attn_mask = self.mask_with_cond.to(tokens.device) if self.mask_with_cond is not None else None
        else:
            # No obs conditioning: only trajectory tokens
            tokens = x_tokens  # (B, T, n_emb)
            pos = self.pos_emb[:, : self.T, :]     # (1, T, n_emb)
            tokens = tokens + pos

            cond_vec = time_emb                    # (B, n_emb)
            attn_mask = self.mask.to(tokens.device) if self.mask is not None else None

        tokens = self.drop(tokens)

        # 4. DiT blocks
        for block in self.blocks:
            tokens = block(tokens, cond_vec, attn_mask=attn_mask)

        # 5. Drop obs tokens (if any), keep only trajectory tokens for output
        if use_obs:
            tokens = tokens[:, self.T_cond :, :]  # (B, T, n_emb)

        # 6. Final norm + head
        x = self.ln_f(tokens)
        x = self.head(x)  # (B, T, output_dim)
        return x


# Optional quick sanity test (kept close to original)
def test():
    # GPT with time embedding
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        causal_attn=True,
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4, 8, 16))
    out = transformer(sample, timestep)
    print("out (no cond):", out.shape)

    # GPT with time embedding and obs cond
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4, 8, 16))
    cond = torch.zeros((4, 4, 10))
    out = transformer(sample, timestep, cond)
    print("out (with cond):", out.shape)

    # BERT-like case is now just handled by the same DiT (time_as_cond flag is ignored in behavior,
    # but kept for API compatibility).
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        time_as_cond=False,
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4, 8, 16))
    out = transformer(sample, timestep)
    print("out (time_as_cond=False):", out.shape)


if __name__ == "__main__":
    test()
