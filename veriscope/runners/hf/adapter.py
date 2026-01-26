from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class HFMetricConfig:
    max_tokens_per_batch: int = 2048
    rp_dim: int = 64


def _cov_eigs_sorted_from_activations(H: torch.Tensor) -> Tuple[Optional[torch.Tensor], float]:
    try:
        if H.ndim != 2:
            return None, float("nan")
        n, d = H.shape
        if n < 2 or d < 1:
            return None, float("nan")

        H = H - H.mean(dim=0, keepdim=True)
        cov = (H.T @ H) / float(max(1, n - 1))
        eigs = torch.linalg.eigvalsh(cov)
        eigs = torch.clamp(eigs, min=0.0)
        eigs_sorted, _ = torch.sort(eigs, descending=True)
        total = float(eigs_sorted.sum().item())
        if not np.isfinite(total) or total <= 0.0:
            return None, float("nan")
        return eigs_sorted, total
    except Exception:
        return None, float("nan")


def _geom_metrics_from_eigs(
    eig_sorted: Optional[torch.Tensor],
    total: float,
    k: Optional[int] = None,
) -> Dict[str, Any]:
    if eig_sorted is None or eig_sorted.numel() == 0:
        return {
            "var_out_k": float("nan"),
            "eff_dim": float("nan"),
            "k_used": 0,
            "var_out_k_valid": 0,
        }

    tot = float(total)
    if not np.isfinite(tot) or tot <= 0.0:
        return {
            "var_out_k": float("nan"),
            "eff_dim": float("nan"),
            "k_used": 0,
            "var_out_k_valid": 0,
        }

    denom = float((eig_sorted * eig_sorted).sum().item())
    denom = denom if denom > 1e-20 else 1e-20
    eff_dim = (tot * tot) / denom

    d_eigs = int(eig_sorted.numel())
    if k is None:
        k_used = int(max(1, min(d_eigs, round(math.sqrt(d_eigs)))))
    else:
        k_used = int(max(1, min(d_eigs, int(k))))

    top = float(eig_sorted[:k_used].sum().item())
    tail = max(0.0, tot - top)
    var_out_k = float(tail / tot)

    return {
        "var_out_k": float(var_out_k),
        "eff_dim": float(eff_dim),
        "k_used": int(k_used),
        "var_out_k_valid": 1,
    }


class HFMetricComputer:
    def __init__(
        self,
        config: HFMetricConfig,
        seed: int,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self._jl_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def _get_jl_matrix(self, d_in: int, d_out: int) -> torch.Tensor:
        key = (d_in, d_out, self.seed)
        if key not in self._jl_cache:
            g = torch.Generator().manual_seed(self.seed + d_in + 31 * d_out)
            A = torch.randn(d_in, d_out, generator=g, device=torch.device("cpu"), dtype=torch.float32) / math.sqrt(
                d_out
            )
            self._jl_cache[key] = A
        return self._jl_cache[key]

    def compute_metrics(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        step: int,
    ) -> Dict[str, Any]:
        # hidden_states: (B, T, D)
        H = hidden_states
        if attention_mask is not None:
            mask = attention_mask.reshape(-1).bool()
            H = H.reshape(-1, H.shape[-1])[mask]
        else:
            H = H.reshape(-1, H.shape[-1])

        if H.numel() == 0:
            return {
                "var_out_k": float("nan"),
                "eff_dim": float("nan"),
                "k_used": 0,
                "var_out_k_valid": 0,
            }

        max_tokens = int(self.config.max_tokens_per_batch)
        if max_tokens > 0 and H.shape[0] > max_tokens:
            g = torch.Generator().manual_seed(self.seed + 101 * int(step))
            idx = torch.randperm(H.shape[0], generator=g)[:max_tokens]
            H = H[idx]

        H = H.float().cpu()
        d_in = H.shape[1]
        d_out = min(int(self.config.rp_dim), d_in)
        A = self._get_jl_matrix(d_in, d_out)
        H_jl = H @ A

        eig_sorted, total = _cov_eigs_sorted_from_activations(H_jl)
        geom = _geom_metrics_from_eigs(eig_sorted, total, k=None)
        return geom
