# veriscope/runners/legacy/data.py
"""
Data and loader utilities for legacy CIFAR/STL runners.

Pulled out of runners/legacy_cli_refactor.py with the same behavior:
- CIFAR-10 train/val/test dataset construction with optional file lock
- STL10 external monitor dataset
- Balanced per-class split construction for monitor/norm_ref
- DataLoader helpers with deterministic seeding and optional persistence
- load_splits(seed, cfg, outdir, data_root) → (tr_aug, tr_take, mon_val, norm_ref, splits_path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List

import inspect
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset, Dataset
import torchvision  # type: ignore[import-untyped]
from torchvision import transforms

import hashlib
import torch.nn.functional as F

from veriscope.runners.legacy.utils import save_json
from veriscope.runners.legacy.determinism import seed_worker
from veriscope.runners.legacy.types import DropCfg as _DropCfg

try:
    from filelock import SoftFileLock as FileLock, Timeout as FileLockTimeout
except Exception:  # pragma: no cover
    # Runtime shim: act like a no-op context manager if filelock is unavailable.
    class FileLock:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FileLockTimeout(Exception):
        """Fallback timeout exception when filelock is unavailable."""

        pass


# Default data root follows the same env knob as the legacy CLI.
DATA_ROOT_DEFAULT = os.environ.get("SCAR_DATA", "./data")

# CIFAR-10 class count (mirrors C=10 in the legacy CLI)
C = 10


# ---------------------------
# Core datasets
# ---------------------------


def _cifar_datasets(data_root: Optional[str] = None):
    """
    Return (tr_aug, tr_eval, te_eval) CIFAR-10 datasets.

    - tr_aug: training with data augmentation
    - tr_eval: training split with eval/normalization transforms
    - te_eval: test split with eval/normalization transforms
    """
    root = data_root or DATA_ROOT_DEFAULT
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    tfm_tr = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    tfm_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    if FileLock is not None:
        lock_path = os.path.join(root, "cifar.lock")
        timeout_s = float(os.environ.get("SCAR_DATA_LOCK_TIMEOUT", "120"))
        try:
            with FileLock(lock_path, timeout=timeout_s):
                tr_aug = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tfm_tr)
                tr_eval = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=tfm_eval)
                te_eval = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tfm_eval)
        except FileLockTimeout:
            print(
                f"[WARN] CIFAR lock timeout after {timeout_s:.0f}s at {lock_path}. Assuming stale lock and proceeding.",
                flush=True,
            )
            try:
                Path(lock_path).unlink()
            except Exception:
                pass
            tr_aug = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tfm_tr)
            tr_eval = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=tfm_eval)
            te_eval = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tfm_eval)
    else:
        tr_aug = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tfm_tr)
        tr_eval = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=tfm_eval)
        te_eval = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tfm_eval)

    return tr_aug, tr_eval, te_eval


def _stl10_monitor_dataset(cfg_ext: Dict[str, Any], data_root: Optional[str] = None):
    """
    External monitor dataset (STL10), resized and normalized like CIFAR-10.
    cfg_ext: typically CFG["external_monitor"] from the legacy CLI.
    """
    root = data_root or DATA_ROOT_DEFAULT
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    size = int(cfg_ext["resize_to"])

    tfm = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    if FileLock is not None:
        lock_path = os.path.join(root, "stl10.lock")
        timeout_s = float(os.environ.get("SCAR_DATA_LOCK_TIMEOUT", "120"))
        try:
            with FileLock(lock_path, timeout=timeout_s):
                ds = torchvision.datasets.STL10(
                    root=root,
                    split=str(cfg_ext["split"]),
                    download=True,
                    transform=tfm,
                )
        except FileLockTimeout:
            print(
                f"[WARN] STL10 lock timeout after {timeout_s:.0f}s at {lock_path}. Assuming stale lock and proceeding.",
                flush=True,
            )
            try:
                Path(lock_path).unlink()
            except Exception:
                pass
            ds = torchvision.datasets.STL10(
                root=root,
                split=str(cfg_ext["split"]),
                download=True,
                transform=tfm,
            )
    else:
        ds = torchvision.datasets.STL10(
            root=root,
            split=str(cfg_ext["split"]),
            download=True,
            transform=tfm,
        )

    return ds


# ---------------------------
# DataLoader helpers
# ---------------------------


def make_loader(
    ds,
    batch: int,
    shuffle: bool,
    workers: int,
    gen: Optional[torch.Generator],
    device: torch.device,
    sampler: Optional[Sampler] = None,
    persistent: Optional[bool] = None,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """
    Build a DataLoader with:
    - optional sampler
    - optional persistent_workers
    - deterministic worker seeding when workers > 0

    `pin_memory` and `persistent` are explicit knobs so this module does
    not depend on the global CFG.
    """
    # Decide persistence: allow reuse only when we expect to reuse across epochs
    if persistent is None:
        persistent = (workers > 0) and (sampler is not None)
    else:
        # Clamp: persistent_workers requires num_workers > 0
        persistent = bool(persistent) and (workers > 0)

    # Default pinning behavior mirrors the legacy CLI:
    # - if pin_memory is None, prefer pinning on CUDA
    if pin_memory is None:
        pin = getattr(device, "type", "cpu") == "cuda"
    else:
        pin = bool(pin_memory)

    extra: Dict[str, Any] = {}
    try:
        if "persistent_workers" in inspect.signature(DataLoader).parameters:
            extra["persistent_workers"] = persistent
    except Exception:
        pass

    dl_kwargs: Dict[str, Any] = dict(
        dataset=ds,
        batch_size=batch,
        shuffle=(shuffle and sampler is None),
        num_workers=int(workers),
        pin_memory=pin,
        sampler=sampler,
        drop_last=False,
        worker_init_fn=seed_worker if workers > 0 else None,
    )
    dl_kwargs.update(extra)

    # Some torch versions accept a generator kwarg
    try:
        if gen is not None and "generator" in inspect.signature(DataLoader).parameters:
            dl_kwargs["generator"] = gen
    except Exception:
        pass

    return DataLoader(**dl_kwargs)


def subset_loader_from_indices(
    ds,
    idxs: np.ndarray,
    batch: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    *,
    num_workers: int,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """
    Build a DataLoader over a Subset defined by idxs.

    This loader is reused across epochs, so we request persistent_workers
    when num_workers > 0.
    """
    sub = Subset(ds, [int(i) for i in idxs])
    gen = torch.Generator().manual_seed(100_000 + int(seed))

    if pin_memory is None:
        pin = getattr(device, "type", "cpu") == "cuda"
    else:
        pin = bool(pin_memory)

    extra: Dict[str, Any] = {}
    try:
        if "persistent_workers" in inspect.signature(DataLoader).parameters:
            extra["persistent_workers"] = num_workers > 0
    except Exception:
        pass

    dl_kwargs: Dict[str, Any] = dict(
        dataset=sub,
        batch_size=min(256, batch),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=pin,
        drop_last=False,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    dl_kwargs.update(extra)

    # generator kwarg if available
    try:
        if "generator" in inspect.signature(DataLoader).parameters:
            dl_kwargs["generator"] = gen
    except Exception:
        pass

    return DataLoader(**dl_kwargs)


# ---------------------------
# Split helpers (balanced monitor/norm_ref)
# ---------------------------


def _balanced_indices_by_class(labels: np.ndarray, per_class: int, rng: np.random.Generator):
    """
    Return (take, pools) where:
      - take: list of indices for a balanced per-class subset
      - pools: dict[class -> remaining indices] for reuse
    """
    idx_by_c: Dict[int, list[int]] = {c: [] for c in range(C)}
    for i, y in enumerate(labels):
        idx_by_c[int(y)].append(int(i))

    take: list[int] = []
    for c in range(C):
        pool = np.array(idx_by_c[c], dtype=np.int64)
        rng.shuffle(pool)
        k = min(per_class, len(pool))
        take.extend(pool[:k].tolist())
        idx_by_c[c] = pool[k:]

    rng.shuffle(take)
    return take, {c: np.array(idx_by_c[c], dtype=np.int64) for c in range(C)}


def _take_from_pools(pools: Dict[int, np.ndarray], per_class: int, rng: np.random.Generator):
    """
    Consume from class-index pools to build another balanced subset.
    Mutates `pools` in place to remove taken indices.
    """
    take: list[int] = []
    for c in range(C):
        pool = pools[c]
        pool_copy = pool.copy()
        rng.shuffle(pool_copy)
        k = min(per_class, len(pool_copy))
        take.extend(pool_copy[:k].tolist())
        pools[c] = pool_copy[k:]

    rng.shuffle(take)
    return take


def load_splits(
    seed: int,
    cfg: Dict[str, Any],
    outdir: Path,
    data_root: Optional[str] = None,
):
    """
    Construct train/monitor/norm_ref splits for a given seed.

    Returns:
        tr_aug      – augmented CIFAR-10 training dataset
        tr_take     – list[int] indices for the main training set
        mon_val     – Subset for monitor validation
        norm_ref    – Subset for normalization reference
        splits_path – Path to JSON file with the split indices
    """
    tr_aug, tr_eval, te_eval = _cifar_datasets(data_root=data_root or DATA_ROOT_DEFAULT)
    rng = np.random.default_rng(4242 + int(seed))

    # CIFAR-10 training labels from the eval-view dataset
    labels_train = np.array(tr_eval.targets, dtype=np.int64)

    mon_per_class = int(cfg.get("monitor_val_per_class", 80))
    norm_ref_per_class = int(cfg.get("norm_ref_per_class", 100))

    mon_take, pools = _balanced_indices_by_class(labels_train, mon_per_class, rng)
    mon_val = Subset(tr_eval, mon_take)

    norm_ref_take = _take_from_pools(pools, norm_ref_per_class, rng)
    norm_ref = Subset(tr_eval, norm_ref_take)

    all_idx = np.arange(len(tr_aug), dtype=np.int64)
    mask = np.ones(len(all_idx), dtype=bool)
    if mon_val is not None:
        mask[np.array(mon_take, dtype=np.int64)] = False
    mask[np.array(norm_ref_take, dtype=np.int64)] = False
    tr_take = all_idx[mask].tolist()

    splits_path = Path(outdir) / f"splits_seed{seed}.json"
    save_json(
        dict(
            MONITOR_VAL=(mon_val.indices if mon_val is not None else []),
            NORM_REF=norm_ref_take,
            TRAIN_REST=tr_take,
        ),
        splits_path,
    )
    return tr_aug, tr_take, mon_val, norm_ref, splits_path

    # ---------------------------
    # Factor / corruption helpers
    # ---------------------------


def _u01_from_hash(*xs: object) -> float:
    """Deterministic U(0,1) from a hash of the inputs."""
    h = hashlib.blake2b("::".join(map(str, xs)).encode("utf-8"), digest_size=8).digest()
    v = int.from_bytes(h, byteorder="big", signed=False)
    # Use 53 bits for IEEE754 double mantissa-style [0,1)
    return (v & ((1 << 53) - 1)) / float(1 << 53)


def apply_factor_to_labels(y: np.ndarray, factor: Dict[str, Any], seed: int) -> np.ndarray:
    """
    Apply label-space factors (uniform noise, class skew, long-tail) to a 1D label array.
    Mirrors the legacy CLI behavior.
    """
    rng = np.random.default_rng(10_000 + int(seed))
    out = y.copy()
    name = factor["name"]

    if name == "uniform_label_noise":
        p = float(factor.get("p", 0.0))
        flip = rng.random(len(y)) < p
        if flip.any():
            k = rng.integers(1, C, size=int(flip.sum()))
            out[flip] = (y[flip] + k) % C

    elif name == "class_skew":
        base_p = float(factor.get("base_p", 0.0))
        hot_k = int(factor.get("hot_k", 4))
        hot = rng.choice(np.arange(C), size=max(1, hot_k), replace=False)
        p_class = np.full(C, base_p, dtype=np.float32)
        p_class[hot] = np.clip(base_p * float(factor.get("hot_scale", 1.5)), 0.0, 0.98)
        u = rng.random(len(y))
        flip = u < p_class[y]
        if flip.any():
            k = rng.integers(1, C, size=int(flip.sum()))
            out[flip] = (y[flip] + k) % C

    elif name == "long_tail":
        base_p = float(factor.get("base_p", 0.0))
        a = float(factor.get("pareto_a", 3.0))
        tail = rng.pareto(a=a, size=C)
        tail = tail / tail.max()
        p_class = np.clip(base_p * (0.5 + 0.8 * tail), 0.0, 0.98)
        flip = rng.random(len(y)) < p_class[y]
        if flip.any():
            k = rng.integers(1, C, size=int(flip.sum()))
            out[flip] = (y[flip] + k) % C

    # other factor types leave labels unchanged
    return out


def maybe_corrupt_input(
    x: torch.Tensor,
    factor: Dict[str, Any],
    seed: int,
    epoch: int,
    idx: int,
) -> torch.Tensor:
    """
    Apply input-space corruption for factors that act on x (e.g. blur + noise).
    Deterministic per (seed, epoch, idx).
    """
    name = factor["name"]
    if name != "input_corruption":
        return x

    # Blur (avg pool) with deterministic trigger
    if _u01_from_hash("blur", seed, epoch, idx) < float(factor.get("blur_p", 0.0)):
        k = 3
        pad = (k - 1) // 2
        x = F.avg_pool2d(x.unsqueeze(0), k, stride=1, padding=pad).squeeze(0)

    # Additive Gaussian noise
    std = float(factor.get("noise_std", 0.0))
    if std > 0.0:
        g = torch.Generator(device=x.device).manual_seed(int(1e6 * _u01_from_hash("noise", seed, epoch, idx)))
        noise = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype) * std
        x = (x + noise).clamp_(-3, 3)

    return x


class FactorisedTrainDataset(Dataset):
    """CIFAR10 train indices with a single active factor per run. Provides epoch-aware dropout semantics."""

    def __init__(self, base: torchvision.datasets.CIFAR10, tr_indices: List[int], factor: Dict, seed: int):
        self.base = base
        self.indices = list(tr_indices)
        orig = np.array(base.targets, dtype=np.int64)[np.array(self.indices, dtype=np.int64)]
        self.factor = factor
        self.seed = seed
        # NEW: epoch-gating state (default: active immediately)
        try:
            self.factor_start_epoch = int(factor.get("factor_start_epoch", 0))
        except Exception:
            self.factor_start_epoch = 0
        self.orig_labels = orig.copy()
        self.labels = apply_factor_to_labels(orig, factor, seed)
        self.drop_cfg: Optional[_DropCfg] = None
        if factor["name"] == "class_dropout_window":
            rng = np.random.default_rng(90_000 + seed)
            dc: _DropCfg = {
                "drop_classes": set(
                    rng.choice(
                        np.arange(C),
                        size=max(1, int(factor.get("drop_classes", 1))),
                        replace=False,
                    ).tolist()
                ),
                "drop_frac": float(factor.get("drop_frac", 0.7)),
                "start": int(factor.get("start", 10)),
                "end": int(factor.get("end", 30)),
            }
            self.drop_cfg = dc
        self._epoch: int = 0

    def _factor_active(self) -> bool:
        """Return True iff the run's factor/pathology is active at the current epoch.

        Fail-open to avoid breaking full runs if something odd happens.
        """
        try:
            return int(self._epoch) >= int(self.factor_start_epoch)
        except Exception:
            return True

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def should_drop(self, i: int) -> bool:
        dc = self.drop_cfg
        if dc is None:
            return False
        if not (int(dc["start"]) <= self._epoch < int(dc["end"])):
            return False
        y = int(self.labels[i])
        if y not in dc["drop_classes"]:
            return False
        u = _u01_from_hash("drop", self.seed, self._epoch, int(self.indices[i]))
        return bool(u < float(dc["drop_frac"]))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x, _ = self.base[idx]

        # NEW: epoch gating for input-space corruptions and label factors.
        # Do NOT gate class_dropout_window here; the sampler/dropout policy
        # is already epoch-scoped via should_drop().
        if self.factor.get("name") != "class_dropout_window":
            if not self._factor_active():
                return x, int(self.orig_labels[i])

        x = maybe_corrupt_input(x, self.factor, self.seed, self._epoch, idx)
        return x, int(self.labels[i])


class DropoutAwareSampler(Sampler):
    """Epoch-aware sampler that excludes indices marked for dropout BEFORE batching (deterministic)."""

    def __init__(self, dataset: FactorisedTrainDataset, batch_size: int, seed: int):
        self.ds = dataset
        self.bs = batch_size
        self.seed = seed
        self._cache_epoch: Optional[int] = None
        self._cache_valid: Optional[List[int]] = None

    def set_epoch(self, epoch: int):
        self.ds.set_epoch(epoch)
        self._cache_epoch = None

    def _refresh(self):
        if self._cache_epoch == self.ds._epoch and self._cache_valid is not None:
            return
        valid: List[int] = [i for i in range(len(self.ds)) if not self.ds.should_drop(i)]
        rng = np.random.default_rng(7777 + self.seed + self.ds._epoch)
        rng.shuffle(valid)
        if self.bs and len(valid) >= self.bs:
            valid = valid[: (len(valid) // self.bs) * self.bs]
        self._cache_epoch = self.ds._epoch
        self._cache_valid = valid

    def __iter__(self):
        self._refresh()
        return iter(self._cache_valid or [])

    def __len__(self):
        self._refresh()
        return len(self._cache_valid or [])
