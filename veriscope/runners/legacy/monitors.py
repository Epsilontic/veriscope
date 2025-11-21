# veriscope/runners/legacy/monitors.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def monitor_entropy(model, loader, device) -> float:
    model.eval()
    ent_sum = 0.0
    total = 0
    with torch.inference_mode():
        it = iter(loader)
        while True:
            try:
                xb, _ = next(it)
            except StopIteration:
                break
            except Exception as e:
                print(f"[WARN] monitor fetch failed: {e}")
                break
            xb = xb.to(device)
            p = F.softmax(model(xb), dim=1).clamp_min(1e-12)
            ent = -(p * p.log()).sum(dim=1)
            ent_sum += float(ent.sum().item())
            total += xb.shape[0]
    return ent_sum / max(1, total)


def monitor_margin_median(model, loader, device) -> float:
    model.eval()
    margins: list[torch.Tensor] = []
    with torch.inference_mode():
        it = iter(loader)
        while True:
            try:
                xb, _ = next(it)
            except StopIteration:
                break
            except Exception as e:
                print(f"[WARN] monitor fetch failed: {e}")
                break
            xb = xb.to(device)
            logits = model(xb)
            top2 = torch.topk(logits, k=2, dim=1).values
            m = (top2[:, 0] - top2[:, 1]).detach().cpu().float()
            margins.append(m)
    if not margins:
        return float("nan")
    mm = torch.cat(margins, dim=0)
    return float(torch.median(mm).item())


def monitor_avg_conf(model, loader, device) -> float:
    model.eval()
    s = 0.0
    n = 0
    with torch.inference_mode():
        it = iter(loader)
        while True:
            try:
                xb, _ = next(it)
            except StopIteration:
                break
            except Exception as e:
                print(f"[WARN] monitor fetch failed: {e}")
                break
            xb = xb.to(device)
            p = F.softmax(model(xb), dim=1)
            m = p.max(dim=1)[0]
            s += float(m.sum().item())
            n += xb.shape[0]
    return s / max(1, n)


def monitor_accuracy(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        it = iter(loader)
        while True:
            try:
                xb, yb = next(it)
            except StopIteration:
                break
            except Exception as e:
                print(f"[WARN] monitor fetch failed: {e}")
                break
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(1, total)


__all__ = [
    "monitor_entropy",
    "monitor_margin_median",
    "monitor_avg_conf",
    "monitor_accuracy",
]