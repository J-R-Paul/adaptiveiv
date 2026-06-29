from __future__ import annotations

import numpy as np
import pandas as pd


def paper_group_strengths(
    dgp: str,
    *,
    n_groups: int,
    strong_fraction: float = 0.25,
    weak_fraction: float = 0.0,
    weak_strength: float = 0.2,
    seed: int | None = None,
) -> np.ndarray:
    """Return group first-stage strengths for the paper's Section 4 DGPs."""
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    if not 0 <= strong_fraction <= 1:
        raise ValueError("strong_fraction must be between 0 and 1")
    if not 0 <= weak_fraction <= 1:
        raise ValueError("weak_fraction must be between 0 and 1")
    if strong_fraction + weak_fraction > 1:
        raise ValueError("strong_fraction + weak_fraction must be at most 1")

    rng = np.random.default_rng(seed)
    dgp_key = dgp.lower().replace("_", "").replace("-", "")

    if dgp_key == "dgp1":
        n_strong = int(round(n_groups * strong_fraction))
        strengths = np.zeros(n_groups, dtype=float)
        strengths[:n_strong] = 1.0
    elif dgp_key == "dgp2":
        n_strong = int(round(n_groups * strong_fraction))
        n_weak = int(round(n_groups * weak_fraction))
        strengths = np.zeros(n_groups, dtype=float)
        strengths[:n_strong] = 1.0
        strengths[n_strong : n_strong + n_weak] = weak_strength
    elif dgp_key == "dgp3":
        n_relevant = int(round(n_groups * 0.10))
        n_low = n_relevant // 2
        n_high = n_relevant - n_low
        strengths = np.zeros(n_groups, dtype=float)
        strengths[:n_low] = rng.normal(0.2, 0.1, size=n_low)
        strengths[n_low:n_relevant] = rng.normal(1.0, 0.25, size=n_high)
    else:
        raise ValueError("dgp must be one of 'dgp1', 'dgp2', or 'dgp3'")

    rng.shuffle(strengths)
    return strengths


def simulate_paper_section4_dgp(
    *,
    dgp: str = "dgp1",
    n_groups: int = 40,
    n_per_group: int = 500,
    beta: float = 0.0,
    strong_fraction: float = 0.25,
    weak_fraction: float = 0.0,
    weak_strength: float = 0.2,
    rho_uv: float = 0.25,
    error_distribution: str = "normal",
    seed: int | None = None,
    group_strengths: np.ndarray | None = None,
) -> pd.DataFrame:
    """Generate Section 4 Monte Carlo data from Abadie, Gu, and Shen."""
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive")
    if not -1 <= rho_uv <= 1:
        raise ValueError("rho_uv must be between -1 and 1")

    rng = np.random.default_rng(seed)
    if group_strengths is None:
        strengths = paper_group_strengths(
            dgp,
            n_groups=n_groups,
            strong_fraction=strong_fraction,
            weak_fraction=weak_fraction,
            weak_strength=weak_strength,
            seed=seed,
        )
    else:
        strengths = np.asarray(group_strengths, dtype=float).copy()
        if strengths.shape != (n_groups,):
            raise ValueError("group_strengths must have length n_groups")
    dgp_key = dgp.lower().replace("_", "").replace("-", "")
    error_key = error_distribution.lower().replace("_", "").replace("-", "")

    frames = []
    for group, rho_g in enumerate(strengths):
        z = rng.normal(size=n_per_group)
        x = rng.normal(size=n_per_group)
        v = _draw_centered_error(rng, n_per_group, error_key)
        e = _draw_centered_error(rng, n_per_group, error_key)
        u = rho_uv * v + np.sqrt(1.0 - rho_uv**2) * e
        w = rho_g * z + x + v
        y = beta * w + x + u
        frames.append(
            pd.DataFrame(
                {
                    "Y": y,
                    "W": w,
                    "Z": z,
                    "X": x,
                    "u": u,
                    "v": v,
                    "rho_g": rho_g,
                    "group": group,
                    "dgp": dgp_key,
                    "error_distribution": error_key,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _draw_centered_error(
    rng: np.random.Generator,
    size: int,
    error_distribution: str,
) -> np.ndarray:
    if error_distribution == "normal":
        return rng.normal(size=size)
    if error_distribution in {"chisq3", "chi2", "chisquare3"}:
        return rng.chisquare(df=3, size=size) - 3.0
    raise ValueError("error_distribution must be 'normal' or 'chisq3'")


def simulate_paper_dgp(
    *,
    n_groups: int = 40,
    n_per_group: int = 500,
    beta: float = 0.0,
    strong_fraction: float = 0.25,
    weak_fraction: float = 0.0,
    weak_strength: float = 0.2,
    rho_uv: float = 0.4,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a simple Abadie-Gu-Shen style grouped IV DGP."""
    rng = np.random.default_rng(seed)
    n_strong = int(round(n_groups * strong_fraction))
    n_weak = int(round(n_groups * weak_fraction))
    strengths = np.zeros(n_groups)
    strengths[:n_strong] = 1.0
    strengths[n_strong : n_strong + n_weak] = weak_strength
    rng.shuffle(strengths)

    frames = []
    for group, rho_g in enumerate(strengths):
        z = rng.normal(size=n_per_group)
        x = rng.normal(size=n_per_group)
        v = rng.normal(size=n_per_group)
        e = rng.normal(size=n_per_group)
        u = rho_uv * v + np.sqrt(1.0 - rho_uv**2) * e
        w = rho_g * z + x + v
        y = beta * w + x + u
        frames.append(
            pd.DataFrame(
                {
                    "Y": y,
                    "W": w,
                    "Z": z,
                    "X": x,
                    "u": u,
                    "v": v,
                    "rho_g": rho_g,
                    "group": group,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
