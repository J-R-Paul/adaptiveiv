import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def valid_iv_data():
    return make_valid_iv_data()


def make_valid_iv_data(n_per_group=120, n_groups=12, beta=0.75, seed=123):
    rng = np.random.default_rng(seed)
    rows = []
    strengths = np.array([1.0, 0.8, 0.6, 0.35, 0.2, 0.0] * 2)[:n_groups]
    for group, rho in enumerate(strengths):
        z = rng.normal(size=n_per_group)
        x1 = rng.normal(size=n_per_group)
        x2 = rng.binomial(1, 0.45, size=n_per_group)
        v = rng.normal(size=n_per_group)
        e = rng.normal(size=n_per_group)
        u = 0.45 * v + np.sqrt(1 - 0.45**2) * e
        w = rho * z + 0.4 * x1 - 0.2 * x2 + v
        y = beta * w + 0.3 * x1 + 0.1 * x2 + u
        rows.append(
            pd.DataFrame(
                {
                    "Y": y,
                    "W": w,
                    "Z": z,
                    "X1": x1,
                    "X2": x2,
                    "group": group,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)

