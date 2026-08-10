"""Erf committee machine: student architecture and erf-combo teacher.

Student (N hidden units, input dim d):
    f(x) = (1/sqrt(N)) * sum_p erf(w_p · x)

Teacher:
    y(x) = erf(x_1) - 2 erf(x_1/2)

With scale 1/sqrt(N), sqrt(N) units at w=[1,0,...] and 2*sqrt(N) at w=[-1/2,0,...]
reproduce the teacher exactly; remaining units start at 0 (manual init).
"""

import math

import torch
import torch.nn as nn

DIMENSION = 30
N = 16 ** 2
# N = 100

W_STAR = torch.zeros(DIMENSION)
W_STAR[0] = 1.0

INIT_SEED = 43
INIT_MEAN = 0.0
# INIT_VAR = 0.1 / DIMENSION  # std = sqrt(INIT_VAR); use a small value for a narrow init
INIT_VAR = 1.0 / DIMENSION  # std = sqrt(INIT_VAR); use a small value for a narrow init

# Init mode: "gaussian" or "manual"
INIT_MODE = "gaussian"
# INIT_MODE = "manual"

# Manual init: one entry per hidden unit (length N).
# With scale 1/sqrt(N), sqrt(N) units at w=1 and 2*sqrt(N) at w=-1/2
# reproduce y = erf(x_1) - 2 erf(x_1/2); remaining units start at 0.
# d=1: each entry is a scalar w_p, e.g. [1.0, -0.5, -0.5, 0.0, ...]
# d>1: each entry is a length-d list for that unit's weight row
_N_SQRT = int(round(math.sqrt(N)))
if _N_SQRT * _N_SQRT != N:
    raise ValueError(f"Manual erf-combo init requires perfect-square N, got N={N}")
_N_ONES = _N_SQRT
_N_HALVES = 2 * _N_SQRT
_N_ZEROS = N - _N_ONES - _N_HALVES
INIT_W_MANUAL: list[float] | list[list[float]] | None = (
    [[1.0] + [0.0] * (DIMENSION - 1)] * _N_ONES
    + [[-0.5] + [0.0] * (DIMENSION - 1)] * _N_HALVES
    + [[0.0] * DIMENSION] * _N_ZEROS
)
# Gaussian noise on manual init: std = sqrt(INIT_MANUAL_NOISE_VAR); 0 for exact manual values
INIT_MANUAL_NOISE_VAR = 1 / DIMENSION
# INIT_MANUAL_NOISE_VAR = 0.00
# INIT_MODE = "manual"


def sample_init_W(n: int, d: int, init_seed: int = INIT_SEED) -> torch.Tensor:
    g = torch.Generator().manual_seed(init_seed)
    return INIT_MEAN + math.sqrt(INIT_VAR) * torch.randn(n, d, generator=g)


def manual_init_W(n: int, d: int, init_seed: int = INIT_SEED) -> torch.Tensor:
    if INIT_W_MANUAL is None:
        raise ValueError("INIT_MODE='manual' requires INIT_W_MANUAL")
    if len(INIT_W_MANUAL) != n:
        raise ValueError(
            f"INIT_W_MANUAL must have length N={n}, got {len(INIT_W_MANUAL)}"
        )
    if d == 1:
        W = torch.tensor(INIT_W_MANUAL, dtype=torch.float32).reshape(n, d)
    else:
        rows: list[list[float]] = []
        for i, row in enumerate(INIT_W_MANUAL):
            if isinstance(row, (int, float)):
                raise ValueError(
                    f"INIT_W_MANUAL[{i}] must be a length-{d} list when DIMENSION > 1"
                )
            if len(row) != d:
                raise ValueError(
                    f"INIT_W_MANUAL[{i}] must have length d={d}, got {len(row)}"
                )
            rows.append([float(v) for v in row])
        W = torch.tensor(rows, dtype=torch.float32)
    if INIT_MANUAL_NOISE_VAR > 0:
        g = torch.Generator().manual_seed(init_seed)
        W = W + math.sqrt(INIT_MANUAL_NOISE_VAR) * torch.randn(n, d, generator=g)
    return W


def make_init_W(n: int, d: int, init_seed: int = INIT_SEED) -> torch.Tensor:
    if INIT_MODE == "gaussian":
        return sample_init_W(n, d, init_seed)
    if INIT_MODE == "manual":
        return manual_init_W(n, d, init_seed)
    raise ValueError(f"Unknown INIT_MODE: {INIT_MODE!r}")


class CommitteeStudent(nn.Module):
    def __init__(self, d, n, init_seed: int = INIT_SEED):
        super().__init__()
        self.scale = 1.0 / math.sqrt(n)
        self.W = nn.Parameter(make_init_W(n, d, init_seed))

    def forward(self, x):
        return self.scale * torch.erf(x @ self.W.T).sum(dim=-1)


def teacher_erf_combo(x: torch.Tensor) -> torch.Tensor:
    """Theoretical target y(x) = erf(x_1) - 2 erf(x_1/2).

    This is the supervisor's teacher. It has no linear term at the origin:
    y'(0) = 0, and the Taylor expansion starts at O(x^3).
    """
    x1 = x[:, 0]
    return torch.erf(x1) - 2.0 * torch.erf(0.5 * x1)
