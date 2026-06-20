from __future__ import annotations
import numpy as np


def trace_field_line_rk4(
    B_func,
    x0,
    box,
    *,
    ds=0.01,
    n_steps=4000,
    min_norm=1e-12,
):
    x = np.zeros((n_steps, 3), dtype=float)
    x[0] = np.asarray(x0, dtype=float)

    for i in range(n_steps - 1):
        curr = x[i].copy()
        curr[0] %= box.Lx
        curr[1] %= box.Ly
        curr[2] %= box.Lz

        b1 = B_func(curr)
        n1 = np.linalg.norm(b1)
        if n1 < min_norm:
            break
        k1 = (b1 / n1) * ds

        p2 = curr + 0.5 * k1
        b2 = B_func(p2)
        n2 = np.linalg.norm(b2)
        if n2 < min_norm:
            break
        k2 = (b2 / n2) * ds

        p3 = curr + 0.5 * k2
        b3 = B_func(p3)
        n3 = np.linalg.norm(b3)
        if n3 < min_norm:
            break
        k3 = (b3 / n3) * ds

        p4 = curr + k3
        b4 = B_func(p4)
        n4 = np.linalg.norm(b4)
        if n4 < min_norm:
            break
        k4 = (b4 / n4) * ds

        nxt = curr + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        nxt[0] %= box.Lx
        nxt[1] %= box.Ly
        nxt[2] %= box.Lz

        x[i + 1] = nxt

    return x
