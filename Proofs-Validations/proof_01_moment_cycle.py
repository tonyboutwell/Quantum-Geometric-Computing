#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal NumPy proof of Moment–Cycle Decomposition for k=2,3.

Validates the dimension‑independent identity
    Tr(ρ_N^k) = (1/N^k) Tr(G^k)  (k ∈ {2,3})
by sampling random ensembles in multiple dimensions. This is the algebraic
foundation that underpins UL‑2 and UL‑3.

Notes:
- No κ is used here.
- Machine‑precision agreement indicates the Gram‑geometry ↔ moment duality
  is exact in floating‑point practice.

"""
import numpy as np

def haar_random_state(d: int) -> np.ndarray:
    v = np.random.randn(d) + 1j*np.random.randn(d)
    return v / np.linalg.norm(v)

def moment_cycle_check(N: int, d: int, k: int, trials: int = 8, atol: float = 1e-12):
    assert k in (2,3)
    for _ in range(trials):
        psi = np.column_stack([haar_random_state(d) for _ in range(N)])
        rho = (psi @ psi.conj().T) / N
        G = psi.conj().T @ psi
        lhs = np.trace(np.linalg.matrix_power(rho, k))
        rhs = (1/(N**k)) * np.trace(np.linalg.matrix_power(G, k))
        if not np.allclose(lhs, rhs, atol=atol, rtol=0):
            raise AssertionError(f"Mismatch LHS={lhs}, RHS={rhs}")
    return True

def main():
    np.random.seed(1234)
    for (N,d) in [(3,5),(5,7),(6,4)]:
        for k in (2,3):
            assert moment_cycle_check(N=N, d=d, k=k, trials=6)
    print("✅ proof_01_moment_cycle: PASS — Tr(ρ^k) = (1/N^k) Tr(G^k) (1e-12)." )

if __name__ == "__main__":
    main()
