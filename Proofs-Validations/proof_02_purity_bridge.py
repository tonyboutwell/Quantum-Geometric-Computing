#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal NumPy proof of UL‑2 Purity Bridge on a triad.

Triad identity (N=3):
    Tr(ρ_Δ^2) = (1 + 2(F_AB + F_AO + F_BO)) / 9
⇔  S2 = F_AB + F_AO + F_BO = (9 Tr(ρ_Δ^2) - 3)/2.

Confirms the algebra used by the no‑peek overlap recovery and purity
calibration routines. No κ enters this proof.
"""
import numpy as np

def normalize(v): 
    n = np.linalg.norm(v); 
    return v if n == 0 else v/n

def haar_state(d: int) -> np.ndarray:
    v = np.random.randn(d) + 1j*np.random.randn(d)
    return normalize(v)

def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(normalize(a), normalize(b)))**2)

def triad_purity(A: np.ndarray, B: np.ndarray, O: np.ndarray) -> float:
    kets = [normalize(A), normalize(B), normalize(O)]
    rho = sum(np.outer(k, k.conj()) for k in kets) / 3.0
    return float(np.real(np.trace(rho @ rho)))

def main():
    np.random.seed(2025)
    for d in (3,5,8):
        for _ in range(10):
            A,B,O = haar_state(d), haar_state(d), haar_state(d)
            F_AB = fidelity(A,B); F_AO = fidelity(A,O); F_BO = fidelity(B,O)
            S2_true = F_AB + F_AO + F_BO
            purity = triad_purity(A,B,O)
            S2_via_purity = (9.0*purity - 3.0)/2.0
            assert abs(S2_true - S2_via_purity) < 1e-10, (S2_true, S2_via_purity)
    print("✅ proof_02_purity_bridge: PASS — UL-2 holds to 1e-10.")

if __name__ == "__main__":
    main()
