#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerical tests of κ physics and the TB‑2 interval (phase‑aware).

Part A (pure triads):
- Uses the phase‑aware κ = √det(G) with
  det(G) = 1 - (F_AB + F_AO + F_BO) + 2 Re(Z_ABO),
  where Z_ABO = ⟨A|B⟩⟨B|O⟩⟨O|A⟩ and F_ij = |⟨i|j⟩|².
- Verifies: (i) CP¹ triads have κ≈0 and the SU(2) feasibility envelope
  contains F_BO; (ii) for arbitrary triads, the TB‑2 interval computed
  from (F_AB,F_AO,κ) contains the true F_BO.

Part B (optional, mixed / channels):
- Demonstrates a fidelity‑only diagnostic κ̃ (proxy) decreases under a
  unital depolarizing channel. TB‑2 and certified bounds always use
  phase‑aware κ; κ̃ is reporting/visualization only.
"""
import numpy as np

def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v/n

def haar(d):
    v = np.random.randn(d) + 1j*np.random.randn(d)
    return normalize(v)

def fidelity(a,b):
    return float(abs(np.vdot(normalize(a), normalize(b)))**2)

def gamma_bargmann(a,b,o):
    z = np.vdot(a,b)*np.vdot(b,o)*np.vdot(o,a)
    return 0.0 if abs(z) < 1e-14 else float(np.angle(z))

def kappa_true(a,b,o):
    Fab = fidelity(a,b); Fao = fidelity(a,o); Fbo = fidelity(b,o)
    ReZ = float(np.real(np.vdot(a,b)*np.vdot(b,o)*np.vdot(o,a)))
    val = 1.0 - (Fab + Fao + Fbo) + 2.0*ReZ
    return float(np.sqrt(max(val, 0.0)))

def envelope_bounds(Fab, Fao):
    center = 0.5*(1.0 + (2*Fab-1.0)*(2*Fao-1.0))
    span   = 2.0*np.sqrt(Fab*(1.0-Fab)*Fao*(1.0-Fao))
    return (center - span, center + span)

def tb2_interval(Fab, Fao, kappa):
    a, b, k = float(Fab), float(Fao), float(kappa)
    term = (1.0 - a)*(1.0 - b) - k*k
    if term < 0: term = 0.0
    cmin = (np.sqrt(max(a*b,0.0)) - np.sqrt(term))**2
    cmax = (np.sqrt(max(a*b,0.0)) + np.sqrt(term))**2
    return (float(np.clip(cmin, 0.0, 1.0)), float(np.clip(cmax, 0.0, 1.0)))

def part_A():
    np.random.seed(7)
    # (1) CP1 triads: envelope must contain F_BO; κ_true ~ 0
    d = 7
    u = haar(d); v_raw = haar(d)
    v = normalize(v_raw - np.vdot(u, v_raw)*u)
    def span2():
        a = np.random.randn(2) + 1j*np.random.randn(2)
        return normalize(a[0]*u + a[1]*v)
    for _ in range(10):
        A,B,O = span2(), span2(), span2()
        k = kappa_true(A,B,O)
        assert k < 5e-8, f"kappa_true should be ~0, got {k}"
        Fab, Fao, Fbo = fidelity(A,B), fidelity(A,O), fidelity(B,O)
        lo,hi = envelope_bounds(Fab, Fao)
        assert lo - 1e-12 <= Fbo <= hi + 1e-12, f"CP1: F_BO={Fbo} not in [{lo},{hi}]"
    # (2) Arbitrary triads: use TB-2 κ interval
    for _ in range(50):
        A,B,O = haar(5), haar(5), haar(5)
        Fab, Fao, Fbo = fidelity(A,B), fidelity(A,O), fidelity(B,O)
        k = kappa_true(A,B,O)
        lo,hi = tb2_interval(Fab, Fao, k)
        assert lo - 1e-12 <= Fbo <= hi + 1e-12, f"TB-2: F_BO={Fbo} not in [{lo},{hi}] (κ={k})"
    print("✅ Part A: PASS — CP¹ envelope holds; general triads satisfy TB‑2 with phase‑aware κ.")

def part_B():
    try:
        from qutip import Qobj, ket2dm, fidelity as qf
    except Exception:
        print("ℹ️  QuTiP not available — skipping Part B (install `qutip`).")
        return
    import numpy as np
    def depol(rho, p):
        from qutip import Qobj
        I = Qobj(np.eye(2))
        return (1-p)*rho + (p/2.0)*I
    def F_dm(r1, r2): 
        return float(qf(r1, r2)**2)
    def kappa_diag(Fab, Fao, Fbo):
        detG = 1.0 + 2.0*Fab*Fao*Fbo - (Fab*Fab + Fao*Fao + Fbo*Fbo)
        return float(np.sqrt(max(detG, 0.0)))
    def rand_qubit():
        v = np.random.randn(2) + 1j*np.random.randn(2)
        return ket2dm(Qobj(v/np.linalg.norm(v)))
    A,B,O = rand_qubit(), rand_qubit(), rand_qubit()
    ps = np.linspace(0.0, 0.9, 10)
    vals = []
    for p in ps:
        Ap,Bp,Op = depol(A,p), depol(B,p), depol(O,p)
        Fab, Fao, Fbo = F_dm(Ap,Bp), F_dm(Ap,Op), F_dm(Bp,Op)
        vals.append(kappa_diag(Fab, Fao, Fbo))
    vals = np.array(vals)
    diffs = np.diff(vals)
    assert np.all(diffs <= 1e-9 + 1e-9*np.abs(vals[1:])), f"κ̃ should be nonincreasing; diffs={diffs}"
    print("✅ Part B: PASS — diagnostic κ̃ nonincreasing under depolarizing.")

if __name__ == "__main__":
    part_A()
    part_B()
