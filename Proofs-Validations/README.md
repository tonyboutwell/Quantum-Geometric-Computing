# Proofs / Validations

This folder contains self‑contained proofs and validation suites for the **Quantum Geometric Computing (QGC)** white paper. Each script prints a pass/fail summary and (where relevant) saves small artifacts (CSVs/PNGs).

---

Basic math-only proofs:

### proof_01_moment_cycle.py
Verification of the Moment–Cycle Decomposition Theorem (Tr(ρ_N^k) = (1/N^k) Tr(G^k) for k=2, 3).The foundational theorem upon which UL-2 and UL-3 are built is numerically verified. This is the cornerstone of the entire QGC framework.

RESULTS: 
✅ proof_01_moment_cycle: PASS — Tr(ρ^k) = (1/N^k) Tr(G^k) (1e-12).

---

### python proof_02_purity_bridge.py
Verification of the Universal Law 2 (UL-2), the Purity Bridge, for a triad of states (N=3). This directly supports the claim that the G-VM can algebraically solve for a missing overlap using one purity and two other overlap measurements. This is a key operational primitive.

RESULTS: 
✅ proof_02_purity_bridge: PASS — UL-2 holds to 1e-10.

---

### proof_03_kappa_physics.py
Verification of κ physics, including the CP¹/SU(2) envelope and the general TB-2 interval, and the monotonicity of the diagnostic κ̃ under a unital channel. Part A: The PASS result provides strong numerical evidence for two key claims:
1) When geometric complexity κ is zero, the triad behaves like a qubit system, and the tight SU(2) law applies.
2) When κ is non-zero, the SU(2) law fails, but the more general, κ-aware TB-2 interval provides a correct and certified bound. This validates the G-VM's core scheduling logic: calculate κ to know which physical law to apply.

RESULTS: 
✅ Part A: PASS — CP¹ envelope holds; general triads satisfy TB‑2 with phase‑aware κ.
When geometric complexity κ is zero, the triad behaves like a qubit system, and the tight SU(2) law applies.

✅ Part B: PASS — diagnostic κ̃ nonincreasing under depolarizing.
This result provides direct numerical evidence for the κ-monotone theorem. It shows that unital noise (depolarization) systematically reduces the geometric complexity of the triad, as measured by κ̃. This supports the profound claim that κ acts as a resource that is consumed by noise, justifying the "Second Law of Context." While this test uses the proxy κ̃, the argument extends to the true κ as well, since Uhlmann fidelity (used in the true κ) is also monotone.
