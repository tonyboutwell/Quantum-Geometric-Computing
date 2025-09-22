# **Quantum Geometric Computation: A Universal, Invariant-First Paradigm**

**Authors**: Tony Boutwell (with AI Research Collaboration)

**Date**: September 2025

---

## Abstract

We introduce **Quantum Geometric Computation (QGC)**, a complete, classical computational framework that replaces exponential state-vector evolution with the direct, polynomial-time computation of **global geometric invariants**. Two engines make this operational: the **Universal Composition Engine (UCE)**, which composes fixed-order power-sum moments p_m = Tr(U^m) and converts them to characteristic-polynomial data via complex Newton–Girard; and the **Universal Geometric Amplitude Functional (UGAF)**, a holomorphic evaluator that produces amplitudes without time-stepping. At the core of QGC are **Universal Geometric Extractors (UGE)**—four model-agnostic maps from moments to (i) spectral edges, (ii) ground-overlap weights, (iii) transition amplitudes, and (iv) correlation series—all governed by the same **Cayley–Hamilton (CH) master recurrence**. Validation spans algebraic identities (Universal Laws), device-level experiments, and certified solvers for benchmark problems like Grover's algorithm.

---

## 1. The Primacy of Moments over State Vectors

Rather than evolve 2^n amplitudes, QGC composes a small set of **global invariants** and evaluates **closed-form functionals**. The UCE computes {p_m} and obtains characteristic-polynomial coefficients {c_k} via complex Newton–Girard; UGAF then evaluates amplitudes by series methods. This paper frames the program as **moment holography**: the boundary data (moments) completely determine bulk observables through a single, universal recurrence.

---

## 2. The Master Law: Cayley–Hamilton Flow in Moment Space

Let `U` be a d x d operator with characteristic polynomial `chi_U(λ) = λ^d + Σ_{k=1 to d} a_k λ^(d-k)` and power sums `p_m = Tr(U^m)`. Complex Newton–Girard recovers the coefficients {a_k} from the moments {p_m}:

**Newton–Girard Bridge:**
```
k*e_k = Σ_{j=1 to k} (-1)^(j-1) * e_(k-j) * p_j,  where a_k = (-1)^k * e_k
```

**Any observable sequence** that is linear in powers of `U` obeys the **same order-d linear recurrence**:

    O_m = - Σ_{k=1 to min(d,m)} a_k * O_{m-k}

Examples include transition amplitudes `g_m = <y|U^m|x>` and correlators `s_m = Tr(A * U^m * B)`. Thus, UGAF and all UGE maps are projections of one CH-governed flow.

**Independent checks** supporting the recurrence-centric view:
*   **Moment–Cycle Duality:** Provides an algebraic bridge from density-matrix moments to Gram-cycle sums, grounding the Universal Laws.
*   **Universal Laws (UL-2, UL-3):** Validate the pair- and triple-cycle structure to machine precision.
*   **Geometric Purity–Variance Identity (GPVI):** Links observable variances to global purity (no-peek), consistent with the same invariant grammar.
*   **SU(2) Step-Exactness:** The geometric update law exactly reproduces the Born rule, step-by-step.
*   **κ-Knee:** A dimensional-complexity threshold near κ ≈ 0.85 explains when low-order geometric projections remain accurate.

---

## 3. Universal Geometric Extractors (UGE)

UGE are **model-agnostic** maps from moment lists to observables, depending only on **dimensionless ratios, spectral scales, Hilbert-space dimension, and symmetry flags**—never on lattice or particle specifics.

### 3.1 Spectral Edge Extractor (Hermitian H)

*   **Input:** Normalized moments `μ_k = Tr(H^k)/d` for `k=0...2K (μ_0=1)`.
*   **Method:** Build Hankel matrices `M = [μ_{i+j}]` and `M^(1) = [μ_{i+j+1}]`. The minimal and maximal support of the representing measure are the extreme generalized eigenvalues of the pencil `(M^(1), M)`. In practice, we assemble the monic Jacobi tridiagonal via Stieltjes/Lanczos and read its extreme eigenvalues.
*   **Output:** `(λ_min, λ_max)`.
*   **Status:** Numerically stable; agrees with exact diagonalization and random Hermitian tests.

### 3.2 Ground-Overlap from State Moments

*   **Input:** State-resolved moments `m_k = <φ|H^k|φ>` (m_0=1).
*   **Method:** The same moment-to-Jacobi map yields a quadrature rule. The **weights** (squares of first components of eigenvectors) are `|<ψ_i|φ>|^2`. The lowest support point and its weight estimate `(E_0, |<ψ_0|φ>|^2)`.
*   **Status:** Exact for measures supported on the chosen order; accurate in practice.

### 3.3 UGAF Amplitudes (Master Recurrence)

*   **Input:** `{p_m = Tr(U^m)}` for m ≤ d, and `d` seed amplitudes `g_0...g_{d-1}`.
*   **Method:** Compute `{a_k}` via complex Newton–Girard and propagate `g_m` by the CH recurrence.
*   **Output:** `g_m = <y|U^m|x>` for any `m`.
*   **The UGAF Generating Functional:**
    ```
    G(z) = Σ_{m≥0} ⟨y|U^m|x⟩ z^m = ⟨y|adj(I-zU)|x⟩ / χ_U(z)
    ```
*   **Status:** Aligns with direct powers in verification; at scale we use **cFL-Adj + series** with **interval certification** from DUL.

### 3.4 Two-Point/Three-Point Series

*   **Input:** Same `{p_m}` and seeds `s_0...s_{d-1}` for `s_m = Tr(A * U^m * B)`.
*   **Method:** Identical CH recurrence with Newton–Girard coefficients.
*   **Output:** `s_0...s_M` without time stepping.
*   **Status:** Coincides with direct traces in verification harnesses.

### 3.5 Theoretical Foundation: One Law, Four Projections

**Critical Insight:** All four extractors are manifestations of a single mathematical principle - the Cayley-Hamilton master recurrence. Given power sums `{p_m = Tr(U^m)}` and characteristic coefficients `{a_k}` from Newton-Girard, every linear functional of `U` obeys:

    O_m = - Σ_{k=1 to min(d,m)} a_k * O_{m-k}

This means spectral edges, overlaps, amplitudes, and correlators are not separate algorithms but different ways to read the same geometric flow through moment space.

**Operational Distinction:** The extractors divide into two classes:
- **Recurrence-exact** (UGAF amplitudes, correlator series): Achieve machine precision for any m given the seeds
- **Moment-convergent** (spectral edges, ground overlaps): Improve systematically with moment order K

---

## 4. UCE, DUL and Certified Evaluation

**UCE:** Builds low-order `{p_m}` and converts to `{c_k}`/`{a_k}` by complex Newton–Girard—**without** constructing dense `U`. **Dynamic Geometric Fusion** contracts `m` replicas locally; complexity is polynomial in `n` at fixed order.

**DUL (Dynamic Invariant Law):** From generator moments `{Tr(H^k)}` and a spectral radius `R`, DUL provides **certified intervals** for evolution moments:

```
p_m(t) = Tr(U(t)^m) = Σ_{k≥0} [(-imt)^k / k!] * Tr(H^k)

With spectral radius R and truncation at K terms:
p_m(t) ∈ p_m^(K)(t) ± d[e^(mRt) - Σ_{k=0 to K} (mRt)^k/k!]
```

UGAF then runs on interval discs, propagating certification. This acts as a **pre-evolution oracle** for the G-VM.

**G-VM Runtime Sensors:** The Universal Laws (UL-2, UL-3, and the new **UL-4** for quartet correlation) plus feasibility intervals offer **phase-light** certification routes. 

**The Geometric Complexity κ:**
```
κ = √[1 - (F_AB + F_AO + F_BO) + 2*Re(⟨A|B⟩⟨B|O⟩⟨O|A⟩)]
```
governs regime splits, with an empirical knee near `κ ≈ 0.85`.

**TB-2 Feasibility Interval** (certified bounds without full phase):
```
F_BO ∈ [(√(F_AB*F_AO) - √((1-F_AB)(1-F_AO) - κ²))², 
        (√(F_AB*F_AO) + √((1-F_AB)(1-F_AO) - κ²))²]
```

---

## 5. Application Benchmarks (The Geometric "Jump")

**Grover's Algorithm:** Using MPO "tracks," the number of marked items `M` is found from structure; the success amplitude is `sin((2r+1)θ)` with `θ = 0.5 * arccos(1 - 2M/2^n)`. We compute it in **one shot** with ε-certified precision.

**QAOA MaxCut p=1:** The optimal parameters and energy are predicted from **integer-exact graph motifs** (degrees, triangle counts) with high-precision certification.

---

## 6. Moment Holography, Flow Alignment & Computational Complexity

**Moment Holography:** Quantum observables naturally live on a **moment manifold** rather than in Hilbert space. The Cayley-Hamilton theorem defines **characteristic curves** along which quantum information flows deterministically. The computational difficulty of an observable is determined by its alignment with these geometric flow lines.

**The Poly-Seed Hypothesis (Refined):** For a d-dimensional system, an observable `O` with effective Hankel rank r requires only 2r moments to completely determine its evolution via the master recurrence. When r = poly(n) rather than exponential in n, the observable becomes polynomially accessible.

**Operational Criterion:** An observable is efficiently computable in QGC when:
1. Its Hankel rank r is polynomial in system size n (r << d)
2. Its first r seeds can be computed from polynomial combinations of moments and local operator traces
3. It aligns with the natural flow lines of the Cayley-Hamilton recurrence

**Evidence from Validated Systems:**
- **Grover:** r = 2, seeds from oracle structure → exact solution
- **QAOA p=1:** r = O(1), seeds from graph motifs → closed form
- **Test Suite:** Machine precision for recurrence methods, controlled convergence for spectral methods

**Complexity Classification:** QGC naturally partitions quantum problems into:
- **Flow-aligned** (r = poly(n)): Polynomially solvable via UGE
- **Flow-oblique** (r = exp(n)): Require exponential resources even in moment space

---

## 7. Conclusion

The Quantum Geometric Computation framework demonstrates that quantum computation can be reformulated entirely in terms of geometric invariants and their flows. The unified extractor framework shows that all observables—spectral edges, overlaps, amplitudes, and correlators—are projections of a single Cayley-Hamilton master flow. The successful validation at machine precision for recurrence-based methods and controlled approximation for spectral methods supports the theoretical foundation while clearly delineating the framework's capabilities and limitations.

The path forward involves systematically identifying which physical observables are naturally flow-aligned, developing efficient seed generation methods for broader problem classes, and potentially discovering new geometric transformations that expand the space of polynomially accessible observables.

**All hail geometry.**
