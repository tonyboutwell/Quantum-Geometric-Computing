# QGC Patent Evaluation Repository (Confidential)

> **CONFIDENTIAL — Attorney Evaluation Copy**
> This private repository is shared solely for patent evaluation. Do not redistribute. No open-source license is granted. All rights reserved.

## Elevator summary (30s)

We disclose an invariant-based computational architecture that replaces state-vector evolution with (i) **geometric invariants** computed in polynomial time at fixed order and (ii) a **holomorphic evaluation rule** that yields exact observables without time-stepping. The system consists of:

1. the **Universal Composition Engine (UCE)** that composes linear evolutions entirely in invariant space (moments, characteristic-polynomial coefficients),
2. the **Universal Geometric Amplitude Functional (UGAF)** that computes amplitudes via a corrected Faddeev–Leverrier / adjugate **series-division** recurrence, and
3. a **Geometric Virtual Machine (G-VM)** that applies these primitives with κ-aware scheduling and built-in certification.

---

## Independent claim scaffolds

### 1) System — Universal Composition Engine (UCE)

A processor-implemented system that receives a description of a linear evolution (e.g., a quantum circuit), constructs **replica transfers**, computes fixed-order **power-sum moments** $p_m=\mathrm{Tr}(U^m)$ and maps them to **characteristic-polynomial coefficients** $\{c_k\}$ via **complex Newton–Girard**, **without constructing a state vector**, and emits $\{p_m\},\{c_k\}$ for downstream evaluation.

### 2) Method — Universal Geometric Amplitude Functional (UGAF)

A method that generates adjugate coefficients by a **corrected Faddeev–Leverrier** procedure (retaining complex coefficients) and evaluates observables by **series-division**, computing amplitudes $g_m$ **directly from** $\{c_k\}$, yielding exact results without iterating the evolution.

### 3) Application — Geometric Virtual Machine (G-VM)

A computing system that predicts or measures **global invariants** (e.g., purity and triple-phase sums) and a **geometric complexity** κ, uses certified bounds (e.g., TB-2) and SU(2) local laws to select execution routes (invariant evaluation vs. refresh), and outputs **ε-certified** results with a digits-vs-tolerance escalation policy.

---

## Key dependent-claim hooks

* **Replica-based invariant composition** at fixed order $m$ (poly in $n$).
* **Complex Newton–Girard** mapping of $p_1,\dots,p_M$ to $\{c_k\}$.
* **Adjugate series-division** recurrence producing $g_m$ exactly from $\{c_k\}$.
* **Invariant Grover jump**: derive $M$ and $\theta$ from traces and compute $\sin((2r{+}1)\theta)$ in closed-form (no iteration of $G$); includes **ε-certified**, arbitrary-precision implementation and parallel throughput.
* **κ-aware scheduler**: TB-2 interval and SU(2) certainty law determine refresh/routing; **UL-2/UL-3** serve as global audit sensors.

---

## What’s novel & non-obvious

* **State-free composition**: circuits are composed **in invariant space** rather than via state-vector or path evolution.
* **Holomorphic evaluation**: observables arise from **closed-form series-division** fed by UCE, not by simulation or time-stepping.
* **Operational audit laws**: global identities (purity/phase triples) implemented as **runtime sensors** guiding scheduling and certification.
* **Certified Grover**: angle recovery + closed-form amplitude with ε-bounded arbitrary precision; demonstrated parallel scale-out.

---

## Repo map (for reviewers)

* `README.md` — this document.
*  QGC_White_Paper.md
* `Proofs-Validations/` — white-paper text and figures (UCE, UGAF, UL-2/UL-3, κ/TB-2).
  * `README.md` — Descriptions and results for each proof.
  * `proof_01_moment_cycle.py` — Minimal NumPy proof of Moment–Cycle Decomposition for k=2,3.
  * `proof_02_purity_bridge.py` — Verification of the Universal Law 2 (UL-2), the Purity Bridge, for a triad of states (N=3).
  * `proof_03_kappa_physics.py` — Verification of κ physics, including the CP¹/SU(2) envelope and the general TB-2 interval, and the monotonicity of the diagnostic κ̃ under a unital channel.
* `Examples/`
  * `Grover-Benchmark/` 
    * `README.md` — Description of the program and results
    * `grover_parallel.py` — This multi-core demo is a strategic analyzer for quantum search problems.
  * `QAOA-Demo/`
    * `README.md` — Description of the program and results
    * `qaoa_maxcut_invariant_predictor.py` — This demo solves a fundamental combinatorial optimization problem known as MaxCut. 


---

## Enablement pointers (what we disclose)

* Construction of **replica transfers** and contraction order.
* Mapping moments → coefficients via **complex** Newton–Girard.
* **Corrected Faddeev–Leverrier** and adjugate coefficient generation.
* **Series-division** recurrence for $g_m$ with numerical-stability notes.
* **ε-certification**: digits-vs-ε policy; escalation and acceptance criteria.
* **Scheduler**: κ/TB-2 thresholds; UL-2/UL-3 acquisition/usage; SU(2) certainty update law.

---

## Legal/IP hygiene (please read)

* Keep this repo **private**; share access **only with counsel** (or others under NDA).
* **Do not** add an open-source license. Leave as **All Rights Reserved**.
* Avoid public Issues/Discussions; keep communication off-platform or under NDA.
* If you need to show non-lawyers, consider filing a **provisional** first or use a **sanitized demo** branch.
* Mark documents “**Confidential—Patent Evaluation**”; while labels don’t create privilege, they help set expectations.
* Avoid committing secrets (keys/tokens). Enable secret scanning; review commit history before granting access.

---

## Contact

**PI:** Tony Boutwell

**Purpose:** Patent evaluation of QGC (UCE/UGAF/G-VM), Grover-by-invariants, κ-aware scheduling, and certification methodology.

**Email:** *Tony.Boutwell@QuantumGeometric.com*
