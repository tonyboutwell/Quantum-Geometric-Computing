### Glossary of Terms and Invariants

This section defines the key mathematical objects and concepts used throughout the Quantum Geometric Computation (QGC) framework and its demonstration software.

---

#### Foundational Mathematical Objects

*   `n`: The number of **qubits** in the system. This is the primary measure of the system's size.
*   `N`: The dimension of the complex vector space, where `N = 2^n`.
*   `U`: An `N x N` **unitary matrix** representing the total evolution of a quantum circuit. It is the primary "map" whose geometry we analyze.
*   `|ψ⟩`: A **ket vector**, representing the `N`-dimensional state of the quantum system.
*   `⟨ψ|`: A **bra vector**, the conjugate transpose of a ket.
*   `H`: An `N x N` **Hamiltonian matrix**, which describes the energy of the system and generates its time evolution (i.e., `U = e^(-iHt)`).
*   `ρ`: The **density matrix**, representing the state of a quantum system, which may be pure or mixed.
*   `G_{ij}`: The **Gram matrix** entry `⟨ψ_i|ψ_j⟩`, encoding overlaps between states in an ensemble.
*   `F_{ij}`: The **fidelity** `|G_{ij}|²`, the magnitude-squared of Gram entries.

---

#### QGC Core Engines & Concepts

*   **QGC**: **Quantum Geometric Computation**. The complete computational paradigm that replaces state-vector evolution with geometric invariant computation.
*   **UCE**: **Universal Composition Engine**. The engine that converts a circuit description (`U`) into its geometric invariants (`{p_m}`, `{c_k}`) via Dynamic Geometric Fusion.
*   **UGAF**: **Universal Geometric Amplitude Functional**. The holomorphic method for calculating exact amplitudes (`⟨y|U|x⟩`) from invariants via the Cayley-Hamilton recurrence.
*   **UGE**: **Universal Geometric Extractors**. Four model-agnostic maps from moments to observables: (i) spectral edges, (ii) ground-overlap weights, (iii) transition amplitudes, (iv) correlation series.
*   **G-VM**: **Geometric Virtual Machine**. The runtime architecture that uses UCE and UGAF, with `κ`-aware scheduling to choose optimal computational paths.
*   **DUL**: **Dynamic Invariant Law**. Provides certified intervals for evolution moments from generator moments and spectral radius, enabling pre-evolution certification.
*   **Dynamic Geometric Fusion**: The polynomial-time process used by the UCE to compute power sum moments (`p_m`) without forming full matrices.
*   **Cayley-Hamilton Recurrence**: The master recurrence governing all QGC extractors: `O_m = -Σ a_k O_{m-k}`, where `{a_k}` are characteristic polynomial coefficients.

---

#### Geometric & Statistical Invariants

*   `p_m`: The **m-th power sum moment** of the unitary `U`, defined as `p_m = Tr(U^m)`. This is a primary global invariant computed by the UCE.
*   `μ_k`: The **k-th Hamiltonian moment**, defined as `μ_k = Tr(H^k)/d`. Used in spectral estimation and ground-state methods.
*   `{c_k}`: The set of **coefficients of the characteristic polynomial** of `U`. Derived from `{p_m}` via Newton-Girard identities; forms the "geometric DNA" of the operator.
*   `χ`: The **bond dimension** of a Matrix Product Operator (MPO). A measure of geometric complexity of an operator.
*   `κ` (kappa): A **measure of geometric complexity**. General form: `κ(M) = ‖offdiag(M)‖_F / ‖M‖_F`. For triads: derived from Gram determinant. Values range 0→1; **κ ≈ 0.85** marks the threshold where low-order geometric methods require escalation.

---

#### Universal Laws (UL)

Exact algebraic identities relating moments to physical observables, validated to machine precision (≤10⁻¹²).

*   **UL-2**: Purity bridge — relates `Tr(ρ²)` to pair correlations (2-cycles in Gram matrix).
*   **UL-3**: Phase bridge — relates `Tr(ρ³)` to triple-phase structure (Bargmann invariant, 3-cycles).
*   **UL-4**: Quartet law — relates `Tr(ρ⁴)` to 4-cycle structure and pair-mixing terms.
*   **UL-5**: SU(2) certainty law — step-exact Born-rule update for geometric evolution.
*   **UL-6**: Sextet law — relates `Tr(ρ⁶)` to 19-motif expansion with orbit normalizers.
*   **UL-7**: Septet law — relates `Tr(ρ⁷)` to odd-order structure with loop corrections.

---

#### Feasibility Intervals

*   **TB-2**: Certified bounds on `Tr(ρ³)` from magnitudes alone (without full phase information).
*   **TB-4**: Certified bounds on `Tr(ρ⁴)` from magnitudes alone, using cycle magnitude bounds `B_3`, `B_4`.

These intervals enable **phase-light certification**: the G-VM can make decisions without expensive phase measurements when intervals are decisive.

---

#### Grover's Search Specific Variables

*   `G`: The `N x N` **Grover iterate operator**, constructed as `G = D @ O`. One application constitutes one "step" of the search.
*   `O`: The `N x N` **Oracle operator**, which marks solutions by flipping their phase.
*   `D`: The `N x N` **Diffuser operator**, which amplifies the amplitude of marked items.
*   `M`: The **number of marked items** (solutions) in the search space.
*   `|s⟩`: The starting state vector, a uniform superposition of all basis states.
*   `|W⟩`: The uniform superposition of all `M` marked items. `⟨W|ψ⟩` is the amplitude in the "winning" subspace.
*   `r`: The integer **number of Grover iterations**.
*   `θ`: The **Grover angle**, the geometric invariant defining rotation speed in the algorithm's 2D subspace.

---

#### QAOA Specific Variables

*   `p`: The number of layers in the QAOA circuit. The `p=1` case is analytically solvable.
*   `β` (beta), `γ` (gamma): The variational **angle parameters** of the `p=1` QAOA circuit.
*   `E`: The **expectation value of the cost Hamiltonian** (objective function). In MaxCut, this counts "cut" edges.
*   `deg(v)`: The **degree** of vertex `v` in the graph.
*   `λ_uv` (lambda): The **number of triangles** containing edge `(u,v)`.

---

#### Hubbard Model Specific Variables

*   `L`: Total number of lattice sites (`L = nx × ny` for 2D).
*   `t`: **Hopping parameter** — energy scale for electron tunneling between adjacent sites.
*   `U` (Hubbard): **On-site interaction strength** — energy cost for double occupancy (distinct from unitary `U`).
*   `n` (filling): **Electron density** per site. Half-filling means `n = 1` (one electron per site on average).
*   `α(U)`: **Dimensional coupling function** — maps 1D Lieb-Wu solution to 2D bulk energy. Exact limits: `α(0) = 2/π`, `α(∞) → 0.843`.
*   `κ` (Hubbard): **Spectral depth parameter** — determines how far below mean energy the ground state lies: `E₀ ≈ μ₁ - κ×σ`.
*   `⟨D⟩`: **Double occupancy** — fraction of sites with both spin-up and spin-down electrons.

---
