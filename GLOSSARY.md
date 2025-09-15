### Glossary of Terms and Invariants

This section defines the key mathematical objects and concepts used throughout the Quantum Geometric Computation (QGC) framework and its demonstration software.

#### Foundational Mathematical Objects

*   `n`: The number of **qubits** in the system. This is the primary measure of the system's size.
*   `N`: The dimension of the complex vector space, where `N = 2^n`.
*   `U`: An `N x N` **unitary matrix** representing the total evolution of a quantum circuit. It is the primary "map" whose geometry we analyze.
*   `|ψ⟩`: A **ket vector**, representing the `N`-dimensional state of the quantum system.
*   `⟨ψ|`: A **bra vector**, the conjugate transpose of a ket.
*   `H`: An `N x N` **Hamiltonian matrix**, which describes the energy of the system and generates its time evolution (i.e., `U = e^(-iHt)`).

#### QGC Core Engines & Concepts

*   **QGC**: **Quantum Geometric Computation**. The complete computational paradigm.
*   **UCE**: **Universal Composition Engine**. The software engine that converts a circuit description (`U`) into its geometric invariants (`{p_m}`, `{c_k}`).
*   **UGAF**: **Universal Geometric Amplitude Functional**. The mathematical method for calculating exact amplitudes (`⟨y|U|x⟩`) from the invariants supplied by the UCE.
*   **G-VM**: **Geometric Virtual Machine**. The operational architecture that uses the UCE and UGAF to execute programs, often using `κ` to make adaptive decisions.
*   **Dynamic Geometric Fusion**: The proprietary, polynomial-time process used by the UCE to compute the power sum moments (`p_m`) of an operator without forming the full matrix.

#### Geometric & Statistical Invariants

*   `p_m`: The **m-th power sum moment** of the unitary `U`, defined as `p_m = Tr(U^m)`. This is a primary global invariant computed by the UCE.
*   `{c_k}`: The set of **coefficients of the characteristic polynomial** of `U`. These are derived from the `{p_m}` and form the "geometric DNA" of the operator.
*   `χ`: The **bond dimension** of a Matrix Product Operator (MPO). A key measure of the geometric complexity of an operator.
*   `κ`: A **measure of geometric complexity** of a triad of states, derived from their Gram determinant. It is the core metric used by the G-VM's scheduler.
*   `ρ`: The **density matrix**, representing the state of a quantum system, which may be pure or mixed.

#### Grover's Search Specific Variables

*   `G`: The specific `N x N` **Grover iterate operator**, constructed as `G = D @ O`. Applying this operator once constitutes one "step" of the search.
*   `O`: The `N x N` **Oracle operator**, which marks solutions by flipping their phase.
*   `D`: The `N x N` **Diffuser operator**, which amplifies the amplitude of the marked items.
*   `M`: The **number of marked items** (or solutions) in the search space.
*   `|s⟩`: The `N`-dimensional starting state vector, which is a uniform superposition of all basis states.
*   `|W⟩`: A state vector representing the uniform superposition of all `M` marked items. `⟨W|ψ⟩` is the amplitude of being in the "winning" subspace.
*   `r`: The integer **number of Grover iterations** or steps.
*   `θ`: The **Grover angle**, a single geometric invariant that defines the rotation speed of the algorithm in its 2D subspace.

#### QAOA Specific Variables

*   `p`: The number of layers in the QAOA circuit. Our demo focuses on the analytically solvable `p=1` case.
*   `β` (beta), `γ` (gamma): The two variational **angle parameters** of the `p=1` QAOA circuit that are optimized.
*   `E`: The **expectation value of the cost Hamiltonian**, which is the objective function we are trying to maximize. In MaxCut, this is the number of "cut" edges.
*   `deg(v)`: The **degree** of a vertex `v` in the graph (a classical graph invariant).
*   `λ_uv` (lambda): The **number of triangles** that an edge `(u,v)` is part of (another classical graph invariant).

---
