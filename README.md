# Quantum Geometric Computation (QGC)

**An invariant-first framework for quantum and classical computation**

QGC computes quantum observables through geometric invariants—moments, Gram structures, and curvature measures—rather than exponential state-vector enumeration. The framework provides polynomial-time algorithms for problems traditionally requiring exponential resources.

## Core Idea

Instead of tracking 2ⁿ amplitudes, QGC extracts answers from low-order invariants of the Hamiltonian or density matrix:

```
Traditional: |ψ⟩ → evolve → measure → statistics
QGC:         H → moments → geometry → observables
```

The key insight is that many quantum observables depend only on global geometric properties (traces, cycles, curvature) that can be computed efficiently.

## Framework Components

### Universal Composition Engine (UCE)
Computes power-sum moments Tr(ρᵏ) via Cayley-Hamilton recurrence relations, avoiding explicit matrix powers.

### Universal Laws (UL-2 → UL-7)
Exact algebraic identities relating moments to physical observables:
- **UL-2**: Purity from pair correlations
- **UL-3**: Triple-phase / Bargmann invariant
- **UL-4**: Quartet correlations and 4-cycles
- **UL-5**: SU(2) certainty law (step-exact Born rule)
- **UL-6/7**: Higher-order motif expansions

All validated to machine precision (10⁻¹² – 10⁻¹⁵).

### The κ Invariant
A geometric complexity parameter measuring "how much energy is in correlations":

```
κ = ‖offdiag(M)‖ / ‖M‖
```

Critical threshold at **κ ≈ 0.85** marks transitions between:
- Efficient geometric computation (κ < 0.85)
- Required escalation to higher-order methods (κ ≥ 0.85)

This threshold has been validated across multiple domains.

## Validated Applications

### Hubbard Model Solver
The `Examples/Hubbard-Model/` directory contains a working solver achieving **~0.5% mean error** against Simons Collaboration QMC benchmarks. Runs in milliseconds vs. hours for traditional methods.

### κ-Threshold Universality
The κ ≈ 0.85 threshold validated in:
- **Quantum systems**: 10× error amplification at threshold
- **Fluid turbulence**: Johns Hopkins Database, p < 10⁻⁸⁰
- **Gravitational chaos**: 3-body problem, 77% prediction accuracy
- **Protein folding**: 88% success rate with κ-guided intervention

### Chaos Prediction
Geometric features (κ trajectories, curvature) predict chaotic vs. regular behavior without simulation. See `TECHNICAL_WHITEPAPER_V3_GEOMETRIC_CHAOS_PREDICTION.md`.

## Key Results

| Domain | Method | Result |
|--------|--------|--------|
| 2D Hubbard | Moment-based solver | 0.48% mean error vs QMC |
| Quantum circuits | UL-law validation | Machine precision |
| 3-body chaos | κ-threshold prediction | 77% accuracy |
| Turbulence | κ correlation | p < 10⁻⁸⁰ significance |

## Repository Structure

```
├── Core-Files/
│   └── universal_composition_engine.py   # UCE implementation
├── Examples/
│   ├── Grover-Benchmark/                 # Grover's algorithm via invariants
│   ├── Hubbard-Model/                    # 2D Hubbard solver (~0.5% error)
│   └── QAOA-Demo/                        # QAOA MaxCut prediction
├── Proofs-Validations/
│   ├── proof_01_moment_cycle.py          # Tr(ρᵏ) = Tr(Gᵏ)/Nᵏ identity
│   ├── proof_02_purity_bridge.py         # UL-2 validation
│   └── proof_03_kappa_physics.py         # κ threshold behavior
├── GLOSSARY.md                           # Term definitions
├── QGC_White_Paper.md                    # Full technical description
└── README.md
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/tonyboutwell/Quantum-Geometric-Computing.git
cd Quantum-Geometric-Computing

# Try the Hubbard solver
cd Examples/Hubbard-Model
python qgc_hubbard_model_explorer.py bench

# Run proof validations
cd ../../Proofs-Validations
python proof_01_moment_cycle.py
python proof_02_purity_bridge.py
```

## Requirements

- Python 3.8+
- NumPy
- SciPy

## Current Status

QGC is an active research project. The Hubbard solver is validated and usable. Other components (chaos prediction, protein folding) are documented but experimental.

**What works now:**
- Hubbard model ground state energies (half-filling)
- Grover's algorithm via invariants
- QAOA MaxCut invariant prediction
- UL-law validation suite (proofs 01-03)

**In development:**
- Doping support for Hubbard
- Additional UL proofs (04-10)
- Hardware validation on quantum devices
- Chaos prediction toolkit

## References

- Simons Collaboration, Phys. Rev. X 5, 041041 (2015)
- Qin, Shi, Zhang, Phys. Rev. B 94, 085103 (2016)  
- Lieb & Wu, Phys. Rev. Lett. 20, 1445 (1968)

## Contact

Tony Boutwell  
Director of AI and Creative Technologies  
Meridian Community College

## License

Research use permitted with attribution.
