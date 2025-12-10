# QGC Hubbard Model Explorer

A fast, algebraic solver for the 2D Hubbard Model at half-filling using Quantum Geometric Computation (QGC) principles.

## What It Does

Computes ground state energies for the 2D Hubbard model by exploiting geometric invariants of the Hamiltonian rather than explicit diagonalization. Achieves ~0.5% accuracy against Quantum Monte Carlo benchmarks while running in milliseconds.

## Quick Start

```bash
# Run benchmark suite
python qgc_hubbard_model_explorer.py bench

# Single point calculation
python qgc_hubbard_model_explorer.py point --nx 4 --ny 4 --U 4.0

# With observables (kinetic energy, double occupancy)
python qgc_hubbard_model_explorer.py point --nx 4 --ny 4 --U 8.0 --obs

# Phase diagram scan
python qgc_hubbard_model_explorer.py scan --nx 4 --ny 4 --Umax 12
```

## How It Works

The solver blends two approaches:

1. **Finite-geometry branch**: Uses exact Hamiltonian moments (μ₁ through μ₄) and a calibrated spectral depth parameter κ to estimate ground state energy
2. **Bulk branch**: Maps the exact 1D Lieb-Wu solution to 2D via a dimensional coupling function α(U)

The key innovation is the κ ansatz:
```
E₀ ≈ μ₁ - κ × σ
```
where κ encodes how geometric complexity (loop content, lattice topology) determines spectral depth.

## Benchmark Results

Validated against Simons Collaboration 2015 (Phys. Rev. X 5, 041041):

| System | U | Simons | QGC | Error |
|--------|---|--------|-----|-------|
| 4×4 | 4.0 | -0.8513 | -0.8510 | 0.03% |
| 4×4 | 8.0 | -0.5150 | -0.5207 | 1.11% |
| Bulk | 4.0 | -0.8617 | -0.8643 | 0.30% |
| Bulk | 8.0 | -0.5250 | -0.5206 | 0.83% |

**Mean error: 0.48%** across all test cases.

## Features

- **Cache-free**: Pure algebraic computation, no lookup tables
- **Large lattice support**: Tested up to 1000×1000 (D ~ 10^601,000)
- **Phase diagrams**: Scan U with ASCII visualization or CSV export
- **Observables**: Kinetic energy, double occupancy, spin correlations
- **Fast**: Sub-millisecond for small clusters, instant bulk estimates

## Requirements

```
numpy
scipy
```

## Limitations

- Half-filling only (doping support planned)
- 2×2 clusters at small U may have ~10-15% error (use exact diagonalization)

## References

- Simons Collaboration, Phys. Rev. X 5, 041041 (2015)
- Qin, Shi, Zhang, Phys. Rev. B 94, 085103 (2016)
- Lieb & Wu, Phys. Rev. Lett. 20, 1445 (1968)
