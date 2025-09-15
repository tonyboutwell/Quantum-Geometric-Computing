#!/usr/bin/env python3
"""
Quantum Geometric Computing (QGC) Proprietary Software License
Version 1.1 – September 2025

Copyright (c) 2025 Tony Boutwell. All Rights Reserved.

NOTICE TO USER: This is a proprietary software program of Tony Boutwell (the "Author").
You may not use, copy, modify, or distribute this software and its associated 
documentation (the "Software") in any way without first obtaining a separate, 
written license agreement from the Author.

1. LICENSE GRANT
   No license is granted to you under this file. The Author reserves all rights,
   title, and interest in and to the Software. You may only read this license
   text. Any other use, including but not limited to execution, modification,
   copying, or redistribution of the Software, is strictly prohibited.

2. RESTRICTIONS
   Without a separate written license, you may not:
     a. Use the Software for any purpose, whether commercial, non-commercial,
        academic, personal, or military.
     b. Prepare derivative works based upon the Software.
     c. Distribute the Software to any third party.
     d. Reverse-engineer, decompile, or disassemble the Software.

3. NO WARRANTY
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
   FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
   AUTHOR BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN 
   ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN 
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

4. OBTAINING A LICENSE
   To request a license for academic, commercial, or any other use, please
   contact the Author at:
   Tony.Boutwell@QuantumGeometric.com
"""

"""
### qaoa_maxcut_invariant_predictor.py

QAOA p=1 MaxCut — invariant predictor with ε‑certified accuracy.
This tool solves a fundamental combinatorial optimization problem 
known as MaxCut. Given a complex network (a graph), it finds the 
optimal settings (β*, γ*) for a specific quantum algorithm 
(QAOA at p=1) that produces the best possible division of the network's 
nodes into two groups to maximize the connections between them.

Features
----------------
1) Integer‑exact motifs: degrees and per‑edge triangle counts.
2) Arbitrary‑precision evaluation via mpmath for the p=1 closed form.
3) Auto‑precision policy + certificate: recompute at higher precision
   until |ΔE| ≤ --eps.
4) Optional high‑precision grid scanning (--mpgrid), or fast float64 grid + high‑precision
   certification at the chosen (β*,γ*).
5) Deterministic RNG seed for RRG construction; optional CSV dump.

Dependencies
------------
- numpy (required)
- mpmath (optional but recommended for certification)
    pip install mpmath

All hail geometry.
"""

import argparse, math, random, time, csv
import numpy as np

# --------------------------- Optional high precision ---------------------------
# QGC Novelty: For problems with extreme parameters (e.g., complex graphs),
# standard float64 precision is insufficient. We import mpmath to enable
# arbitrary-precision arithmetic, which is essential for certification.
try:
    import mpmath as mp
except Exception:
    mp = None

# --------------------------- Graph builders (integer-exact) --------------------
# This section defines functions to create the classical graph problems that QAOA
# is designed to solve. The graphs are represented by a list of edges.
def build_cycle(n: int):
    """Cycle C_n (each vertex degree 2), edges sorted as (min,max)."""
    edges = []
    for u in range(n):
        v = (u + 1) % n
        edges.append((min(u, v), max(u, v)))
    return n, sorted(set(edges))

def build_rrg(n: int, d: int, seed: int = 1):
    """Random d-regular simple graph via pairing (deterministic for given seed)."""
    rng = random.Random(seed)
    if (n * d) % 2 != 0:
        raise ValueError("n*d must be even for a regular graph.")
    for _ in range(2000):  # retry to avoid self-loops or multi-edges
        stubs = [v for v in range(n) for _ in range(d)]
        rng.shuffle(stubs)
        edges = set()
        ok = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            a, b = (u, v) if u < v else (v, u)
            if a == b or (a, b) in edges:
                ok = False
                break
            edges.add((a, b))
        if ok:
            return n, sorted(edges)
    raise RuntimeError("Failed to construct a simple d-regular graph.")

# --------------------------- Motifs: degrees + triangles -----------------------
# QGC Novelty: This is the core of the "invariant-based" approach for QAOA.
# The final energy expectation does not depend on the 2^n quantum state vector.
# Instead, it depends only on a small number of local, classical graph properties
# or "motifs". These motifs are the geometric invariants for this problem.
def degrees_and_triangles(n: int, E):
    """
    Compute integer‑exact graph invariants ("motifs"):
      - deg[v] for all vertices v
      - λ_uv = |N(u) ∩ N(v)| for each edge (u,v)∈E (i.e., the number of
        triangles that the edge (u,v) is a part of).
    """
    adj = [set() for _ in range(n)]
    for u, v in E:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[v]) for v in range(n)]
    lambdas = {}
    for u, v in E:
        # Intersect the smaller neighborhood set into the larger for efficiency.
        # This computation is exact as it only involves integer counting.
        if len(adj[u]) < len(adj[v]):
            c = sum(1 for w in adj[u] if w in adj[v])
        else:
            c = sum(1 for w in adj[v] if w in adj[u])
        lambdas[(u, v)] = c
    return deg, lambdas  # deg: List[int], lambdas: Dict[(u,v)->int]

# --------------------------- p=1 edge expectations -----------------------------
# This section implements the analytical formula for the p=1 QAOA MaxCut energy.
# This formula, derived by Wang et al. (2018), is a direct "geometric jump" for QAOA,
# allowing us to compute the final energy without simulating the quantum evolution.

def edge_expectation_p1_float(u, v, deg, lambdas, beta, gamma):
    """
    Calculates the expected contribution of a single edge to the MaxCut objective
    using the p=1 analytical formula, evaluated in standard float64 precision.
    The inputs are the graph motifs (invariants), not quantum states.
    """
    du = deg[u] - 1
    dv = deg[v] - 1
    lam = lambdas[(u, v)]
    s4b = math.sin(4.0 * beta)
    sg = math.sin(gamma)
    cg = math.cos(gamma)
    s2b2 = (math.sin(2.0 * beta)) ** 2
    term1 = 0.5
    term2 = 0.25 * s4b * sg * ( (cg ** du) + (cg ** dv) )
    power_term = du + dv - 2 * lam
    term3 = 0.25 * s2b2 * (cg ** power_term) * (1.0 - (math.cos(2.0 * gamma)) ** lam)
    return term1 + term2 - term3

def edge_expectation_p1_mp(u, v, deg, lambdas, beta, gamma):
    """
    The same analytical formula, evaluated using the mpmath library for arbitrary
    precision. This is crucial for certification and for cases with large
    degrees where float64 might lose precision in the cg**du terms.
    """
    du = int(deg[u] - 1)
    dv = int(deg[v] - 1)
    lam = int(lambdas[(u, v)])
    b = mp.mpf(beta)
    g = mp.mpf(gamma)
    s4b = mp.sin(4 * b)
    sg = mp.sin(g)
    cg = mp.cos(g)
    s2b2 = mp.sin(2 * b) ** 2
    term1 = mp.mpf('0.5')
    term2 = mp.mpf('0.25') * s4b * sg * ( (cg ** du) + (cg ** dv) )
    power_term = du + dv - 2 * lam
    term3 = mp.mpf('0.25') * s2b2 * (cg ** power_term) * (1 - (mp.cos(2 * g) ** lam))
    return term1 + term2 - term3

def objective_p1_general_float(n, E, deg, lambdas, beta, gamma):
    """Calculates the total MaxCut objective by summing over all edge expectations in float64."""
    return sum(edge_expectation_p1_float(u, v, deg, lambdas, beta, gamma) for (u, v) in E)

def objective_p1_general_mp(n, E, deg, lambdas, beta, gamma):
    """Calculates the total MaxCut objective in arbitrary precision."""
    tot = mp.mpf('0')
    for (u, v) in E:
        tot += edge_expectation_p1_mp(u, v, deg, lambdas, beta, gamma)
    return tot

# --------------------------- Precision policy & certification ------------------
# This section implements the same philosophy as our hardened Grover tool:
# automatically determine the necessary precision and then certify the result.

def required_dps_qaoa(num_edges: int, deg_max: int, eps: float) -> int:
    """
    Heuristic to determine the required decimal precision for mpmath.
    It considers potential error accumulation from summing many edges and
    precision loss from taking large integer powers of numbers less than 1.
    """
    if eps <= 0:
        eps = 1e-12
    base = math.ceil(math.log10(max(num_edges, 1)) - math.log10(eps)) + 16
    return max(40, base)

def certify_objective(n, E, deg, lambdas, beta, gamma, eps=1e-12, dps_hint=0, max_tries=5):
    """
    Computes the objective function and certifies its value to a tolerance eps.
    It runs the calculation, then re-runs it at a higher precision to check if
    the result is stable. If not, it escalates precision and repeats.
    """
    if mp is None:
        # No mpmath installed — return float64 result without certificate
        E0 = objective_p1_general_float(n, E, deg, lambdas, beta, gamma)
        return float(E0), 16, False, float('nan')

    # Initial digits
    if dps_hint and dps_hint > 0:
        dps = int(dps_hint)
    else:
        dps = required_dps_qaoa(len(E), max(deg) if deg else 0, eps)

    tries = 0
    last_val = None
    last_dps = None
    while True:
        tries += 1
        mp.mp.dps = dps
        val = objective_p1_general_mp(n, E, deg, lambdas, beta, gamma)
        if last_val is None:
            last_val = val
            last_dps = dps
            dps = dps + 12
            if tries >= max_tries:
                return float(val), last_dps, False, float('inf')
            continue
        delta = abs(val - last_val)
        if delta <= eps:
            return float(val), dps, True, float(delta)
        # escalate
        last_val = val
        last_dps = dps
        dps = dps + max(12, int(0.25 * dps))
        if tries >= max_tries:
            return float(val), dps, False, float(delta)

# --------------------------- Dense simulator (small n) -------------------------
# This section provides the "ground truth" validation. It performs a full,
# brute-force state-vector simulation of the QAOA circuit. This is exponentially
# slow and memory-intensive, and is only used to prove that the geometric
# invariant formula is correct for small, verifiable systems.

def psi_plus(n):
    """Creates the initial |+⟩^⊗n state vector."""
    return np.ones(1 << n, dtype=complex) / (2 ** (n / 2.0))

def apply_cost_diagonal(psi, n, E, gamma):
    """Applies the cost Hamiltonian e^(-iγH_C) by adding phases to the state vector."""
    N = psi.size
    phases = np.zeros(N, dtype=float)
    for (i, j) in E:
        bit_i = 1 << i
        bit_j = 1 << j
        for s in range(N):
            zi = 1.0 if ((s & bit_i) == 0) else -1.0
            zj = 1.0 if ((s & bit_j) == 0) else -1.0
            phases[s] += (gamma / 2.0) * (zi * zj)
    psi *= np.exp(1j * phases)

def apply_mixer_all(psi, n, beta):
    """Applies the mixer Hamiltonian e^(-iβH_B) by applying RX(2β) to each qubit."""
    cb = math.cos(beta)
    sb = -1j * math.sin(beta)
    for q in range(n):
        bit = 1 << q
        for base in range(0, psi.size, 2 * bit):
            for k in range(bit):
                i0 = base + k
                i1 = base + bit + k
                a, b = psi[i0], psi[i1]
                psi[i0] = cb * a + sb * b
                psi[i1] = sb * a + cb * b

def dense_expectation(n, E, beta, gamma):
    """Calculates the final MaxCut expectation value from the final state vector."""
    # 1. Evolve the state vector.
    psi = psi_plus(n)
    apply_cost_diagonal(psi, n, E, gamma)
    apply_mixer_all(psi, n, beta)
    probs = np.abs(psi) ** 2
    # 2. Compute the expectation value ⟨ψ_final| H_C |ψ_final⟩.
    total = 0.0
    for (i, j) in E:
        bit_i = 1 << i
        bit_j = 1 << j
        zij = 0.0
        for s in range(1 << n):
            zi = 1.0 if ((s & bit_i) == 0) else -1.0
            zj = 1.0 if ((s & bit_j) == 0) else -1.0
            zij += zi * zj * probs[s]
        total += 0.5 * (1.0 - zij) # Add contribution from this edge.
    return total

# --------------------------- Orchestration -------------------------------------
# This section manages the user interface, parameter scanning, and final reporting.

def grid_points(grid_str: str):
    """Parses the grid search dimensions from a string like '25x25'."""
    Gx, Gy = [int(x) for x in grid_str.split('x')]
    betas = np.linspace(0.0, math.pi / 2.0, Gx)
    gammas = np.linspace(0.0, math.pi, Gy)
    return betas, gammas

def scan_grid_float(n, E, deg, lambdas, betas, gammas):
    """Finds the optimal (β,γ) by scanning the grid using the fast float64 formula."""
    best = (-1.0, 0.0, 0.0)
    for beta in betas:
        for gamma in gammas:
            F = objective_p1_general_float(n, E, deg, lambdas, beta, gamma)
            if F > best[0]:
                best = (F, beta, gamma)
    return best  # (F_best, beta_best, gamma_best)

def scan_grid_mp(n, E, deg, lambdas, betas, gammas, dps_hint=0):
    """Finds the optimal (β,γ) by scanning the grid using the high-precision mpmath formula."""
    if mp is None:
        return scan_grid_float(n, E, deg, lambdas, betas, gammas)
    if dps_hint and dps_hint > 0:
        mp.mp.dps = int(dps_hint)
    else:
        mp.mp.dps = required_dps_qaoa(len(E), max(deg) if deg else 0, 1e-12)
    best_val = mp.mpf('-inf'); best_b = 0.0; best_g = 0.0
    for beta in betas:
        for gamma in gammas:
            val = objective_p1_general_mp(n, E, deg, lambdas, beta, gamma)
            if val > best_val:
                best_val = val; best_b = beta; best_g = gamma
    return (float(best_val), best_b, best_g)

def maybe_write_csv(path, betas, gammas, n, E, deg, lambdas, use_mp=False):
    """Optional utility to dump the entire energy landscape to a CSV file."""
    if path is None or len(path) == 0:
        return
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beta", "gamma", "E"])
        for beta in betas:
            for gamma in gammas:
                if use_mp and mp is not None:
                    val = objective_p1_general_mp(n, E, deg, lambdas, beta, gamma)
                    w.writerow([beta, gamma, float(val)])
                else:
                    val = objective_p1_general_float(n, E, deg, lambdas, beta, gamma)
                    w.writerow([beta, gamma, float(val)])

def main():
    """Parses arguments, orchestrates the computation, and prints the final summary."""
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--graph", type=str, default="cycle", choices=["cycle", "rrg"], help="Graph family")
    ap.add_argument("--n", type=int, default=8, help="Number of vertices")
    ap.add_argument("--d", type=int, default=3, help="Degree for random regular graphs")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed for rrg")
    ap.add_argument("--grid", type=str, default="25x25", help="Grid size for (β,γ) scan, e.g., 25x25")
    ap.add_argument("--validate", type=int, default=1, help="1=run dense validator (tiny n only)")
    ap.add_argument("--mpgrid", type=int, default=0, help="1=evaluate whole grid with mpmath (slower, more precise)")
    ap.add_argument("--eps", type=float, default=1e-12, help="Target absolute ε for certification of E(β*,γ*)")
    ap.add_argument("--prec", type=int, default=0, help="Initial decimal digits for mpmath (0=auto)")
    ap.add_argument("--csv", type=str, default="", help="Optional: path to write a CSV of the grid (beta,gamma,E)")
    args = ap.parse_args()

    # --- Step 1: Build graph and compute classical invariants (motifs) ---
    if args.graph == "cycle":
        n, E = build_cycle(args.n)
    else:
        n, E = build_rrg(args.n, args.d, seed=args.seed)

    t0 = time.time()
    deg, lambdas = degrees_and_triangles(n, E)
    t_motifs = time.time() - t0

    # --- Step 2: Scan parameter space to find optimal angles (β*,γ*) ---
    betas, gammas = grid_points(args.grid)
    if args.csv:
        maybe_write_csv(args.csv, betas, gammas, n, E, deg, lambdas, use_mp=bool(args.mpgrid))

    t0_grid = time.time()
    if args.mpgrid and mp is not None:
        F0, beta0, gamma0 = scan_grid_mp(n, E, deg, lambdas, betas, gammas, dps_hint=args.prec)
    else:
        F0, beta0, gamma0 = scan_grid_float(n, E, deg, lambdas, betas, gammas)
    t_grid = time.time() - t0_grid

    # --- Step 3: Certify the energy at the optimal point with high precision ---
    t0_cert = time.time()
    E_cert, dps_used, ok, delta = certify_objective(
        n, E, deg, lambdas, beta0, gamma0, eps=args.eps, dps_hint=args.prec
    )
    t_cert = time.time() - t0_cert

    # --- Step 4: Print summary report ---
    print("\n=== QAOA p=1 MaxCut — Invariant Predictor ===")
    print(f"Graph: {args.graph} | n={n} | |E|={len(E)} | degree range=[{min(deg)},{max(deg)}]")
    print(f"Grid:  {args.grid}  |  scan in {'mpmath' if (args.mpgrid and mp is not None) else 'float64'}")
    print("\nRESULT (geometry-only; no 2^n objects):")
    print(f"  β*={beta0:.8f},  γ*={gamma0:.8f}")
    print(f"  E*(float grid) = {F0:.12f}   (ratio={F0/len(E):.6f})")
    print(f"  E*(certified)  = {E_cert:.12f}   (ratio={E_cert/len(E):.6f})")
    if mp is None:
        print("  [CERT] mpmath not installed → reported E* is float64 only (no certificate).")
    else:
        print(f"  [CERT] precision = {dps_used} dps | certified = {ok} | Δ≤ {args.eps}?  (Δ={delta})")
    print(f"\nTime: build motifs={t_motifs:.4f}s | grid eval={t_grid:.4f}s | certify={t_cert:.4f}s")

    # --- Step 5: Optional validation against brute-force simulation ---
    if args.validate:
        if n > 14:
            print("\n[DENSE VALIDATION] Skipped: n > 14 too large for dense path.")
        else:
            print("\n[DENSE VALIDATION] at (β*,γ*) from grid:")
            td0 = time.time()
            E_dense = dense_expectation(n, E, beta0, gamma0)
            t_dense = time.time() - td0
            diff = abs(E_cert - E_dense)
            print(f"  E_dense(β*,γ*) = {E_dense:.12f}")
            print(f"  |E_cert - E_dense| = {diff:.3e}")
            print(f"  Dense time (n={n}): {t_dense:.3f}s")
            print("  ✅ Match" if diff < 1e-9 else "  ⚠ Mismatch detected")

if __name__ == "__main__":
    main()
