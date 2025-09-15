### `README.md`

### qaoa_maxcut_invariant_predictor.py

This tool solves a fundamental **combinatorial optimization problem** known as MaxCut. Given a complex network (a graph), it finds the optimal settings (`β*`, `γ*`) for a specific quantum algorithm (QAOA at p=1) that produces the best possible division of the network's nodes into two groups to maximize the connections between them.

It does this **without ever running the quantum algorithm**. It uses a geometric formula that takes simple, classical properties of the network (vertex degrees and triangle counts) and directly predicts the exact, certified energy for any given settings. This allows it to rapidly scan thousands of potential settings to find the optimal one.

#### Who Could Use It and For What:

This tool is for anyone in fields that rely on finding the best configuration in a complex, interconnected system.

*   **Logistics & Operations Research:** A company could model its supply chain as a graph and use this tool to find the optimal way to partition distribution centers or delivery routes to maximize efficiency (i.e., "cut" the fewest essential links).
*   **Financial Portfolio Management:** An analyst could model a portfolio of assets, where connections represent risk correlation. They could use this to find the optimal way to split the portfolio into two sub-portfolios to maximize diversification (i.e., maximize the cuts between correlated assets).
*   **Circuit Design (VLSI):** An engineer designing a computer chip could use this to find the optimal physical layout of millions of components, partitioning them onto different sections of the chip to minimize wire congestion between them.
*   **Drug Discovery & Protein Folding:** Scientists can model molecular interactions as a graph. This tool could help find optimal configurations by maximizing favorable interactions (cuts) between different parts of a molecule.

### RESULTS
```
  === QAOA p=1 MaxCut — Invariant Predictor (v2, ε-certified) ===
Graph: cycle | n=8 | |E|=8 | degree range=[2,2]
Grid:  25x25  |  scan in float64

RESULT (geometry-only; no 2^n objects):
  β*=0.39269908,  γ*=0.78539816
  E*(float grid) = 6.000000000000   (ratio=0.750000)
  E*(certified)  = 6.000000000000   (ratio=0.750000)
  [CERT] precision = 52 dps | certified = True | Δ≤ 1e-12?  (Δ=3.536782072844101e-41)

Time: build motifs=0.0000s | grid eval=0.0024s | certify=0.0010s

[DENSE VALIDATION] at (β*,γ*) from grid:
  E_dense(β*,γ*) = 6.000000000000
  |E_cert - E_dense| = 0.000e+00
  Dense time (n=8): 0.001s
  ✅ Match
```
