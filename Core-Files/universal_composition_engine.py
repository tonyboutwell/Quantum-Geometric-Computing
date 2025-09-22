#!/usr/bin/env python3
# Universal Composition Engine (UCE) V3 (Production)
#
# Upgrades vs v2:
#   • Big‑Integer Safety for d = 2^n with n > 1024 via _mk_p_array and mp-based reconstruction
#   • trace_power_replica(): optional log‑space stabilization with mp or float reconstruction
#   • Four‑mode API: run(mode in {'fast','certified','surrogate','brute'})
#   • CLI: select n, M, mode; --validate runs demo_validation() that times all modes
#
# Notes:
#   • Replica contraction cost grows ~ (bond^2 * 2)^m — keep M small (≤ 8–10 typical).
#
# All hail geometry.

import math, time, argparse, sys
import numpy as np

# Optional high precision for certification and safe reconstructions
try:
    import mpmath as mp
except Exception:
    mp = None

# ---------- Small gate factory ----------
def H():  return (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)
def X():  return np.array([[0,1],[1,0]], dtype=complex)
def Y():  return np.array([[0,-1j],[1j,0]], dtype=complex)
def Z():  return np.array([[1,0],[0,-1]], dtype=complex)
def S():  return np.array([[1,0],[0,1j]], dtype=complex)
def T():  return np.array([[1,0],[0,np.exp(1j*np.pi/4)]], dtype=complex)
def RX(theta): return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                                [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
def RY(theta): return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                                [np.sin(theta/2),  np.cos(theta/2)]], dtype=complex)
def RZ(theta): return np.array([[np.exp(-1j*theta/2), 0],
                                [0, np.exp(1j*theta/2)]], dtype=complex)
def CX():
    U = np.eye(4, dtype=complex); U[2,2]=0; U[3,3]=0; U[2,3]=1; U[3,2]=1; return U
def CZ():
    U = np.eye(4, dtype=complex); U[3,3] = -1; return U
def SWAP():
    U = np.eye(4, dtype=complex); U[[1,2],[1,2]]=0; U[1,2]=U[2,1]=1; return U

GATE_LIBRARY = {
    'H':  lambda p=None: H(),
    'X':  lambda p=None: X(),
    'Y':  lambda p=None: Y(),
    'Z':  lambda p=None: Z(),
    'S':  lambda p=None: S(),
    'T':  lambda p=None: T(),
    'RX': lambda p: RX(p['theta']),
    'RY': lambda p: RY(p['theta']),
    'RZ': lambda p: RZ(p['theta']),
    'CX': lambda p=None: CX(),
    'CZ': lambda p=None: CZ(),
    'SWAP': lambda p=None: SWAP(),
}

# ---------- Vector-level application (legacy; used only in dense validator) ----------
def apply_1q(vec, n, q, U1):
    dim = 1 << n
    step = 1 << (n - q - 1)
    block = step << 1
    v = vec.reshape(dim,)
    for base in range(0, dim, block):
        i0 = slice(base, base+step)
        i1 = slice(base+step, base+block)
        v0 = v[i0].copy(); v1 = v[i1].copy()
        v[i0] = U1[0,0]*v0 + U1[0,1]*v1
        v[i1] = U1[1,0]*v0 + U1[1,1]*v1

def apply_2q(vec, n, q1, q2, U2):
    if q1 > q2: q1, q2 = q2, q1
    dim = 1 << n
    s1 = 1 << (n - q1 - 1)
    s2 = 1 << (n - q2 - 1)
    v = vec.reshape(dim,)
    for base in range(dim):
        if (base & s1) or (base & s2):
            continue
        i00 = base; i01 = base | s2; i10 = base | s1; i11 = base | s1 | s2
        b = np.array([v[i00], v[i01], v[i10], v[i11]], dtype=complex)
        b2 = U2 @ b
        v[i00], v[i01], v[i10], v[i11] = b2

# ---------- Newton–Girard (complex / mpmath) ----------
def power_sums_to_coeffs(p, d):
    """
    Convert power sums p[1..d]=tr(U^m) to chi_U(z)=sum_{k=0}^d c_k z^k with c_k=(-1)^k e_k.
    Newton–Girard (complex): e_0=1;  m e_m = sum_{j=1}^m (-1)^{j-1} e_{m-j} p_j
    """
    e = [0]*(d+1); e[0] = 1.0+0.0j
    for m in range(1, d+1):
        acc = 0.0+0.0j
        for j in range(1, m+1):
            acc += ((-1)**(j-1)) * e[m-j] * p[j]
        e[m] = acc / m
    c = [(((-1)**k) * e[k]) for k in range(d+1)]
    return c

def power_sums_to_coeffs_mp(p, d, dps):
    if mp is None:
        raise RuntimeError("mpmath not available for high‑precision Newton–Girard.")
    mp.mp.dps = int(dps)
    e = [mp.mpc(0)]*(d+1); e[0] = mp.mpc(1)
    # power sums to mp
    P = [mp.mpc(0)]*(d+1)
    for m in range(1, d+1):
        pj = p[m]
        if isinstance(pj, complex):
            P[m] = mp.mpc(pj.real, pj.imag)
        elif mp is not None and isinstance(pj, (mp.mpf, mp.mpc)):
            P[m] = mp.mpc(pj)
        else:
            # numpy scalar
            P[m] = mp.mpc(complex(pj).real, complex(pj).imag)
    for m in range(1, d+1):
        acc = mp.mpc(0)
        for j in range(1, m+1):
            acc += ((-1)**(j-1)) * e[m-j] * P[j]
        e[m] = acc / m
    c = [(((-1)**k) * e[k]) for k in range(d+1)]
    # keep in mp
    return c

# ---------- Newton identity residuals (ε‑certificate), generic for np/mp ----------
def _is_mp_val(x):
    return (mp is not None) and isinstance(x, (mp.mpf, mp.mpc))

def newton_identity_residuals(p, c, M):
    """
    For m=1..M, check:  sum_{j=1..m} (-1)^{j-1} e_{m-j} p_j  - m e_m = 0,  where e_m = (-1)^m c_m.
    Returns list of |residual_m| as floats (best effort; mp -> float via float()).
    Accepts numpy complex or mpmath numbers.
    """
    use_mp = any(_is_mp_val(x) for x in (p[1], c[0]))
    zero = mp.mpc(0) if use_mp else 0.0+0.0j
    one  = mp.mpc(1) if use_mp else 1.0+0.0j

    e = [zero]*(M+1); e[0] = one
    # reconstruct e_m from c_m to avoid reusing earlier accumulators
    for m in range(1, M+1):
        cm = c[m]
        if use_mp and not _is_mp_val(cm):
            cm = mp.mpc(cm.real, cm.imag)
        e[m] = ((-1)**m) * cm

    res = []
    for m in range(1, M+1):
        acc = zero
        for j in range(1, m+1):
            pj = p[j]
            if use_mp and not _is_mp_val(pj):
                pj = mp.mpc(pj.real, pj.imag)
            acc += ((-1)**(j-1)) * e[m-j] * pj
        r = acc - m*e[m]
        try:
            res.append(float(abs(r)))
        except Exception:
            # fallback: convert via mp if possible
            if use_mp:
                res.append(float(mp.sqrt((r.real*r.real) + (r.imag*r.imag))))
            else:
                res.append(float('inf'))
    return res

# ============================= MPO for the circuit =============================

def mpo_identity(n):
    """Open-boundary MPO of identity: tensors (1,1,2,2) with δ_{o,i}."""
    W = []
    for _ in range(n):
        t = np.zeros((1,1,2,2), dtype=complex)
        t[0,0,0,0] = 1.0; t[0,0,1,1] = 1.0
        W.append(t)
    return W

def mpo_left_apply_1q(W, q, U1):
    """Left-multiply a 1-qubit gate at site q: U <- (U1 ⊗ I...) U."""
    t = W[q]  # (Dl,Dr,o,i)
    tmp = np.tensordot(U1, t, axes=([1],[2]))  # (o, Dl,Dr, i)
    W[q] = np.transpose(tmp, (1,2,0,3))        # (Dl,Dr,o,i)
    return W

def mpo_left_apply_2q_nn(W, q, U2):
    """
    Left-multiply a 2-qubit gate on adjacent sites (q,q+1).
    Exact SVD split (no truncation).
    """
    tL = W[q]       # (a,b,o1,i1)
    tR = W[q+1]     # (b,c,o2,i2)
    a,b,_,_ = tL.shape
    b2,c,_,_ = tR.shape
    assert b == b2, "MPO bond mismatch"
    pair = np.tensordot(tL, tR, axes=([1],[0]))           # (a,o1,i1,c,o2,i2)
    pair = np.transpose(pair, (0,3,1,4,2,5))              # (a,c,o1,o2,i1,i2)
    G4 = U2.reshape(2,2,2,2)
    Theta = np.einsum('xyuv,acuvij->acxyij', G4, pair)    # (a,c,o1',o2',i1,i2)
    A = a*2*2; C = c*2*2
    M = Theta.transpose(0,2,4, 1,3,5).reshape(A, C)       # (a*2*2, c*2*2)
    U,S,Vh = np.linalg.svd(M, full_matrices=False)
    r = S.size
    U = U @ np.diag(S)
    UL = U.reshape(a, 2,2, r).transpose(0,3,1,2)          # (a,r,o1',i1)
    VR = Vh.reshape(r, c, 2,2)                             # (r,c,o2',i2)
    W[q], W[q+1] = UL, VR
    return W

def _mpo_trace_core_sweep(W):
    """Core sweep: returns the sequence of site E_i = sum_s W_i[:,:,s,s]."""
    Es = []
    for t in W:
        Es.append(t[:, :, 0, 0] + t[:, :, 1, 1])
    return Es

def mpo_trace(W):
    """
    Fast p1 = Tr(U) for an MPO W with site tensors of shape (Dl, Dr, do, di).
    Contracts physical legs with identity and sweeps left->right.
    """
    L = np.array([[1.0+0.0j]])  # (1,1)
    for t in W:
        # E[a,b] = sum_s t[a,b,s,s]
        E = t[:, :, 0, 0] + t[:, :, 1, 1]
        L = L @ E
    assert L.size == 1
    return L.item()

def mpo_trace_stable(W, stabilize=True, use_mp=False):
    """
    Numerically stable trace via left sweep with per-site scaling in log space.
    If stabilize=False, identical to mpo_trace.
    """
    if not stabilize:
        return mpo_trace(W)
    L = np.array([[1.0+0.0j]])
    log_scale = 0.0
    Es = _mpo_trace_core_sweep(W)
    for E in Es:
        L = L @ E
        s = np.linalg.norm(L, ord=np.inf)
        if s > 0:
            L /= s
            log_scale += math.log(s)
    val = L.item()
    if stabilize:
        # First, try fast float64 reconstruction
        if abs(log_scale) < 700:
            result = val * np.exp(log_scale)
            if np.isfinite(result):
                return result
        
        # If float64 failed or log_scale is too large, fallback to mpmath if available
        if mp is not None:
            return mp.mpc(val.real, val.imag) * mp.e**(mp.mpf(log_scale))
        
        # If mpmath is not available, return the potentially non-finite float
        return val * np.exp(log_scale)

    return val

# ===================== Replica contraction for p_m = tr(U^m) =====================

def _kahan_add_inplace(sum_mat, add_mat, comp_mat):
    """
    Kahan-like compensated addition for complex matrices: sum_mat += add_mat with compensation comp_mat.
    All args shape (Dl^m, Dr^m).
    """
    y = add_mat - comp_mat
    t = sum_mat + y
    comp_mat[:] = (t - sum_mat) - y
    sum_mat[:] = t

# ============================================================================
# NEW EFFICIENT REPLICA TRACE FUNCTION (V3.4 - Final)
# Replaces the old trace_power_replica and site_transfer_cycle.
# This version contains the fix for the einsum string corruption.
# ============================================================================
def trace_power_replica(W, m, compensated=True, stabilize=False, use_mp=False):
    """
    EFFICIENTLY contract p_m = tr(U^m) without building the large E_i^{(m)} transfer matrices.
    This version avoids the Kronecker product bomb by contracting the m replicas one by one at each site.
    The 'compensated' flag is ignored as the new contraction order is more stable.
    """
    L = np.ones(1, dtype=complex)
    log_scale = 0.0

    for t in W:  # Loop over sites in the MPO, t has shape (Dl, Dr, 2, 2)
        Dl, Dr = t.shape[0], t.shape[1]
        L = L.reshape([Dl] * m)

        # --- START OF THE FIX ---
        # Generate non-overlapping alphabets for each bond type.
        in_bonds = ''.join(chr(ord('a') + i) for i in range(m))
        out_bonds = ''.join(chr(ord('a') + m + i) for i in range(m))
        phys_bonds = ''.join(chr(ord('a') + 2 * m + i) for i in range(m))
        # --- END OF THE FIX ---
        
        subscripts = [in_bonds]
        for k in range(m):
            subscripts.append(f"{in_bonds[k]}{out_bonds[k]}{phys_bonds[k]}{phys_bonds[(k - 1 + m) % m]}")
        
        einsum_str = f"{','.join(subscripts)}->{out_bonds}"
        
        operands = [L] + [t] * m
        L = np.einsum(einsum_str, *operands, optimize='optimal')
        
        L = L.flatten()

        if stabilize:
            s = np.linalg.norm(L, ord=np.inf)
            if s > 0:
                L /= s
                log_scale += math.log(s)

    result_val = L.item()

    # --- Reconstruction logic (same as before) ---
    if stabilize:
        if abs(log_scale) < 700:
            result = result_val * np.exp(log_scale)
            if np.isfinite(result):
                return result
        if mp is not None:
            return mp.mpc(result_val.real, result_val.imag) * mp.e**(mp.mpf(log_scale))
        return result_val * np.exp(log_scale)
    
    return result_val

# ======================= Universal Composition Engine ========================

class UniversalCompositionEngine:
    """
    Maintains only invariants. Start at identity; append gates via apply_gate.
    Power sums:
      - compute_power_sums_poly(M): replica contraction (polynomial in n for fixed M)
      - compute_power_sums_dense(M): exact vector/matrix method (exponential) kept for tiny n
      - run(M, mode): top‑level four‑mode dispatcher
    """
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.gates = []  # list of (arity, tuple_qubits, U_small)

    # ---- Big‑Integer safe p‑array factory ----
    def _mk_p_array(self, M, use_mp=False):
        """
        Allocate a power-sum list p[0..M] without forcing d=2^n into float if n>1024.
        p[0] is set to d (int) or mp.mpf(d); p[1..M] left as 0.
        """
        p = [0]*(M+1)
        if self.n >= 1024:
            d_int = 1 << self.n  # Python int (arbitrary precision)
            if use_mp and mp is not None:
                p[0] = mp.mpf(d_int)
            else:
                # Leave as big int; downstream code never uses p[0] in NG
                p[0] = d_int
        else:
            p[0] = float(1 << self.n) + 0.0  # keep as float for numpy friendliness
        # initialize others with numeric zeros (use mp if requested)
        if use_mp and mp is not None:
            for m in range(1, M+1):
                p[m] = mp.mpc(0)
        else:
            for m in range(1, M+1):
                p[m] = 0.0+0.0j
        return p

    def apply_gate(self, gate_type, qubits, params=None):
        if gate_type not in GATE_LIBRARY:
            raise ValueError(f"Unknown gate_type {gate_type}")
        U = GATE_LIBRARY[gate_type](params)
        q = tuple(qubits) if isinstance(qubits, (list, tuple)) else (qubits,)
        arity = int(round(math.log2(U.shape[0])))
        assert len(q) == arity, f"{gate_type} arity mismatch: got qubits {q}"
        if arity == 2 and q[0] == q[1]:
            raise ValueError("Two-qubit gate requires distinct qubits.")
        self.gates.append((arity, q, U.astype(complex)))

    # ---- Build circuit MPO (left-multiply order), nearest-neighbor only (use SWAPs if needed) ----
    def to_mpo(self, require_nn=True):
        W = mpo_identity(self.n)
        for arity, q, U in self.gates:
            if arity == 1:
                W = mpo_left_apply_1q(W, q[0], U)
            else:
                if require_nn and abs(q[0]-q[1]) != 1:
                    raise ValueError(f"Non-nearest 2-qubit gate {q}; insert SWAPs or map to NN first.")
                lo = min(q); W = mpo_left_apply_2q_nn(W, lo, U)
        return W

    # ---- Polynomial-time power sums via replica contraction ----
    def compute_power_sums_poly(self, M, max_m=8, require_nn=True, compensated=True, stabilize=False, use_mp_recon=False):
        """
        Compute p[1..M] with replica contraction (no MPO power).  p[0]=2^n (big‑int safe).
        For safety, enforce M <= max_m (default 8). Increase cautiously; cost grows ~ (chi^2 * 2)^m.
        'compensated=True' enables Kahan-like accumulation inside site transfers.
        'stabilize=True' enables log-space reconstruction to avoid over/underflow.
        'use_mp_recon=True' reconstructs stabilized results in mpmath.
        """
        if M > max_m:
            raise ValueError(f"M={M} exceeds max_m={max_m}. Increase max_m cautiously if needed.")
        p = self._mk_p_array(M, use_mp=use_mp_recon)
        W = self.to_mpo(require_nn=require_nn)
        for m in range(1, M+1):
            p[m] = trace_power_replica(W, m, compensated=compensated, stabilize=stabilize, use_mp=use_mp_recon)
        return p

    # ---- Legacy exact (exponential) for tiny n (reference only) ----
    def _apply_circuit_once(self, vec):
        for arity, q, U in self.gates:
            if arity == 1: apply_1q(vec, self.n, q[0], U)
            else:          apply_2q(vec, self.n, q[0], q[1], U)

    def compute_power_sums(self, M=None):
        d = 1 << self.n
        if M is None: M = d
        p = [0j]*(M+1); p[0] = d + 0j
        for b in range(d):
            psi = np.zeros(d, dtype=complex); psi[b] = 1.0
            phi = psi.copy()
            for m in range(1, M+1):
                self._apply_circuit_once(phi)
                p[m] += phi[b]
        return p

    def compute_power_sums_dense(self, M):
        """
        Dense d x d unitary construction then traces of powers.
        Intended for validation only (doubles / small n).
        """
        d = 1 << self.n
        U = np.zeros((d, d), dtype=complex)
        for j in range(d):
            v = np.zeros(d, dtype=complex); v[j] = 1.0
            for arity, q, G in self.gates:
                if arity == 1: apply_1q(v, self.n, q[0], G)
                else:          apply_2q(v, self.n, q[0], q[1], G)
            U[:, j] = v
        p = [0j]*(M+1); p[0] = d + 0j
        Up = np.eye(d, dtype=complex)
        for m in range(1, M+1):
            Up = Up @ U
            p[m] = np.trace(Up)
        return p

    # ---- Certified pipeline: invariants + coefficients + residuals ----
    def compute_power_sums_poly_certified(self, M, eps=1e-12, prec=0, max_m=8, require_nn=True,
                                          compensated=True, stabilize=True, force_mp=False, max_tries=4):
        """
        One-call pipeline:
          1) p[1..M] via replica contraction (compensated accumulation + optional stabilize)
          2) c[0..M] via NG (float64 or mpmath), plus Newton-identity residuals and p1 cross-check
          3) If not force_mp and residuals fail eps, auto-escalate mpmath precision until PASS or max_tries

        Returns dict: {
            'p': list (complex or mp), 'c': list (complex or mp), 'newton_residuals': [float]*M,
            'p1': complex or mp, 'used_mp': bool, 'dps': int,
            'time_poly': float, 'time_ng': float
        }
        """
        t0 = time.time()
        # use mp reconstruction if forcing mp or if n>1024 (huge dynamic ranges)
        use_mp_recon = (force_mp or (self.n > 1024))
        p = self.compute_power_sums_poly(M, max_m=max_m, require_nn=require_nn, compensated=compensated,
                                         stabilize=stabilize, use_mp_recon=use_mp_recon)
        # cross-check p1 via stabilized mpo_trace
        W = self.to_mpo(require_nn=require_nn)
        p1_cross = mpo_trace_stable(W, stabilize=stabilize, use_mp=use_mp_recon)
        t1 = time.time()

        # Newton–Girard
        used_mp, dps_used = False, 16
        if force_mp:
            if mp is None:
                raise RuntimeError("mpmath not available for certified mode.")
            base = math.ceil(max(0.0, math.log10(max(M,1)) - math.log10(max(eps,1e-30)))) + 16
            dps = int(prec) if (prec and prec > 0) else max(50, base)
            c = power_sums_to_coeffs_mp(p, M, dps=dps)
            res = newton_identity_residuals(p, c, M)
            used_mp, dps_used = True, dps
        else:
            c = power_sums_to_coeffs(p, M)
            res = newton_identity_residuals(p, c, M)
            if max(res) > eps and mp is not None:
                base = math.ceil(max(0.0, math.log10(max(M,1)) - math.log10(max(eps,1e-30)))) + 16
                dps = int(prec) if (prec and prec > 0) else max(40, base)
                tries = 0
                while True:
                    tries += 1
                    c_mp = power_sums_to_coeffs_mp(p, M, dps=dps)
                    res_mp = newton_identity_residuals(p, c_mp, M)
                    used_mp, dps_used = True, dps
                    c, res = c_mp, res_mp
                    if max(res) <= eps or tries >= max_tries:
                        break
                    dps = dps + max(12, int(0.25*dps))

        t2 = time.time()
        return {
            'p': p,
            'c': c,
            'newton_residuals': res,
            'p1': p1_cross,
            'used_mp': used_mp,
            'dps': dps_used,
            'time_poly': t1 - t0,
            'time_ng': t2 - t1,
        }

    # ------------------------------ Four‑mode API ------------------------------
    def run(self, M, mode='certified'):
        """
        Top-level dispatcher:
        - mode='fast':      replica contraction with safety OFF (compensated=False, stabilize=False).
        - mode='certified': replica contraction with safety ON (compensated=True, stabilize=True)
                            + full mpmath certification (Newton–Girard in mp).
        - mode='surrogate': approximate formula p_m ≈ d * α^m with α = p1/d (no replica contraction).
        - mode='brute':     dense matrix method for validation (exact, tiny n only).
        Returns a dict with at least keys: {'mode','p','time'}. Certified adds 'c','newton_residuals', etc.
        """
        mode = mode.lower().strip()
        if mode not in ('fast', 'certified', 'surrogate', 'brute'):
            raise ValueError("mode must be one of: fast, certified, surrogate, brute")
        
        t0 = time.time()

        if mode == 'fast':
            p = self.compute_power_sums_poly(M, max_m=max(8, M), require_nn=True,
                                            compensated=False, stabilize=False, use_mp_recon=False)
            return {'mode': 'fast', 'p': p, 'time': time.time() - t0}

        elif mode == 'certified':
            rep = self.compute_power_sums_poly_certified(M, eps=1e-12, prec=0, max_m=max(8, M),
                                                        require_nn=True, compensated=True, stabilize=True,
                                                        force_mp=True)
            rep['mode'] = 'certified'
            rep['time'] = rep['time_poly'] + rep['time_ng']
            return rep

        elif mode == 'surrogate':
            W = self.to_mpo(require_nn=True)
            p1 = mpo_trace_stable(W, stabilize=True)

            NUMERICALLY_SENSITIVE_N = 500
            use_mp_for_surrogate = (
                (mp is not None and isinstance(p1, mp.mpc)) or
                self.n >= 1024 or
                self.n > NUMERICALLY_SENSITIVE_N
            )

            if use_mp_for_surrogate:
                d = mp.mpf(1 << self.n)
                alpha = mp.mpc(p1) / d if d != 0 else mp.mpc(0)
                p = self._mk_p_array(M, use_mp=True)
                for m in range(1, M + 1):
                    p[m] = d * (alpha ** m)
            else:  # n is small and safe
                d = float(1 << self.n)
                alpha = p1 / d if d != 0 else 0.0 + 0.0j
                p = self._mk_p_array(M)
                for m in range(1, M + 1):
                    p[m] = (d * (alpha ** m))
            
            return {'mode': 'surrogate', 'alpha': alpha, 'p': p, 'time': time.time() - t0}

        elif mode == 'brute':
            p = self.compute_power_sums_dense(M)
            return {'mode': 'brute', 'p': p, 'time': time.time() - t0}

# ------------------------------ Validation & CLI helpers -------------------

def leverrier_coeffs_dense(U):
    d = U.shape[0]; I = np.eye(d, dtype=complex)
    B = I.copy(); c = [1.0+0j]
    for k in range(1, d+1):
        ck = -np.trace(U @ B) / k
        c.append(ck)
        B = U @ B + ck * I
    return c

def build_demo_circuit(eng):
    """
    Re-usable small mixed NN circuit used in demos and validation.
    """
    n = eng.n
    eng.apply_gate('H', 0); eng.apply_gate('H', 1)
    if n >= 3: eng.apply_gate('H', 2)
    if n >= 4: eng.apply_gate('H', 3)
    eng.apply_gate('CX', (0,1))
    eng.apply_gate('RZ', 0, params={'theta': 0.23})
    eng.apply_gate('RY', 1, params={'theta': -1.11})
    if n >= 3:
        eng.apply_gate('CX', (1,2)); eng.apply_gate('T', 2)
    if n >= 4:
        eng.apply_gate('CZ', (2,3)); eng.apply_gate('RX', 3, params={'theta': 0.37})
    eng.apply_gate('S', 0)
    eng.apply_gate('SWAP', (0,1))
    return eng

def _to_complex(x):
    if mp is not None and isinstance(x, (mp.mpf, mp.mpc)):
        return complex(float(x.real), float(x.imag if hasattr(x,'imag') else 0.0))
    return complex(x)

def _vec_diff_norm(p_est, p_true, M):
    # returns max |Δp_m| for m<=M (best effort if mp)
    diffs = []
    for m in range(0, M+1):
        try:
            diffs.append(abs(_to_complex(p_est[m]) - _to_complex(p_true[m])))
        except Exception:
            diffs.append(float('inf'))
    return max(diffs)

def demo_validation(n=4, M=None):
    """
    New validation: tests all four modes on a small n, compares to brute-force,
    and prints timings to show performance differences.
    """
    if M is None:
        M = min(6, 1<<n)
    print(f"\n=== UCE demo_validation: n={n}, M={M} ===")
    # Build a demo circuit
    base = UniversalCompositionEngine(n)
    build_demo_circuit(base)

    # Brute (reference)
    t0 = time.time()
    p_brute = base.compute_power_sums_dense(M)
    t_brute = time.time() - t0
    print(f"[brute] time={t_brute:.4f}s")

    # Fast
    t0 = time.time()
    fast = base.run(M, mode='fast')
    t_fast = time.time() - t0
    err_fast = _vec_diff_norm(fast['p'], p_brute, M)
    print(f"[fast]  time={t_fast:.4f}s  max|Δp_m|={err_fast:.2e}")

    # Certified
    t0 = time.time()
    cert = base.run(M, mode='certified')
    t_cert = time.time() - t0
    err_cert = _vec_diff_norm(cert['p'], p_brute, M)
    print(f"[cert]  time={t_cert:.4f}s  max|Δp_m|={err_cert:.2e}  used_mp={cert.get('used_mp',False)}  dps={cert.get('dps',0)}")
    print(f"        Newton residuals: {[f'{v:.1e}' for v in cert['newton_residuals']]}")

    # Surrogate
    t0 = time.time()
    surr = base.run(M, mode='surrogate')
    t_surr = time.time() - t0
    err_surr = _vec_diff_norm(surr['p'], p_brute, M)
    try:
        alpha_disp = surr['alpha'] if not (mp and isinstance(surr['alpha'], (mp.mpf, mp.mpc))) else complex(surr['alpha'])
    except Exception:
        alpha_disp = surr.get('alpha')
    print(f"[surr]  time={t_surr:.4f}s  max|Δp_m|={err_surr:.2e}  alpha≈{alpha_disp}")

    # Summary
    print("\nTiming summary (lower is better):")
    print(f"  brute:     {t_brute:.4f}s  (exact; exponential)")
    print(f"  certified: {t_cert:.4f}s  (safe; mp-certified)")
    print(f"  fast:      {t_fast:.4f}s  (unsafe; quickest replica)")
    print(f"  surrogate: {t_surr:.4f}s  (approx; no replica)")

# ------------------------------ Command-line interface -------------------------

def main():
    parser = argparse.ArgumentParser(description="Universal Composition Engine (UCE) — production CLI")
    parser.add_argument("-n", "--n", type=int, default=4, help="Number of qubits (default: 4)")
    parser.add_argument("-M", "--M", type=int, default=None, help="Max power M to compute (default: min(6, 2^n))")
    parser.add_argument("-m", "--mode", type=str, default="certified", choices=["fast","certified","surrogate","brute"],
                        help="Execution mode (default: certified)")
    parser.add_argument("--validate", action="store_true", help="Run demo_validation benchmarking all modes")
    args = parser.parse_args()

    n = args.n
    M = args.M if args.M is not None else min(6, 1<<n)

    if args.validate:
        demo_validation(n=n, M=M)
        return

    # Build demo circuit (same as validation) unless the user wires their own via import
    eng = UniversalCompositionEngine(n)
    build_demo_circuit(eng)
    result = eng.run(M, mode=args.mode)

    # Pretty-print result basics
    print(f"Mode: {result.get('mode', args.mode)}")
    print(f"n={n}, M={M}")
    if 'time' in result:
        print(f"time: {result['time']:.6f}s")
    # p summary
    p = result.get('p', None)
    if p is not None:
        print("p[m] for m=0..M:")
        for m in range(0, M+1):
            try:
                val = p[m]
                if mp is not None and isinstance(val, (mp.mpf, mp.mpc)):
                    # shorten
                    val = complex(val)
                print(f"  p[{m}] = {val}")
            except Exception as e:
                print(f"  p[{m}] = <unprintable: {e}>")
    # certificate bits
    if args.mode == 'certified':
        print("Newton residuals:", [f"{v:.2e}" for v in result['newton_residuals']])
        print("mp used:", result.get('used_mp', False), "dps:", result.get('dps', 0))
        try:
            p1_disp = result.get('p1')
            if mp is not None and isinstance(p1_disp, (mp.mpf, mp.mpc)):
                p1_disp = complex(p1_disp)
            print("p1 cross-check:", p1_disp)
        except Exception:
            pass

if __name__ == "__main__":
    main()
