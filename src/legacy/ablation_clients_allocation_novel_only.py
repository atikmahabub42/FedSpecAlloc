#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 ablation_clients_allocation_2p4GHz.py
 ABLATION: number of federated clients x data partition for the NOVEL
 forecast-driven allocation (BEGA-LS, FSF-BEGA, GRAIL) + the best SOTA
 allocator (Greedy+WMMSE) as reference, 2.4 GHz ISM, per scenario and
 combined, h = 1, 4, 8, per-SNR bins 6-8 ... 28-30 dB and "> 30".
 Same simulated secondary network (seeded topology pool), measured PU
 future, metrics and utility as novel_allocation_2p4GHz.py.

 WHAT CHANGES WITH K (clients) AND THE PARTITION (iid / noniid)
   * the forecaster: the best novel forecaster (validation Brier) trained by
     FedAvg with K clients on that partition (ablation_clients_forecasting
     hand-off); every method uses it;
   * GRAIL's policy: FedAvg over K clients holding the SAME partition of the
     training slots (noniid: one scenario, one contiguous time block per
     client; iid: random equal shares of all slots). Each client takes
     fl_steps_per_round / K local steps per round (K = 4 -> 50 = main run),
     so the computation per round does not depend on K.
   * FSF fusion stays per scenario (site calibration, never communicated).
   reference part: centralised forecaster + centralised GRAIL.

 RUNNING: no arguments = all parts + merge; --task i (0..5 = K 4..32, 6 =
 centralised reference) = one part (SLURM array); --merge = CSVs only.
 OUTPUT: alloc_overall.csv / _agg.csv, alloc_per_snr.csv, alloc_complexity.csv,
         fl_communication.csv / _agg.csv, client_partitions.csv
 (+ columns partition, n_clients, forecaster_config, family)
================================================================================
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import math
import os
import random
import sys
import time
import zlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("PyTorch is required:  pip install torch")

# ==========================================================================
#                                  CONFIG
# ==========================================================================
# ---- paths -------------------------------------------------------------
# Defaults follow the repository layout: data/ and results/ next to src/.
# Override with the environment variables SPECTRUM_DATA, SPECTRUM_RESULTS
# (and SPECTRUM_PROJECT for relative paths) or with the command-line options.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_NARVAL = os.environ.get("SPECTRUM_PROJECT", _REPO)   # project root (name kept from the cluster runs)
_ON_NARVAL = False
_RES = os.environ.get("SPECTRUM_RESULTS", os.path.join(_REPO, "results"))

CONFIG: Dict = {
    "forecast_dir": os.path.join(_RES, "Ablation_Clients", "Forecasting"),
    "out_dir": os.path.join(_RES, "Ablation_Clients", "Allocation"),
    "methods": ["bega_ls", "fsf_bega", "grail", "greedy_wmmse"],
    # ---- ABLATION grid (must match the forecasting ablation) -----------------
    "ab_clients": [4, 8, 12, 16, 24, 32],
    "ab_partitions": ["iid", "noniid"],
    "seeds": [0, 1, 2],
    "seed": 20261019,
    "horizons": [1, 4, 8],
    # ---- secondary network: IDENTICAL to the SOTA run -------------------------
    "su_links": 48, "su_area_m": 15.0, "su_link_dist_m": (2.0, 8.0),
    "su_pathloss_exp": 3.0, "su_shadow_db": 4.0, "su_snr_ref_db": 40.0,
    "collision_penalty": 2.0, "tx_power_min": 1e-3,
    "topo_pool_test": 64, "topo_block": 50,
    "eta_db": 6.0, "wmmse_iters": 30,
    # ---- novel methods ---------------------------------------------------------
    "ls_passes": 2,
    "grail_samples": 8, "grail_batch": 32, "gnn_hidden": 64,
    "train_steps": 800, "alloc_batch": 32, "alloc_lr": 1e-3,
    "fl_rounds": 16, "fl_local_steps": 50,
    "fl_steps_per_round": 200, "fl_min_local_steps": 4,      # local steps per client = max(4, 200 / K)
    # ---- profiling -------------------------------------------------------------
    "prof_iters": 30, "prof_warmup": 3,
    "device": "auto",
    "verbose": True,
}

QUICK = {"train_steps": 4, "fl_rounds": 1, "fl_local_steps": 2, "fl_steps_per_round": 8, "fl_min_local_steps": 1, "ab_clients": [4, 8], "seeds": [0], "horizons": [1],
         "grail_samples": 2, "prof_iters": 3, "prof_warmup": 1, "ls_passes": 1}

_LOG_FH = None


def log(m: str = "") -> None:
    if CONFIG.get("verbose", True):
        print(m, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(m + "\n")
        _LOG_FH.flush()


def set_seed(sd: int) -> None:
    random.seed(sd)
    np.random.seed(sd % (2 ** 32 - 1))
    torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_seed(cfg: Dict, *parts) -> int:
    key = "|".join(str(p) for p in parts)
    return int((int(cfg["seed"]) + zlib.crc32(key.encode())) % (2 ** 31 - 1))


def get_device(cfg: Dict) -> torch.device:
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


# ==========================================================================
#          INPUT: the stage2/ hand-off of spectrum_occupancy_forecasting
# ==========================================================================
def _search_bases() -> List[str]:
    b = [os.getcwd(), _RES, _NARVAL, os.path.dirname(_NARVAL),
         os.path.dirname(os.path.abspath(__file__))]
    return list(dict.fromkeys(b))


def find_stage2(forecast_dir: str):
    """The forecasting out_dir (or its stage2/ folder itself), given as an
    absolute path or relative to: the current folder, the results folder,
    the project folder, the code folder. -> (stage2 dir or None, places tried)."""
    p = os.path.expanduser(forecast_dir)
    roots = [p] if os.path.isabs(p) else [os.path.join(b, p) for b in _search_bases()]
    tried = []
    for r in roots:
        for d in (os.path.join(r, "stage2"), r):
            d = os.path.normpath(d)
            if d in tried:
                continue
            tried.append(d)
            if os.path.isfile(os.path.join(d, "manifest.json")):
                return d, tried
    return None, tried


def resolve_out(out_dir: str) -> str:
    p = os.path.expanduser(out_dir)
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(os.getcwd(), p))


class Bundle:
    """Measured labels + forecasts written by script 1. Nothing is generated."""

    def __init__(self, cfg: Dict):
        d, tried = find_stage2(cfg["forecast_dir"])
        if d is None:
            log("\nNO FORECAST HAND-OFF FOUND. Looked for stage2/manifest.json in:\n  "
                + "\n  ".join(tried) +
                "\nRun spectrum_occupancy_forecasting_2p4GHz.py first (it writes stage2/),\n"
                "or set CONFIG['forecast_dir'] / --forecast-dir to its out_dir.\n"
                "Nothing was run and no data was invented.")
            sys.exit(1)
        mp = os.path.join(d, "manifest.json")
        self.dir = d
        self.man = json.load(open(mp))
        z = np.load(os.path.join(d, "labels.npz"))
        self.names = json.loads(str(z["names"]))
        self.C = int(z["C"])
        self.edges = np.asarray(z["edges_snr"], dtype=np.float64)
        self.eta = float(z["eta_db"])
        self.H = int(z["horizon"])
        self.D = {s: {k: np.asarray(z[f"{s}__{k}"], dtype=np.float32)
                      for k in ("y_te", "E_te", "y_tr1", "E_tr1", "inr_busy", "inr_idle")}
                  for s in self.names}
        self.labels = self.man.get("model_labels", {})
        log(f"  hand-off: {d}\n  scenarios {self.names}, C = {self.C}, H = {self.H}, "
            f"best forecaster by mode {self.man['best_by_mode']}")

    def label(self, key):
        return self.labels.get(key, key)

    def preds(self, model: str, mode: str, seed: int) -> Optional[Dict]:
        """{scenario: {'te': [N,H,C], 'tr1': [N,C]}} or None."""
        fn = os.path.join(self.dir, f"preds__{model}__{mode}__s{seed}.npz")
        if not os.path.isfile(fn):
            c = sorted(glob.glob(os.path.join(self.man.get("cache_dir", ""),
                                              f"{model}__{mode}__s{seed}__*.npz")),
                       key=os.path.getmtime)
            if not c:
                return None
            fn = c[-1]
        z = np.load(fn, allow_pickle=True)
        out = {}
        for s in self.names:
            out[s] = {"te": np.asarray(z[f"{s}__te"], dtype=np.float32)}
            if f"{s}__tr1" in z:
                out[s]["tr1"] = np.asarray(z[f"{s}__tr1"], dtype=np.float32)
        return out


# ==========================================================================
#                               SMALL HELPERS
# ==========================================================================
def _bin_mask(E, lo, hi, last, eta):
    lower = (E > lo) if abs(lo - eta) < 1e-9 else (E >= lo)
    return lower & ((E <= hi) if last else (E < hi))


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    log(f"  -> {os.path.basename(path)} ({len(rows)} rows)")


def aggregate(rows: List[Dict], keys: List[str]) -> List[Dict]:
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    out = []
    for g, rs in groups.items():
        o = dict(zip(keys, g))
        o["n_seeds"] = len(rs)
        for k in rs[0]:
            if k in keys or k == "seed":
                continue
            v = [r[k] for r in rs if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
            if len(v) == len(rs):
                o[k] = float(np.nanmean(v)) if not all(np.isnan(v)) else float("nan")
                o[k + "_std"] = float(np.nanstd(v)) if not all(np.isnan(v)) else float("nan")
            elif k not in o:
                o[k] = rs[0][k]
        out.append(o)
    return out


def model_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.state_dict().values())


def _avg_states(states, weights):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = {}
    for k in states[0]:
        if states[0][k].dtype.is_floating_point:
            out[k] = sum(float(wi) * s[k] for wi, s in zip(w, states))
        else:
            out[k] = states[0][k].clone()
    return out


def count_flops(fn, *args) -> float:
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                fn(*args)
        return float(fc.get_total_flops())
    except Exception:
        return float("nan")


def _time_call(fn, n_iter, n_warm, sync=None) -> float:
    for _ in range(n_warm):
        fn()
    if sync:
        sync()
    ts = []
    for _ in range(n_iter):
        t = time.perf_counter(); fn()
        if sync:
            sync()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


# ==========================================================================
#              SECONDARY NETWORK (simulated) + MEASURED PU CHANNEL
# ==========================================================================
#  K SU links (tx -> rx) in an indoor area; the spectrum sensor that recorded
#  the data sits at the centre and stands for the PU receiver.
#    G[j, k]  gain tx j -> rx k,  Gsp[k]  gain SU tx k -> PU receiver
#  normalised so the noise power of one sub-channel is 1. This network is
#  SIMULATED (seeded); everything about the primary users is MEASURED:
#    y[c]    true occupancy of sub-channel c in the slot
#    inr[c]  10^(excess/10) - 1 >= 0 : PU power over noise at the receivers
#  A decision may use only the forecast phat[c] and the expected INR
#    inr_hat[c] = phat*E[inr | busy] + (1 - phat)*E[inr | idle]
#  (conditional means from the TRAINING span).
# ==========================================================================
def gen_topologies(rng: np.random.Generator, n: int, cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    K, L = cfg["su_links"], cfg["su_area_m"]
    dmin, dmax = cfg["su_link_dist_m"]
    G = np.empty((n, K, K), np.float32)
    Gsp = np.empty((n, K), np.float32)
    pu = np.array([L / 2, L / 2])
    pl = lambda D, shape: (cfg["su_snr_ref_db"] - 10 * cfg["su_pathloss_exp"] * np.log10(D)
                           + rng.normal(0, cfg["su_shadow_db"], shape))
    for i in range(n):
        tx = rng.uniform(0, L, (K, 2))
        d = rng.uniform(dmin, dmax, K)
        th = rng.uniform(0, 2 * np.pi, K)
        rx = np.clip(tx + d[:, None] * np.stack([np.cos(th), np.sin(th)], 1), 0, L)
        D = np.maximum(np.linalg.norm(tx[:, None, :] - rx[None, :, :], axis=-1), 0.5)
        G[i] = np.power(10.0, pl(D, (K, K)) / 10.0)
        Dp = np.maximum(np.linalg.norm(tx - pu[None], axis=-1), 0.5)
        Gsp[i] = np.power(10.0, pl(Dp, (K,)) / 10.0)
    return G, Gsp


def inr_from_excess(e):
    return torch.clamp(torch.pow(10.0, e / 10.0) - 1.0, min=0.0)


class SlotBatch:
    """B slots: forecast, truth, measured excess, SU topology, on device."""

    def __init__(self, phat, y, e, G, Gsp, inr_busy, inr_idle, dev, ppers=None):
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
        self.phat, self.y, self.e, self.G, self.Gsp = t(phat), t(y), t(e), t(G), t(Gsp)
        self.ppers = t(ppers) if ppers is not None else None
        self.inr = inr_from_excess(self.e)
        ib, ii = t(inr_busy), t(inr_idle)
        if ib.dim() == 1:
            ib, ii = ib[None].expand_as(self.phat), ii[None].expand_as(self.phat)
        self.inr_busy, self.inr_idle = ib, ii
        self.inr_hat = self.phat * ib + (1 - self.phat) * ii
        self.B, self.C = self.phat.shape
        self.K = self.G.shape[1]
        self.dev = dev

    def oracle(self):
        o = copy.copy(self)
        o.phat, o.inr_hat = self.y, self.inr
        return o

    def gdirect_db(self):
        return 10 * torch.log10(torch.diagonal(self.G, dim1=1, dim2=2) + 1e-12)

    def node_feats(self):
        B, K, C = self.B, self.K, self.C
        return torch.cat([self.phat[:, None, :].expand(B, K, C),
                          (torch.log1p(self.inr_hat) / 3.0)[:, None, :].expand(B, K, C),
                          (self.gdirect_db() / 40.0)[..., None]], -1)       # [B, K, 2C+1]

    def edge_feats(self):
        return 10 * torch.log10(self.G + 1e-12) / 40.0                      # [B, K, K]


def sb_repeat(sb: SlotBatch, n: int) -> SlotBatch:
    o = copy.copy(sb)
    for k in ("phat", "y", "e", "G", "Gsp", "inr", "inr_hat", "inr_busy", "inr_idle"):
        setattr(o, k, getattr(sb, k).repeat_interleave(n, 0))
    if sb.ppers is not None:
        o.ppers = sb.ppers.repeat_interleave(n, 0)
    o.B = sb.B * n
    return o


def su_side(sb: SlotBatch, ch, pw, cfg):
    """SU rates under the TRUE measured PU interference. ch in [0, C], C = idle."""
    C, K = sb.C, sb.K
    tx = (ch < C) & (pw > cfg["tx_power_min"])
    chc = ch.clamp(max=C - 1)
    A = F.one_hot(chc, C).float() * tx[..., None].float()
    same = A @ A.transpose(1, 2) * (1 - torch.eye(K, device=sb.dev))
    p = pw * tx.float()
    I = (same * p[:, :, None] * sb.G).sum(1)
    sinr = p * torch.diagonal(sb.G, dim1=1, dim2=2) / (1.0 + torch.gather(sb.inr, 1, chc) + I)
    rate = torch.log2(1 + sinr) * tx.float()
    return tx, chc, A, p, sinr, rate


def expected_utility(sb: SlotBatch, ch, pw, cfg) -> torch.Tensor:
    """Utility predicted from the FORECAST only (used to choose among samples)."""
    C, K = sb.C, sb.K
    tx = (ch < C) & (pw > cfg["tx_power_min"])
    chc = ch.clamp(max=C - 1)
    A = F.one_hot(chc, C).float() * tx[..., None].float()
    same = A @ A.transpose(1, 2) * (1 - torch.eye(K, device=sb.dev))
    p = pw * tx.float()
    I = (same * p[:, :, None] * sb.G).sum(1)
    q = torch.gather(sb.phat, 1, chc)
    r = torch.log2(1 + p * torch.diagonal(sb.G, dim1=1, dim2=2)
                   / (1 + torch.gather(sb.inr_hat, 1, chc) + I))
    return (((1 - q) * r - cfg["collision_penalty"] * q) * tx.float()).sum(1)


def _rates(S, N, I):
    return torch.log2(1 + S / (N + I))


@torch.no_grad()
def greedy_assign(sb: SlotBatch, cfg: Dict, oracle: bool = False, allowed=None, lam=None):
    """Links in decreasing direct gain; each takes the sub-channel with the
    largest positive MARGINAL expected utility (own expected rate, minus the
    rate it removes from links already there, minus the expected collision
    charge), or stays idle. Reuse allowed. Forecast only, unless oracle=True.
    allowed: optional [B, C] bool mask of sub-channels that may be used."""
    s = sb.oracle() if oracle else sb
    B, K, C, dev = s.B, s.K, s.C, s.dev
    lam = cfg["collision_penalty"] if lam is None else lam
    Gd = torch.diagonal(s.G, dim1=1, dim2=2)
    order = torch.argsort(-Gd, dim=1)
    ch = torch.full((B, K), C, dtype=torch.long, device=dev)
    A = torch.zeros(B, K, C, device=dev)
    Icur = torch.zeros(B, K, device=dev); Ncur = torch.ones(B, K, device=dev)
    qcur = torch.zeros(B, K, device=dev)
    ar = torch.arange(B, device=dev)
    for i in range(K):
        k = order[:, i]
        Sk = Gd[ar, k]
        Ikc = torch.einsum("bjc,bj->bc", A, s.G[ar, :, k])
        own = (1 - s.phat) * _rates(Sk[:, None], 1 + s.inr_hat, Ikc) - lam * s.phat
        g_out = s.G[ar, k, :]
        assigned = A.sum(-1) > 0
        loss_j = qcur * (_rates(Gd, Ncur, Icur) - _rates(Gd, Ncur, Icur + g_out)) * assigned
        val = own - torch.einsum("bjc,bj->bc", A, loss_j)
        if allowed is not None:
            val = val.masked_fill(~allowed, -1e9)
        best, c_star = val.max(1)
        ok = best > 0
        if ok.any():
            b = ar[ok]; kk = k[ok]; cc = c_star[ok]
            ch[b, kk] = cc
            Icur[b] = Icur[b] + A[b, :, cc] * g_out[ok]
            A[b, kk, cc] = 1.0
            Icur[b, kk] = Ikc[ok, cc]
            Ncur[b, kk] = 1 + s.inr_hat[b, cc]
            qcur[b, kk] = 1 - s.phat[b, cc]
    return ch, (ch < C).float()


@torch.no_grad()
def wmmse_power(sb: SlotBatch, ch, cfg: Dict, noise=None, iters=None):
    """WMMSE power control (Shi, Razaviyayn, Luo, He, IEEE TSP 2011) on the
    co-channel interference graph of a fixed sub-channel assignment."""
    iters = iters or cfg["wmmse_iters"]
    C = sb.C
    tx = (ch < C).float()
    chc = ch.clamp(max=C - 1)
    A = F.one_hot(chc, C).float() * tx[..., None]
    h2 = sb.G * (A @ A.transpose(1, 2))
    hkk = torch.sqrt(torch.diagonal(sb.G, dim1=1, dim2=2))
    if noise is None:
        noise = 1.0 + torch.gather(sb.inr_hat, 1, chc)
    v = tx.clone()
    for _ in range(iters):
        u = hkk * v / (noise + torch.einsum("bjk,bj->bk", h2, v ** 2))
        w = 1.0 / (1.0 - u * hkk * v).clamp(min=1e-6)
        den = torch.einsum("bkj,bj->bk", h2, w * u ** 2)
        v = (w * u * hkk / den.clamp(min=1e-12)).clamp(0, 1) * tx
    return v ** 2


# ==========================================================================
#                   TRAINING SLOTS + TEST BLOCKS
# ==========================================================================
class SlotSource:
    """Training slots of one scenario: forecast (h=1) + measured truth.
    Topologies are drawn fresh from the training RNG (never the test pool)."""

    def __init__(self, name, phat, y, e, inr_busy, inr_idle, cfg, dev):
        self.name, self.phat, self.y, self.e = name, phat, y, e
        self.inr_busy, self.inr_idle, self.cfg, self.dev = inr_busy, inr_idle, cfg, dev
        self.N = len(phat)

    def _sb(self, idx, G, Gsp):
        return SlotBatch(self.phat[idx], self.y[idx], self.e[idx], G, Gsp,
                         self.inr_busy, self.inr_idle, self.dev)

    def batch(self, rng, B):
        G, Gsp = gen_topologies(rng, B, self.cfg)
        return self._sb(rng.integers(0, self.N, B), G, Gsp)

    def sequence(self, rng, M, L):
        L = min(L, self.N - 1)
        st = rng.integers(0, self.N - L, M)
        G, Gsp = gen_topologies(rng, M, self.cfg)
        return [self._sb(st + t, G, Gsp) for t in range(L)]


class PooledSource:
    def __init__(self, sources):
        self.sources = sources
        self.N = sum(s.N for s in sources)
        self.w = np.array([s.N for s in sources], np.float64) / self.N

    def pick(self, rng):
        return self.sources[int(rng.choice(len(self.sources), p=self.w))]

    def batch(self, rng, B):
        return self.pick(rng).batch(rng, B)

    def sequence(self, rng, M, L):
        return self.pick(rng).sequence(rng, M, L)


def test_blocks(bd: Bundle, phat_of, ppers_of, h: int, pool, cfg):
    """Consecutive test slots of every scenario, cut into blocks of
    topo_block slots; each block uses one SU topology of the fixed TEST pool."""
    L = cfg["topo_block"]
    G_pool, Gsp_pool = pool
    out = defaultdict(list)
    for si, s in enumerate(bd.names):
        D = bd.D[s]
        ph = phat_of(s)[:, h - 1, :]
        pp = ppers_of(s)[:, h - 1, :] if ppers_of else ph
        y, e = D["y_te"][:, h - 1, :], D["E_te"][:, h - 1, :]
        for b in range(ph.shape[0] // L):
            sl = slice(b * L, (b + 1) * L)
            j = (zlib.crc32(s.encode()) + b) % len(G_pool)
            out["PH"].append(ph[sl]); out["PP"].append(pp[sl])
            out["Y"].append(y[sl]); out["E"].append(e[sl])
            out["G"].append(G_pool[j]); out["Gsp"].append(Gsp_pool[j])
            out["SID"].append(si); out["IB"].append(D["inr_busy"]); out["II"].append(D["inr_idle"])
    return {k: np.stack(v) if k != "SID" else np.array(v) for k, v in out.items()}


def block_slot(blk, t, dev):
    return SlotBatch(blk["PH"][:, t], blk["Y"][:, t], blk["E"][:, t], blk["G"], blk["Gsp"],
                     blk["IB"], blk["II"], dev, ppers=blk["PP"][:, t])


# ==========================================================================
#                          COMMON POLICY PIECES
# ==========================================================================
class Allocator:
    learned = False
    sequential = False
    label = ""
    ref = ""

    def __init__(self, cfg, dev, C):
        self.cfg, self.dev, self.C = cfg, dev, C
        self.nets: Dict[str, nn.Module] = {}

    def reset(self, B):
        pass

    def act(self, sb: SlotBatch):
        raise NotImplementedError

    def feedback(self, sb, ch, out):
        pass

    def train_steps(self, src, n, rng, client=0):
        pass

    def opt_for(self, key, net):
        if not hasattr(self, "_opts"):
            self._opts = {}
        return torch.optim.Adam(net.parameters(), lr=self.cfg["alloc_lr"])


class QNet(nn.Module):
    def __init__(self, fin, nout, h=256):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(fin, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                               nn.Linear(h, nout))

    def forward(self, x):
        return self.f(x)


class _Replay:
    def __init__(self, cap):
        self.cap, self.data, self.pos, self.n = cap, None, 0, 0

    def add(self, *ts):
        ts = [t.detach() for t in ts]
        if self.data is None:
            self.data = [torch.zeros((self.cap,) + t.shape[1:], dtype=t.dtype, device=t.device) for t in ts]
        m = ts[0].shape[0]
        idx = (torch.arange(m, device=ts[0].device) + self.pos) % self.cap
        for d, t in zip(self.data, ts):
            d[idx] = t
        self.pos = (self.pos + m) % self.cap
        self.n = min(self.n + m, self.cap)

    def sample(self, rng, bs):
        j = torch.as_tensor(rng.integers(0, self.n, bs), device=self.data[0].device)
        return [d[j] for d in self.data]


def train_policy(alloc: Allocator, sources: List[SlotSource], mode: str, cfg, seed) -> Dict:
    """Centralised (pooled slots) or FedAvg (one client per scenario)."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    if not alloc.learned:
        return {"train_s": 0.0, "comm_bytes": 0}
    if mode == "fedavg" and len(sources) > 1:
        rounds = cfg["fl_rounds"]
        for _ in range(rounds):
            g = {k: copy.deepcopy(n.state_dict()) for k, n in alloc.nets.items()}
            states, ws = [], []
            for ci, src in enumerate(sources):
                for k, n in alloc.nets.items():
                    n.load_state_dict(g[k])
                alloc.train_steps(src, cfg["fl_local_steps"], rng, client=ci)
                states.append({k: copy.deepcopy(n.state_dict()) for k, n in alloc.nets.items()})
                ws.append(src.N)
            for k, n in alloc.nets.items():
                n.load_state_dict(_avg_states([s[k] for s in states], ws))
        nb = sum(model_bytes(n) for n in alloc.nets.values())
        return {"train_s": time.time() - t0, "comm_bytes": 2 * nb * len(sources) * rounds,
                "rounds": rounds}
    alloc.train_steps(PooledSource(sources), cfg["train_steps"], rng)
    return {"train_s": time.time() - t0, "comm_bytes": 0}


def profile_policy(a: Allocator, C: int, cfg: Dict) -> Dict:
    """Never stops the run: a profiling failure is logged and reported as NaN."""
    try:
        return _profile_policy(a, C, cfg)
    except Exception as e:
        log(f"  profiling {a.label} failed ({type(e).__name__}: {e}) -> NaN in the complexity CSV")
        return {"method": a.label, "method_key": getattr(a, "key", type(a).__name__),
                "reference": a.ref, "profiling_error": f"{type(e).__name__}: {e}"}


def _profile_policy(a: Allocator, C: int, cfg: Dict) -> Dict:
    """Params, torch-counted FLOPs and latency of ONE slot decision
    (K links, C sub-channels): CPU batch 1, and GPU batch 1 if available."""
    rng = np.random.default_rng(0)
    G, Gsp = gen_topologies(rng, 1, cfg)
    ph = rng.uniform(0, 1, (1, C)).astype(np.float32)
    ib, ii = np.full(C, 10.0, np.float32), np.zeros(C, np.float32)
    row = {"method": a.label, "method_key": getattr(a, "key", type(a).__name__),
           "reference": a.ref, "su_links": cfg["su_links"], "subchannels": C,
           "params": int(sum(p.numel() for n in a.nets.values() for p in n.parameters()))}
    row["params_M"] = row["params"] / 1e6
    for devname in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
        dv = torch.device(devname)
        b = copy.copy(a)
        b.dev = dv
        b.nets = {k: copy.deepcopy(n).to(dv) for k, n in a.nets.items()}
        for attr, val in vars(a).items():           # e.g. power levels, diffusion schedule
            if attr == "nets":
                continue
            if torch.is_tensor(val):
                setattr(b, attr, val.to(dv))
            elif isinstance(val, nn.Module):
                setattr(b, attr, copy.deepcopy(val).to(dv))
        b.prev = None
        sb = SlotBatch(ph, (ph > 0.5).astype(np.float32), np.zeros((1, C), np.float32),
                       G, Gsp, ib, ii, dv, ppers=ph)
        b.reset(1)
        sync = torch.cuda.synchronize if devname == "cuda" else None
        if devname == "cpu":
            row["flops_per_slot"] = count_flops(lambda: b.act(sb))
            row["mflops_per_slot"] = row["flops_per_slot"] / 1e6
        row[f"{devname}_latency_ms_per_slot"] = _time_call(
            lambda: b.act(sb), max(3, cfg["prof_iters"] // 5), cfg["prof_warmup"], sync) * 1e3
    return row


# ==========================================================================
#         NOVEL STAGE-2 ENGINE: sensing memory, fusion, exact expectation
# ==========================================================================
#  Facts from the SOTA results that the novel methods act on:
#   * Greedy with the TRUE future reaches 168 of the oracle's 169, but only
#     138 with the forecast: the loss is in how uncertainty is used, not in
#     the assignment search.
#   * The classic greedy scores a sub-channel with ONE rate at the MEAN
#     interference. Busy-channel PU power spans 6-50 dB, so E[log(1+SINR)]
#     differs strongly from log(1+SINR at mean INR) (Jensen).
#   * The sensor keeps measuring: at forecast time the occupancy and PU
#     power of the last slot are known (sensing memory). No SOTA method
#     used them; a persistent PU keeps a similar strength.
#   * Weak PUs (6-16 dB) cause most violations; strong PUs tolerate sharing.
# ==========================================================================
Q_ATOMS = 8


def scenario_stats(y1: np.ndarray, e1: np.ndarray, eta: float, C: int) -> Dict:
    """From TRAINING slots (h = 1) of one scenario: busy-INR quantile atoms per
    sub-channel, idle INR mean, and how often a busy PU keeps its strength."""
    inr = np.maximum(np.power(10.0, e1 / 10.0) - 1.0, 0.0)
    busy = y1 > 0.5
    lv = (np.arange(Q_ATOMS) + 0.5) / Q_ATOMS
    allb = inr[busy] if busy.any() else np.array([10.0])
    atoms = np.empty((C, Q_ATOMS), np.float32)
    idle = np.empty(C, np.float32)
    for c in range(C):
        v = inr[busy[:, c], c]
        atoms[c] = np.quantile(v if v.size >= 8 else allb, lv)
        w = inr[~busy[:, c], c]
        idle[c] = w.mean() if w.size else 0.0
    both = busy[1:] & busy[:-1]
    keep = float((np.abs(e1[1:] - e1[:-1]) <= 3.0)[both].mean()) if both.any() else 0.0
    return {"atoms": atoms, "inr_idle": idle, "persist": keep}


def memory_of(y: np.ndarray, e: np.ndarray):
    """Sensing memory of CONSECUTIVE windows: what was measured at the forecast
    time of window i is the h=1 target of window i-1 (observed, causal)."""
    ym = np.zeros_like(y); em = np.zeros_like(e); ok = np.zeros(len(y), np.float32)
    ym[1:], em[1:], ok[1:] = y[:-1], e[:-1], 1.0
    return ym, em, ok


class FSF(nn.Module):
    """Forecast-Sensing Fusion (novel): calibrated occupancy probability from
    the forecast logit, the last sensed state and the last margin of the same
    sub-channel. Fitted per scenario on TRAINING slots (local, never shared)."""

    def __init__(self, C):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        self.b = nn.Parameter(torch.zeros(C))

    def forward(self, ph, ym, em, ok, eta):
        lg = torch.logit(ph.clamp(1e-4, 1 - 1e-4))
        x = torch.stack([lg, (2 * ym - 1) * ok[:, None], ((em - eta) / 10).clamp(-2, 4) * ok[:, None],
                         ok[:, None].expand_as(lg)], -1)
        return torch.sigmoid((x * self.w).sum(-1) + self.b)


def fit_fsf(ph, y, ym, em, ok, eta, C, dev, steps=300) -> FSF:
    m = FSF(C).to(dev)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    ph, y, ym, em, ok = t(ph), t(y), t(ym), t(em), t(ok)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(steps):
        p = m(ph, ym, em, ok, eta)
        loss = F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


class Ctx:
    """Per-slot inputs of the novel methods (B slots on device)."""

    def __init__(self, sb: SlotBatch, q, atoms, wts, n_idle):
        self.sb, self.q, self.atoms, self.wts, self.n_idle = sb, q, atoms, wts, n_idle


def build_ctx(sb: SlotBatch, st_list: List[Dict], sid: np.ndarray, use_memory: bool,
              fsf_list: Optional[List] = None, cfg=None) -> Ctx:
    """q = fused (or raw) occupancy probability; busy-INR distribution =
    training quantile atoms, mixed with the last measured PU power when the
    sub-channel was busy at forecast time (persistent PU)."""
    dev, B, C = sb.dev, sb.B, sb.C
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
    atoms = t(np.stack([st_list[i]["atoms"] for i in sid]))                     # [B,C,Q]
    n_idle = 1.0 + t(np.stack([st_list[i]["inr_idle"] for i in sid]))          # [B,C]
    persist = t([st_list[i]["persist"] for i in sid])[:, None]
    q = sb.phat
    if fsf_list is not None:
        q = torch.empty_like(sb.phat)
        for i in np.unique(sid):
            m = torch.as_tensor(sid == i, device=dev)
            with torch.no_grad():
                q[m] = fsf_list[i](sb.phat[m], sb.ym[m], sb.em[m], sb.mok[m], cfg["eta_db"])
    wq = torch.full((B, C, Q_ATOMS), 1.0 / Q_ATOMS, device=dev)
    if use_memory:
        last = torch.clamp(torch.pow(10.0, sb.em / 10.0) - 1.0, min=0.0)[..., None]
        w = (persist * (sb.ym > 0.5).float() * sb.mok[:, None])[..., None]      # [B,C,1]
        atoms = torch.cat([atoms, last], -1)
        wts = torch.cat([wq * (1 - w), w], -1)
    else:
        wts = wq
    return Ctx(sb, q.clamp(0, 1), atoms, wts, n_idle)


class Engine:
    """Exact-expectation state of one allocation (B slots, K links).
    Expected rate of link j on sub-channel c with interference I:
      (1-q_c) log2(1 + p S_j/(N_idle,c + I)) + q_c sum_a w_ca log2(1 + p S_j/(1 + X_ca + I))
    Sharing mode adds the PU-violation risk of each busy sub-channel:
      q_c sum_a w_ca 1[X_ca/(1 + I_pu,c) < gamma]."""

    def __init__(self, ctx: Ctx, cfg, sharing: bool, mu_offset=None):
        sb = ctx.sb
        self.ctx, self.cfg, self.sharing = ctx, cfg, sharing
        self.B, self.K, self.C = sb.B, sb.K, sb.C
        self.G, self.S = sb.G, torch.diagonal(sb.G, dim1=1, dim2=2)
        self.ar = torch.arange(self.B, device=sb.dev)
        self.ch = torch.full((self.B, self.K), self.C, dtype=torch.long, device=sb.dev)
        self.P = torch.zeros(self.B, self.K, device=sb.dev)
        self.Ipu = torch.zeros(self.B, self.C, device=sb.dev)
        self.lam = cfg.get("collision_penalty", 2.0)
        self.mu = cfg.get("violation_penalty", 10.0)
        self.gam = 10 ** (cfg.get("pu_sinr_min_db", 3.0) / 10.0)
        self.mu_offset = mu_offset                                       # [B,C] logit offset (OR-PACS)

    # ---- expected quantities (broadcast over any trailing dims) ----
    @staticmethod
    def _r(S, I, q, ni, X, W):
        """E[log2(1+SINR)]: idle branch + busy branch over the PU-power atoms."""
        r_idle = torch.log2(1 + S / (ni + I))
        r_busy = (W * torch.log2(1 + S[..., None] / (1 + X + I[..., None]))).sum(-1)
        return (1 - q) * r_idle + q * r_busy

    def _gather_ch(self, c):
        """Per-link channel parameters for channel index c [B,K]."""
        q = torch.gather(self.ctx.q, 1, c)
        ni = torch.gather(self.ctx.n_idle, 1, c)
        X = self.ctx.atoms[self.ar[:, None], c]
        Wt = self.ctx.wts[self.ar[:, None], c]
        return q, ni, X, Wt

    def _vp(self, Ipu):
        """Violation probability for Ipu [B,C,*] -> same shape."""
        extra = Ipu.dim() - 2
        X = self.ctx.atoms.reshape(self.B, self.C, *([1] * extra), -1)
        Wt = self.ctx.wts.reshape(self.B, self.C, *([1] * extra), -1)
        q = self.ctx.q.reshape(self.B, self.C, *([1] * extra))
        v = (Wt * (X / (1 + Ipu[..., None]) < self.gam).float()).sum(-1)
        p = q * v * (Ipu > 0).float()
        if self.mu_offset is not None:
            off = self.mu_offset.reshape(self.B, self.C, *([1] * extra))
            p = torch.sigmoid(torch.logit(p.clamp(1e-5, 1 - 1e-5)) + off) * (Ipu > 0).float()
        return p

    def viol_prob(self, Ipu):
        return self._vp(Ipu)

    def interference(self):
        A = F.one_hot(self.ch.clamp(max=self.C - 1), self.C).float() * (self.ch < self.C).float()[..., None]
        return torch.einsum("bjc,bjk->bkc", A * self.P[..., None], self.G), A

    def link_values(self, k, levels):
        """Marginal expected objective of putting link k ([B]) on each
        sub-channel at each power level -> [B, C, L]; link k must be unassigned."""
        B, C, ar = self.B, self.C, self.ar
        lv = torch.as_tensor(levels, dtype=torch.float32, device=self.G.device)
        I_all, A = self.interference()
        Ikc = I_all[ar, k]                                                  # [B,C]
        Sk = self.S[ar, k]
        # own expected rate on every (sub-channel, level)
        S_own = Sk[:, None, None] * lv[None, None, :]                       # [B,1,L]
        own = self._r(S_own, Ikc[..., None], self.ctx.q[..., None], self.ctx.n_idle[..., None],
                      self.ctx.atoms[:, :, None, :], self.ctx.wts[:, :, None, :])   # [B,C,L]
        # loss caused to the links already on each sub-channel
        jch = self.ch.clamp(max=C - 1)
        on = (self.ch < C).float()
        q, ni, X, Wt = self._gather_ch(jch)                                 # [B,K], [B,K,Q]
        Ij = torch.gather(I_all, 2, jch[..., None]).squeeze(-1)
        Sj = self.P * self.S
        base = self._r(Sj, Ij, q, ni, X, Wt)                                # [B,K]
        g_out = self.G[ar, k, :]                                            # tx k -> rx j
        new = self._r(Sj[..., None], Ij[..., None] + g_out[..., None] * lv, q[..., None], ni[..., None],
                      X[:, :, None, :], Wt[:, :, None, :])                  # [B,K,L]
        harm = torch.einsum("bjc,bjl->bcl", A, (base[..., None] - new) * on[..., None])
        val = own - harm
        if self.sharing:
            gsp = self.ctx.sb.Gsp[ar, k][:, None, None]
            val = val - self.mu * (self._vp(self.Ipu[..., None] + gsp * lv) - self._vp(self.Ipu)[..., None])
        else:
            val = val - self.lam * self.ctx.q[..., None]
        return val

    def place(self, k, c, p, mask):
        b, kk, cc, pp = self.ar[mask], k[mask], c[mask], p[mask]
        self.ch[b, kk] = cc
        self.P[b, kk] = pp
        self.Ipu[b, cc] += pp * self.ctx.sb.Gsp[b, kk]

    def remove(self, k):
        b = self.ar
        on = self.ch[b, k] < self.C
        if on.any():
            bb, kk = b[on], k[on]
            cc = self.ch[bb, kk]
            self.Ipu[bb, cc] = (self.Ipu[bb, cc] - self.P[bb, kk] * self.ctx.sb.Gsp[bb, kk]).clamp(min=0)
        self.ch[b, k] = self.C
        self.P[b, k] = 0.0

    def best_response(self, k, levels):
        vals = self.link_values(k, levels)                                 # [B,C,L]
        flat = vals.reshape(self.B, -1)
        best, idx = flat.max(1)
        c, li = idx // len(levels), idx % len(levels)
        p = torch.as_tensor(levels, device=self.G.device)[li]
        self.place(k, c, p, best > 0)

    def solve(self, levels, ls_passes=2, init=None):
        """Greedy (strongest link first) + best-response local search: every
        move strictly increases the expected objective, so it terminates."""
        order = torch.argsort(-self.S, dim=1)
        if init is not None:
            ch0, p0 = init
            for i in range(self.K):
                k = order[:, i]
                c, p = ch0[self.ar, k], p0[self.ar, k]
                self.place(k, c.clamp(max=self.C - 1), p, c < self.C)
        else:
            for i in range(self.K):
                self.best_response(order[:, i], levels)
        for _ in range(ls_passes):
            for i in range(self.K):
                k = order[:, i]
                self.remove(k)
                self.best_response(k, levels)
        return self.ch.clone(), self.P.clone()

    def objective(self):
        """Expected objective of the current state [B]."""
        C = self.C
        I_all, _ = self.interference()
        jch = self.ch.clamp(max=C - 1)
        q, ni, X, Wt = self._gather_ch(jch)
        Ij = torch.gather(I_all, 2, jch[..., None]).squeeze(-1)
        on = (self.ch < C).float()
        u = (self._r(self.P * self.S, Ij, q, ni, X, Wt) * on).sum(1)
        if self.sharing:
            return u - self.mu * self._vp(self.Ipu).sum(1)
        return u - self.lam * (q * on).sum(1)


def engine_score(ctx, cfg, sharing, ch, pw, mu_offset=None):
    e = Engine(ctx, cfg, sharing, mu_offset)
    on = ch < ctx.sb.C
    for k in range(e.K):
        kk = torch.full((e.B,), k, dtype=torch.long, device=ctx.sb.dev)
        e.place(kk, ch[:, k].clamp(max=e.C - 1), pw[:, k], on[:, k])
    return e.objective()


# ---------------------------------------------------------------- memory-aware data
class MemSource(SlotSource):
    """Training slots with sensing memory (consecutive training windows)."""

    def __init__(self, name, sid, phat, y, e, inr_busy, inr_idle, cfg, dev):
        super().__init__(name, phat, y, e, inr_busy, inr_idle, cfg, dev)
        self.sid = sid
        self.ym, self.em, self.ok = memory_of(y, e)

    def _sb(self, idx, G, Gsp):
        sb = SlotBatch(self.phat[idx], self.y[idx], self.e[idx], G, Gsp,
                       self.inr_busy, self.inr_idle, self.dev)
        return attach_memory(sb, self.ym[idx], self.em[idx], self.ok[idx], np.full(len(idx), self.sid))


def attach_memory(sb, ym, em, ok, sid):
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=sb.dev)
    sb.ym, sb.em, sb.mok, sb.sid = t(ym), t(em), t(ok), np.asarray(sid)
    return sb


def test_blocks_mem(bd, phat_of, ppers_of, h, pool, cfg):
    blk = test_blocks(bd, phat_of, ppers_of, h, pool, cfg)
    L = cfg["topo_block"]
    YM, EM, OK = [], [], []
    for s in bd.names:
        D = bd.D[s]
        ym, em, ok = memory_of(D["y_te"][:, 0, :], D["E_te"][:, 0, :])
        for b in range(phat_of(s).shape[0] // L):
            sl = slice(b * L, (b + 1) * L)
            YM.append(ym[sl]); EM.append(em[sl]); OK.append(ok[sl])
    blk["YM"], blk["EM"], blk["OK"] = np.stack(YM), np.stack(EM), np.stack(OK)
    return blk


def block_slot_mem(blk, t, dev):
    sb = block_slot(blk, t, dev)
    return attach_memory(sb, blk["YM"][:, t], blk["EM"][:, t], blk["OK"][:, t], blk["SID"])


# ---------------------------------------------------------------- GRAIL network
class GRAILNet(nn.Module):
    """Graph Residual Allocation with Interference-aware Learning (novel).
    Per-link tokens carry the ANALYTIC marginal values of every action (from
    the engine) + gains; two attention layers over the links with an edge
    bias from the cross gains; output = analytic values / tau + learned
    residual (zero-initialised): starts as the engine's own preference and
    learns corrections from the measured truth of training slots."""

    def __init__(self, n_act, h=64, nhead=4):
        super().__init__()
        self.nhead = nhead
        self.inp = nn.Linear(n_act + 3, h)
        self.att = nn.ModuleList([nn.MultiheadAttention(h, nhead, batch_first=True) for _ in range(2)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(h, 2 * h), nn.GELU(), nn.Linear(2 * h, h)) for _ in range(2)])
        self.n = nn.ModuleList([nn.LayerNorm(h) for _ in range(4)])
        self.edge = nn.Linear(2, nhead)
        self.out = nn.Linear(h, n_act)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.log_tau = nn.Parameter(torch.zeros(1))

    def forward(self, vals, gk, gsp, gx):
        """vals [B,K,A] analytic values (idle = 0 appended), gk/gsp [B,K] dB/40,
        gx [B,K,K] cross gains dB/40."""
        B, K, A = vals.shape
        z = self.inp(torch.cat([vals / 5.0, gk[..., None], gsp[..., None], vals.amax(-1, keepdim=True) / 5], -1))
        eb = self.edge(torch.stack([gx, gx.transpose(1, 2)], -1)).permute(0, 3, 1, 2).reshape(B * self.nhead, K, K)
        for i in range(2):
            h = self.n[2 * i](z)
            z = z + self.att[i](h, h, h, attn_mask=eb, need_weights=False)[0]
            z = z + self.ff[i](self.n[2 * i + 1](z))
        return vals / torch.exp(self.log_tau) + self.out(z)


def all_link_values(eng: Engine, levels):
    """Analytic value of every action for every link, each evaluated with the
    link removed from the current state -> [B,K,C*L+1] (last = idle = 0)."""
    B, K, C = eng.B, eng.K, eng.C
    out = []
    for k in range(K):
        kk = torch.full((B,), k, dtype=torch.long, device=eng.G.device)
        save = (eng.ch[:, k].clone(), eng.P[:, k].clone())
        eng.remove(kk)
        v = eng.link_values(kk, levels).reshape(B, -1)
        on = save[0] < C
        eng.place(kk, save[0].clamp(max=C - 1), save[1], on)
        out.append(torch.cat([v, torch.zeros(B, 1, device=v.device)], 1))
    return torch.stack(out, 1)


def decode_actions(a, levels, C):
    L = len(levels)
    idle = a == C * L
    ch = torch.where(idle, torch.full_like(a, C), a // L)
    lv = torch.as_tensor(levels, device=a.device)[(a % L).clamp(max=L - 1)]
    return ch, lv * (~idle).float()


# ==========================================================================
#                    ALLOCATION OUTCOME (measured future)
# ==========================================================================
def evaluate_alloc(sb: SlotBatch, ch, pw, cfg: Dict) -> Dict:
    tx, chc, A, p, sinr, rate = su_side(sb, ch, pw, cfg)
    col = tx & (torch.gather(sb.y, 1, chc) > 0.5)
    sr = rate.sum(1)
    return {"rate": rate, "tx": tx, "col": col, "hit": A.sum(1) > 0, "busy": sb.y > 0.5,
            "sinr": sinr, "sum_rate": sr, "pw": p,
            "utility": sr - cfg["collision_penalty"] * col.float().sum(1)}


# ==========================================================================
#                           NOVEL ALLOCATORS
# ==========================================================================
class NovelAlloc(Allocator):
    """Shared plumbing: per-scenario training statistics and fusion models."""
    use_fsf, use_memory = False, False

    def __init__(self, cfg, dev, C):
        super().__init__(cfg, dev, C)
        self.st, self.fsf, self.oracle_fc = None, None, False

    def setup(self, st_list, fsf_list):
        self.st, self.fsf = st_list, fsf_list

    def ctx(self, sb):
        sb2 = sb
        if self.oracle_fc:                                    # perfect forecast: no fusion needed
            return build_ctx(sb2, self.st, sb.sid, False, None, self.cfg)
        return build_ctx(sb2, self.st, sb.sid, self.use_memory,
                         self.fsf if self.use_fsf else None, self.cfg)

    def act(self, sb):
        with torch.no_grad():
            eng = Engine(self.ctx(sb), self.cfg, sharing=False)
            return eng.solve([1.0], self.cfg["ls_passes"])


class BEGALS(NovelAlloc):
    label = "BEGA-LS"
    ref = "novel: Bayes-exact expected utility (busy-INR quantiles) + greedy + best-response local search"


class FSFBEGA(NovelAlloc):
    label = "FSF-BEGA"
    ref = "novel: forecast-sensing fusion (calibrated q, PU-strength memory) + BEGA-LS"
    use_fsf, use_memory = True, True


class GRAIL(NovelAlloc):
    label = "GRAIL"
    ref = "novel: attention policy residual on analytic values, REINFORCE on measured truth, " \
          "best-of-N under the exact expected utility, then local search"
    learned = True
    use_fsf, use_memory = True, True

    def __init__(self, cfg, dev, C):
        super().__init__(cfg, dev, C)
        self.nets["pi"] = GRAILNet(C + 1, cfg["gnn_hidden"]).to(dev)
        self.levels = [1.0]
        self.steps_done = 0

    def _feats(self, sb):
        gk = sb.gdirect_db() / 40.0
        gsp = 10 * torch.log10(sb.Gsp + 1e-12) / 40.0
        return gk, gsp, sb.edge_feats()

    def act(self, sb):
        net = self.nets["pi"]; net.eval()
        with torch.no_grad():
            ctx = self.ctx(sb)
            eng = Engine(ctx, self.cfg, False)
            ch0, p0 = eng.solve(self.levels, self.cfg["ls_passes"])
            if self.oracle_fc:
                return ch0, p0
            vals = all_link_values(eng, self.levels)
            lg = net(vals, *self._feats(sb))
            cands = [(ch0, p0), decode_actions(lg.argmax(-1), self.levels, sb.C)]
            g = torch.Generator(device=sb.dev).manual_seed(77)
            prob = torch.softmax(lg, -1).reshape(-1, lg.shape[-1])
            for _ in range(self.cfg["grail_samples"]):
                a = torch.multinomial(prob, 1, generator=g).reshape(lg.shape[:2])
                cands.append(decode_actions(a, self.levels, sb.C))
            sc = torch.stack([engine_score(ctx, self.cfg, False, c, p) for c, p in cands], 1)
            best = sc.argmax(1)
            ar = torch.arange(sb.B, device=sb.dev)
            ch = torch.stack([c for c, _ in cands], 1)[ar, best]
            pw = torch.stack([p for _, p in cands], 1)[ar, best]
            eng2 = Engine(ctx, self.cfg, False)
            return eng2.solve(self.levels, 1, init=(ch, pw))

    def train_steps(self, src, n, rng, client=0):
        net = self.nets["pi"]; opt = self.opt_for("pi", net)
        tot = max(1, self.cfg["train_steps"])
        for _ in range(n):
            sb = src.batch(rng, self.cfg["grail_batch"])
            with torch.no_grad():
                ctx = self.ctx(sb)
                eng = Engine(ctx, self.cfg, False)
                ch0, p0 = eng.solve(self.levels, 1)
                vals = all_link_values(eng, self.levels)
                u0 = evaluate_alloc(sb, ch0, p0, self.cfg)["utility"]
            net.train()
            lg = net(vals, *self._feats(sb))
            dist = torch.distributions.Categorical(logits=lg)
            a = dist.sample()
            ch, pw = decode_actions(a, self.levels, sb.C)
            with torch.no_grad():
                adv = (evaluate_alloc(sb, ch, pw, self.cfg)["utility"] - u0) / sb.K
                adv = adv - adv.mean()
            beta = 0.01 * max(0.0, 1 - self.steps_done / tot)
            loss = -(adv[:, None] * dist.log_prob(a)).sum(1).mean() - beta * dist.entropy().mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            self.steps_done += 1


METHODS = {"bega_ls": BEGALS, "fsf_bega": FSFBEGA, "grail": GRAIL}


# ==========================================================================
#                              EVALUATION
# ==========================================================================
def eval_method(a, blk, cfg, dev, edges, eta):
    NB, L, C = blk["PH"].shape
    K = blk["G"].shape[1]
    acc = defaultdict(lambda: np.zeros(NB))
    rate_acc = np.zeros((NB, K))
    nb = len(edges) - 1
    sh, sn = np.zeros((NB, nb)), np.zeros((NB, nb))
    a.reset(NB)
    t_act = 0.0
    cpu = lambda v: v.float().detach().cpu().numpy()
    for t in range(L):
        sb = block_slot_mem(blk, t, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ch, pw = a.act(sb)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_act += time.perf_counter() - t0
        o = evaluate_alloc(sb, ch, pw, cfg)
        a.feedback(sb, ch, o)
        tx, col, hit, busy = o["tx"], o["col"], o["hit"], o["busy"]
        acc["sum_rate"] += cpu(o["sum_rate"]); acc["utility"] += cpu(o["utility"])
        acc["n_tx"] += cpu(tx.sum(1)); acc["n_col"] += cpu(col.sum(1))
        acc["hit_busy"] += cpu((hit & busy).sum(1)); acc["n_busy"] += cpu(busy.sum(1))
        acc["hit_idle"] += cpu((hit & ~busy).sum(1)); acc["n_idle"] += cpu((~busy).sum(1))
        acc["outage"] += cpu(((o["sinr"] < 1) & tx).sum(1))
        acc["sum_pw"] += cpu(o["pw"].sum(1))
        rate_acc += cpu(o["rate"])
        e, bz, hz = blk["E"][:, t], cpu(busy) > 0.5, cpu(hit) > 0.5
        for i in range(nb):
            m = bz & _bin_mask(e, edges[i], edges[i + 1], i == nb - 1, eta)
            sn[:, i] += m.sum(1); sh[:, i] += (m & hz).sum(1)
    s1, s2 = rate_acc.sum(1), (rate_acc ** 2).sum(1)
    acc["jain"] = np.where(s2 > 0, s1 ** 2 / (K * np.maximum(s2, 1e-12)), 0.0)
    return acc, sh, sn, t_act / (NB * L)


def alloc_row(acc, mask, L, K):
    sl = float(mask.sum() * L)
    S = lambda k: float(acc[k][mask].sum())
    ntx = max(S("n_tx"), 1.0)
    return {"n_slots": int(sl), "sum_rate_per_slot": S("sum_rate") / sl,
            "rate_per_link": S("sum_rate") / sl / K, "rate_per_active_link": S("sum_rate") / ntx,
            "utility_per_slot": S("utility") / sl, "collisions_per_slot": S("n_col") / sl,
            "collision_rate": S("n_col") / ntx, "pu_harm_rate": S("hit_busy") / max(S("n_busy"), 1.0),
            "idle_utilization": S("hit_idle") / max(S("n_idle"), 1.0),
            "access_ratio": S("n_tx") / (sl * K), "outage_rate": S("outage") / ntx,
            "jain_fairness": float(acc["jain"][mask].mean()),
            "energy_eff_bit_per_unit_power": S("sum_rate") / max(S("sum_pw"), 1e-9)}


def oracle_utility(blk, cfg, dev):
    """Same oracle as the SOTA script (greedy + WMMSE with the true future),
    so regret_vs_oracle is directly comparable."""
    NB, L, C = blk["PH"].shape
    U = np.zeros(NB)
    for t in range(L):
        sb = block_slot_mem(blk, t, dev)
        ch, _ = greedy_assign(sb, cfg, oracle=True)
        pw = wmmse_power(sb, ch, cfg, noise=1.0 + torch.gather(sb.inr, 1, ch.clamp(max=C - 1)))
        U += evaluate_alloc(sb, ch, pw, cfg)["utility"].cpu().numpy()
    return U


def write_outputs(out_dir, ROWS, SNR, CX, COMM):
    if not ROWS:
        return
    keys = ["method", "method_key", "trained", "mode", "forecaster", "forecaster_key", "scenario", "horizon_step"]
    write_csv(os.path.join(out_dir, "alloc_overall.csv"), ROWS)
    write_csv(os.path.join(out_dir, "alloc_overall_agg.csv"), aggregate(ROWS, keys))
    write_csv(os.path.join(out_dir, "alloc_per_snr.csv"), aggregate(SNR, keys + ["snr_bin"]))
    write_csv(os.path.join(out_dir, "alloc_complexity.csv"), list(CX.values()))
    write_csv(os.path.join(out_dir, "fl_communication.csv"), COMM)


def fit_fsf_all(bd, P, names, cfg, dev):
    out = []
    for s in names:
        ym, em, ok = memory_of(bd.D[s]["y_tr1"], bd.D[s]["E_tr1"])
        out.append(fit_fsf(P[s]["tr1"], bd.D[s]["y_tr1"], ym, em, ok, cfg["eta_db"], bd.C, dev))
    return out


def run(cfg, dev, bd: Bundle, bdn: Optional[Bundle], out_dir):
    names, C, edges = bd.names, bd.C, bd.edges
    pool = gen_topologies(np.random.default_rng(cfg["seed"] + 999), cfg["topo_pool_test"], cfg)
    st_list = [scenario_stats(bd.D[s]["y_tr1"], bd.D[s]["E_tr1"], bd.eta, C) for s in names]
    ROWS, SNR, CX, COMM = [], [], {}, []
    K = cfg["su_links"]
    pers = bd.preds("persistence", "non-learned", 0)
    for seed in cfg["seeds"]:
        for mode in cfg["modes"]:
            bm = bd.man["best_by_mode"].get(mode)
            P = bd.preds(bm, mode, seed) if bm else None
            if P is None or any("tr1" not in P[s] for s in names):
                log(f"  [{mode} seed {seed}] no SOTA forecasts in the hand-off -> skipped")
                continue
            fsf_sota = fit_fsf_all(bd, P, names, cfg, dev)       # per scenario, local in FL
            sources = [MemSource(s, i, P[s]["tr1"], bd.D[s]["y_tr1"], bd.D[s]["E_tr1"],
                                 bd.D[s]["inr_busy"], bd.D[s]["inr_idle"], cfg, dev) for i, s in enumerate(names)]
            ms = {}
            for mk in cfg["methods"]:
                a = METHODS[mk](cfg, dev, C); a.key = mk
                a.setup(st_list, fsf_sota)
                set_seed(run_seed(cfg, mk, mode, seed))
                t0 = time.time()
                info = train_policy(a, sources, mode, cfg, run_seed(cfg, mk, mode, seed, "rng"))
                ms[mk] = (a, info)
                if a.learned:
                    log(f"  trained {a.label:<10s} {mode:<11s} seed {seed} ({time.time() - t0:.0f} s)")
                    COMM.append({"method": a.label, "mode": mode, "seed": seed,
                                 "comm_MB": info.get("comm_bytes", 0) / 1e6,
                                 "rounds": info.get("rounds", ""), "train_s": info["train_s"],
                                 "note": "FSF fusion fitted per scenario, never communicated"})
            fcs = []
            for fc in cfg["forecasters"]:
                if fc == "oracle":
                    fcs.append((fc, "Oracle forecast", lambda s: bd.D[s]["y_te"], None, True))
                elif fc == "best":
                    fcs.append((fc, f"{bd.label(bm)} ({mode})", lambda s, P=P: P[s]["te"], fsf_sota, False))
                elif fc == "best_novel":
                    nm = bdn.man["best_by_mode"].get(mode) if bdn else None
                    Pn = bdn.preds(nm, mode, seed) if nm else None
                    if Pn is None:
                        log(f"  novel forecaster for {mode} seed {seed} not found -> skipped")
                        continue
                    fsf_n = fit_fsf_all(bd, Pn, names, cfg, dev) if all("tr1" in Pn[s] for s in names) else fsf_sota
                    fcs.append((fc, f"{bdn.label(nm)} ({mode}, novel)", lambda s, Pn=Pn: Pn[s]["te"], fsf_n, False))
                else:
                    Pb = bd.preds(fc, "non-learned", 0)
                    if Pb is None:
                        continue
                    fcs.append((fc, bd.label(fc), lambda s, Pb=Pb: Pb[s]["te"], fsf_sota, False))
            for fc, flab, phat_of, fsf_fc, is_or in fcs:
                for h in cfg["horizons"]:
                    if h > bd.H:
                        continue
                    blk = test_blocks_mem(bd, phat_of, (lambda s: pers[s]["te"]) if pers else None, h, pool, cfg)
                    L = blk["PH"].shape[1]
                    U_or = oracle_utility(blk, cfg, dev)
                    for mk in cfg["methods"]:
                        a, info = ms[mk]
                        a.fsf, a.oracle_fc = fsf_fc, is_or
                        acc, sh, sn, lat = eval_method(a, blk, cfg, dev, edges, bd.eta)
                        for si, s in enumerate(names + (["Combined"] if len(names) > 1 else [])):
                            mask = np.ones(len(blk["SID"]), bool) if s == "Combined" else blk["SID"] == si
                            if not mask.any():
                                continue
                            r = {"method": a.label, "method_key": mk, "reference": a.ref,
                                 "trained": mode if a.learned else "non-learned", "mode": mode,
                                 "forecaster": flab, "forecaster_key": fc, "seed": seed,
                                 "scenario": s, "horizon_step": h, **alloc_row(acc, mask, L, K),
                                 "decision_latency_ms_per_slot": lat * 1e3,
                                 "train_s": info.get("train_s", 0.0), "comm_MB": info.get("comm_bytes", 0) / 1e6}
                            uo = float(U_or[mask].sum()) / (mask.sum() * L)
                            r["oracle_utility_per_slot"] = uo
                            r["regret_vs_oracle"] = 1 - r["utility_per_slot"] / uo if uo > 0 else float("nan")
                            ROWS.append(r)
                            bk = {k: r[k] for k in ("method", "method_key", "trained", "mode", "forecaster",
                                                    "forecaster_key", "seed", "scenario", "horizon_step")}
                            SNR.append({**bk, "snr_bin": f"<= {bd.eta:g} (idle)",
                                        "n_cells": float(acc["n_idle"][mask].sum()),
                                        "idle_utilization": r["idle_utilization"]})
                            for i in range(len(edges) - 1):
                                n_ = float(sn[mask, i].sum())
                                SNR.append({**bk, "snr_bin": f"{edges[i]:g}-{edges[i + 1]:g}", "n_cells": n_,
                                            "pu_hit_rate": float(sh[mask, i].sum()) / n_ if n_ else float("nan")})
                    log(f"  evaluated {len(cfg['methods'])} novel methods | {flab} | h={h} | {mode} | seed {seed}")
            for mk, (a, _) in ms.items():
                if mk not in CX:
                    a.fsf, a.oracle_fc = fsf_sota, False
                    CX[mk] = profile_novel_policy(a, C, cfg)
            write_outputs(out_dir, ROWS, SNR, CX, COMM)
    if not ROWS:
        log("  nothing evaluated (see messages above)")
        return
    write_outputs(out_dir, ROWS, SNR, CX, COMM)


def profile_novel_policy(a, C, cfg) -> Dict:
    try:
        rng = np.random.default_rng(0)
        G, Gsp = gen_topologies(rng, 1, cfg)
        ph = rng.uniform(0, 1, (1, C)).astype(np.float32)
        row = {"method": a.label, "method_key": a.key, "reference": a.ref, "su_links": cfg["su_links"],
               "subchannels": C, "params": int(sum(p.numel() for n in a.nets.values() for p in n.parameters())),
               "params_fsf_per_scenario": 4 + C}
        row["params_M"] = row["params"] / 1e6
        for devname in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
            dv = torch.device(devname)
            b = copy.copy(a); b.dev = dv
            b.nets = {k: copy.deepcopy(n).to(dv) for k, n in a.nets.items()}
            b.fsf = [copy.deepcopy(f).to(dv) for f in a.fsf] if a.fsf else None
            sb = SlotBatch(ph, (ph > 0.5).astype(np.float32), np.zeros((1, C), np.float32), G, Gsp,
                           np.full(C, 10.0, np.float32), np.zeros(C, np.float32), dv, ppers=ph)
            attach_memory(sb, (ph > 0.5).astype(np.float32), np.full((1, C), 10.0, np.float32), np.ones(1, np.float32), np.zeros(1, int))
            sync = torch.cuda.synchronize if devname == "cuda" else None
            if devname == "cpu":
                row["flops_per_slot"] = count_flops(lambda: b.act(sb))
                row["mflops_per_slot"] = row["flops_per_slot"] / 1e6
            row[f"{devname}_latency_ms_per_slot"] = _time_call(lambda: b.act(sb), max(3, cfg["prof_iters"] // 5),
                                                              cfg["prof_warmup"], sync) * 1e3
        return row
    except Exception as e:
        log(f"  profiling {a.label} failed ({type(e).__name__}: {e})")
        return {"method": a.label, "method_key": getattr(a, "key", ""), "profiling_error": str(e)}




# ==========================================================================
#          ABLATION (stage 2): client configurations of the forecaster
#          and of the federated policy training
# ==========================================================================
def ab_tasks(cfg) -> List[Tuple[str, Optional[int]]]:
    return [("K", k) for k in cfg["ab_clients"]] + [("ref", None)]


def task_tag(t) -> str:
    return "references" if t[0] == "ref" else f"K{t[1]:02d}"


def task_configs(t, cfg) -> List[Tuple[str, int, str]]:
    """-> [(partition, n_clients, hand-off key)]"""
    if t[0] == "ref":
        return [("pooled", 1, "centralized")]
    return [(p, t[1], f"fedavg-{p}-K{t[1]}") for p in cfg["ab_partitions"]]


class AbBundle(Bundle):
    """stage2/ of ablation_clients_forecasting_2p4GHz.py."""

    def best(self, mkey: str) -> Optional[str]:
        f = os.path.join(self.dir, f"best__{mkey}.json")
        if not os.path.isfile(f):
            return None
        return json.load(open(f))["model"]


class ShardSource:
    """One federated client: a subset of the training slots of one or more
    scenarios. A batch is drawn from one of its scenario parts (weighted by
    size), exactly as the pooled centralised source does."""

    def __init__(self, parts):
        self.parts = [(s, np.asarray(ix)) for s, ix in parts if len(ix) > 0]
        self.N = int(sum(len(ix) for _, ix in self.parts))
        self.w = np.array([len(ix) for _, ix in self.parts], np.float64) / max(self.N, 1)
        self.mix = {s.name: len(ix) for s, ix in self.parts}

    def batch(self, rng, B):
        s, ix = self.parts[int(rng.choice(len(self.parts), p=self.w))]
        G, Gsp = gen_topologies(rng, B, s.cfg)
        return s._sb(ix[rng.integers(0, len(ix), B)], G, Gsp)

    def sequence(self, rng, M, L):
        s, ix = self.parts[int(rng.choice(len(self.parts), p=self.w))]
        L = min(L, len(ix) - 1)
        st = rng.integers(0, len(ix) - L, M)
        G, Gsp = gen_topologies(rng, M, s.cfg)
        return [s._sb(ix[st + t], G, Gsp) for t in range(L)]


def shard_sources(sources: List, partition: str, K: int, seed: int, cfg) -> List[ShardSource]:
    """Same partitions as the forecasting ablation, on the training SLOTS:
    noniid = K/S contiguous time blocks of ONE scenario per client,
    iid    = uniformly random equal shares of the pooled slots."""
    S = len(sources)
    if partition == "noniid":
        per = [K // S + (1 if i < K % S else 0) for i in range(S)]
        out = []
        for i, src in enumerate(sources):
            cut = np.linspace(0, src.N, per[i] + 1).astype(int)
            out += [ShardSource([(src, np.arange(a, b))]) for a, b in zip(cut[:-1], cut[1:])]
        return out
    if partition == "iid":
        allp = np.concatenate([np.stack([np.full(s.N, i), np.arange(s.N)], 1) for i, s in enumerate(sources)])
        rng = np.random.default_rng(run_seed(cfg, "iid-slots", K, seed))
        out = []
        for sh in np.array_split(rng.permutation(len(allp)), K):
            sel = allp[np.sort(sh)]
            out.append(ShardSource([(sources[i], sel[sel[:, 0] == i, 1]) for i in range(S)]))
        return out
    raise ValueError(partition)


def shard_rows(clients: List[ShardSource], names, partition, K, seed) -> List[Dict]:
    rows = []
    for i, c in enumerate(clients):
        r = {"partition": partition, "n_clients": K, "seed": seed, "client": i, "n_train_slots": c.N}
        for s in names:
            r[f"frac_{s}"] = c.mix.get(s, 0) / max(c.N, 1)
        rows.append(r)
    return rows


def train_policy_k(alloc, clients: List[ShardSource], cfg, seed) -> Dict:
    """FedAvg with K clients. Every round each client takes
    max(fl_min_local_steps, fl_steps_per_round / K) local steps, so the total
    computation per round is the same for every K (K = 4 -> 50 steps = the
    main run); sample-weighted averaging."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    if not alloc.learned:
        return {"train_s": 0.0, "comm_bytes": 0}
    K = len(clients)
    steps = max(cfg["fl_min_local_steps"], int(round(cfg["fl_steps_per_round"] / K)))
    rounds = cfg["fl_rounds"]
    for _ in range(rounds):
        g = {k: copy.deepcopy(n.state_dict()) for k, n in alloc.nets.items()}
        states, ws = [], []
        for ci, src in enumerate(clients):
            for k, n in alloc.nets.items():
                n.load_state_dict(g[k])
            alloc.train_steps(src, steps, rng, client=ci)
            states.append({k: copy.deepcopy(n.state_dict()) for k, n in alloc.nets.items()})
            ws.append(src.N)
        for k, n in alloc.nets.items():
            n.load_state_dict(_avg_states([s[k] for s in states], ws))
    nb = sum(model_bytes(n) for n in alloc.nets.values())
    return {"train_s": time.time() - t0, "comm_bytes": 2 * nb * K * rounds, "rounds": rounds,
            "local_steps": steps, "comm_bytes_per_client_per_round": 2 * nb}


def make_method(mk, cfg, dev, C, st_list, fsf):
    cls = METHODS.get(mk) or REF_METHODS.get(mk)
    a = cls(cfg, dev, C)
    a.key = mk
    if hasattr(a, "setup"):
        a.setup(st_list, fsf)
    return a


def _atomic_pickle(path, obj):
    import pickle
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)


def merge_parts(out_dir, prefix, keys, snr_keys):
    import pickle
    files = sorted(glob.glob(os.path.join(out_dir, "parts", "*.pkl")))
    if not files:
        log("  no parts to merge yet"); return
    R, SN, CM, CX, CL = [], [], [], {}, []
    for f in files:
        with open(f, "rb") as fh:
            p = pickle.load(fh)
        R += p["ROWS"]; SN += p["SNR"]; CM += p["COMM"]; CX.update(p["CX"]); CL += p.get("CLI", [])
    W = lambda n, rows: write_csv(os.path.join(out_dir, n), rows)
    W(f"{prefix}_overall.csv", R)
    W(f"{prefix}_overall_agg.csv", aggregate(R, keys))
    W(f"{prefix}_per_snr.csv", aggregate(SN, snr_keys))
    W("fl_communication.csv", CM)
    W("fl_communication_agg.csv", aggregate(CM, ["method", "mode", "partition", "n_clients"]))
    W("client_partitions.csv", CL)
    if CX:
        W(f"{prefix}_complexity.csv", list(CX.values()))
    log(f"  merged {len(files)} part(s): {', '.join(os.path.basename(f)[:-4] for f in files)}")


def ab_main(desc, prefix, run_task, keys, snr_keys, all_methods):
    global _LOG_FH
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--forecast-dir", help="out_dir of ablation_clients_forecasting_2p4GHz.py")
    ap.add_argument("--out"); ap.add_argument("--seeds"); ap.add_argument("--methods")
    ap.add_argument("--clients"); ap.add_argument("--partitions")
    ap.add_argument("--task", type=int, help="run one part only (SLURM array index)")
    ap.add_argument("--merge", action="store_true", help="only rebuild the merged CSVs from parts/")
    ap.add_argument("--device"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = CONFIG
    if a.quick:
        cfg.update(QUICK)
    if a.forecast_dir:
        cfg["forecast_dir"] = a.forecast_dir
    if a.out:
        cfg["out_dir"] = a.out
    if a.seeds:
        cfg["seeds"] = [int(x) for x in a.seeds.split(",")]
    if a.methods:
        cfg["methods"] = [x.strip() for x in a.methods.split(",")]
    if a.clients:
        cfg["ab_clients"] = [int(x) for x in a.clients.split(",")]
    if a.partitions:
        cfg["ab_partitions"] = [x.strip() for x in a.partitions.split(",")]
    if a.device:
        cfg["device"] = a.device
    bad = [m for m in cfg["methods"] if m not in all_methods]
    if bad:
        sys.exit(f"unknown method(s) {bad}; choose from {list(all_methods)}")
    cfg["out_dir"] = resolve_out(cfg["out_dir"])
    out_dir = cfg["out_dir"]
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    tasks = ab_tasks(cfg)
    if a.task is not None and not 0 <= a.task < len(tasks):
        sys.exit(f"--task must be 0..{len(tasks) - 1}: " + ", ".join(f"{i}={task_tag(t)}" for i, t in enumerate(tasks)))
    sel = tasks if a.task is None else [tasks[a.task]]
    tag = "merge" if a.merge else ("all" if a.task is None else task_tag(sel[0]))
    _LOG_FH = open(os.path.join(out_dir, f"run_log_{tag}.txt"), "a", encoding="utf-8")
    log("=" * 78)
    log(f"{desc}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"output folder: {out_dir}")
    if a.merge:
        merge_parts(out_dir, prefix, keys, snr_keys); return
    dev = get_device(cfg)
    log(f"device {dev} | parts: {', '.join(task_tag(t) for t in sel)} | methods {cfg['methods']} | seeds {cfg['seeds']}")
    log("=" * 78)
    with open(os.path.join(out_dir, f"config_used_{tag}.json"), "w") as fh:
        json.dump(cfg, fh, indent=1, default=str)
    bd = AbBundle(cfg)
    for t in sel:
        log(f"\n[part {task_tag(t)}]")
        part = run_task(t, cfg, dev, bd)
        if part is None:
            continue
        _atomic_pickle(os.path.join(out_dir, "parts", f"{task_tag(t)}.pkl"), part)
        merge_parts(out_dir, prefix, keys, snr_keys)
    log(f"\nDone. Results in {out_dir}")


# ==========================================================================
#            SOTA REFERENCE (best SOTA allocator of the main run)
# ==========================================================================
class GreedyWMMSE(Allocator):
    label, ref = "Greedy+WMMSE", "Shi et al., IEEE TSP 2011 (best SOTA allocator of the main run)"

    def act(self, sb):
        ch, _ = greedy_assign(sb, self.cfg)
        return ch, wmmse_power(sb, ch, self.cfg)


REF_METHODS = {"greedy_wmmse": GreedyWMMSE}


# ==========================================================================
#                         ABLATION PART (allocation)
# ==========================================================================
def run_task_alloc(t, cfg, dev, bd):
    names, C, edges = bd.names, bd.C, bd.edges
    pool = gen_topologies(np.random.default_rng(cfg["seed"] + 999), cfg["topo_pool_test"], cfg)
    st_list = [scenario_stats(bd.D[s]["y_tr1"], bd.D[s]["E_tr1"], bd.eta, C) for s in names]
    ROWS, SNR, COMM, CX, CLI = [], [], [], {}, []
    Ksu = cfg["su_links"]
    pers = bd.preds("persistence", "non-learned", 0)
    U_cache = {}
    for part, K, mkey in task_configs(t, cfg):
        bm = bd.best(mkey)
        if bm is None:
            log(f"  [{mkey}] no forecaster in the hand-off (run the forecasting ablation first) -> skipped")
            continue
        mode = "centralized" if part == "pooled" else "fedavg"
        for seed in cfg["seeds"]:
            P = bd.preds(bm, mkey, seed)
            if P is None or any("tr1" not in P[s] for s in names):
                log(f"  [{mkey} seed {seed}] forecasts missing -> skipped")
                continue
            fsf = fit_fsf_all(bd, P, names, cfg, dev)            # per scenario (site), never communicated
            sources = [MemSource(s, i, P[s]["tr1"], bd.D[s]["y_tr1"], bd.D[s]["E_tr1"],
                                 bd.D[s]["inr_busy"], bd.D[s]["inr_idle"], cfg, dev) for i, s in enumerate(names)]
            clients = None
            if mode == "fedavg":
                clients = shard_sources(sources, part, K, seed, cfg)
                CLI += shard_rows(clients, names, part, K, seed)
            ms = {}
            for mk in cfg["methods"]:
                a = make_method(mk, cfg, dev, C, st_list, fsf)
                set_seed(run_seed(cfg, mk, mkey, seed))
                rs = run_seed(cfg, mk, mkey, seed, "rng")
                t0 = time.time()
                info = train_policy_k(a, clients, cfg, rs) if clients else train_policy(a, sources, "centralized", cfg, rs)
                ms[mk] = (a, info)
                if a.learned:
                    log(f"  trained {a.label:<10s} {mkey:<18s} seed {seed} ({time.time() - t0:.0f} s)")
                    COMM.append({"method": a.label, "mode": mode, "partition": part, "n_clients": K, "seed": seed,
                                 "rounds": info.get("rounds", ""), "local_steps_per_client": info.get("local_steps", ""),
                                 "comm_MB_total": info.get("comm_bytes", 0) / 1e6,
                                 "comm_MB_per_client_per_round": info.get("comm_bytes_per_client_per_round", 0) / 1e6,
                                 "train_s": info["train_s"], "note": "FSF fusion fitted per scenario, never communicated"})
            flab = f"{bd.label(bm)} ({mkey})"
            for h in cfg["horizons"]:
                if h > bd.H:
                    continue
                blk = test_blocks_mem(bd, lambda s, P=P: P[s]["te"], (lambda s: pers[s]["te"]) if pers else None,
                                      h, pool, cfg)
                L = blk["PH"].shape[1]
                if h not in U_cache:
                    U_cache[h] = oracle_utility(blk, cfg, dev)          # truth only: same for every configuration
                U_or = U_cache[h]
                for mk, (a, info) in ms.items():
                    if hasattr(a, "oracle_fc"):
                        a.fsf, a.oracle_fc = fsf, False
                    acc, sh, sn, lat = eval_method(a, blk, cfg, dev, edges, bd.eta)
                    for si, s in enumerate(names + (["Combined"] if len(names) > 1 else [])):
                        mask = np.ones(len(blk["SID"]), bool) if s == "Combined" else blk["SID"] == si
                        if not mask.any():
                            continue
                        r = {"method": a.label, "method_key": mk, "reference": a.ref,
                             "family": "novel" if mk in METHODS else "SOTA reference",
                             "trained": mode if a.learned else "non-learned", "mode": mode,
                             "partition": part, "n_clients": K,
                             "forecaster": flab, "forecaster_key": bm, "forecaster_config": mkey, "seed": seed,
                             "scenario": s, "horizon_step": h, **alloc_row(acc, mask, L, Ksu),
                             "decision_latency_ms_per_slot": lat * 1e3,
                             "train_s": info.get("train_s", 0.0), "comm_MB": info.get("comm_bytes", 0) / 1e6}
                        uo = float(U_or[mask].sum()) / (mask.sum() * L)
                        r["oracle_utility_per_slot"] = uo
                        r["regret_vs_oracle"] = 1 - r["utility_per_slot"] / uo if uo > 0 else float("nan")
                        ROWS.append(r)
                        bk = {k: r[k] for k in ("method", "method_key", "trained", "mode", "partition", "n_clients",
                                                "forecaster_config", "seed", "scenario", "horizon_step")}
                        SNR.append({**bk, "snr_bin": f"<= {bd.eta:g} (idle)",
                                    "n_cells": float(acc["n_idle"][mask].sum()),
                                    "idle_utilization": r["idle_utilization"]})
                        for i in range(len(edges) - 1):
                            n_ = float(sn[mask, i].sum())
                            SNR.append({**bk, "snr_bin": f"{edges[i]:g}-{edges[i + 1]:g}", "n_cells": n_,
                                        "pu_hit_rate": float(sh[mask, i].sum()) / n_ if n_ else float("nan")})
                log(f"  evaluated {len(ms)} methods | {flab} | h={h} | seed {seed}")
            for mk, (a, _) in ms.items():
                if mk not in CX:
                    CX[mk] = profile_novel_policy(a, C, cfg) if hasattr(a, "setup") else profile_policy(a, C, cfg)
    if not ROWS:
        log("  nothing evaluated in this part")
        return None
    return {"ROWS": ROWS, "SNR": SNR, "COMM": COMM, "CX": CX, "CLI": CLI}


if __name__ == "__main__":
    _keys = ["method", "method_key", "family", "trained", "mode", "partition", "n_clients", "scenario", "horizon_step"]
    ab_main("ablation_clients_allocation_2p4GHz", "alloc", run_task_alloc, _keys,
            ["method", "method_key", "trained", "mode", "partition", "n_clients", "scenario", "horizon_step", "snr_bin"],
            {**METHODS, **REF_METHODS})
