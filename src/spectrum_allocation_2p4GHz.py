#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 spectrum_allocation_2p4GHz.py                       (script 2 of 3)
 Forecast-driven SPECTRUM (RESOURCE) ALLOCATION, 2.4 GHz ISM.
 Who transmits on which 2.5 MHz sub-channel and with what power, for K
 secondary links, in the next slot -- maximising sum-rate while avoiding
 sub-channels the primary users (measured Wi-Fi / BT / other ISM traffic)
 will occupy. Per scenario (IEEE, Office, Home, Campus) and combined.
 Centralised and FEDERATED (FedAvg, one client per scenario) training.
 Single file, runs with no arguments. Deterministic seeds. CSV output.

 INPUT  <forecast_dir>/stage2/  written by spectrum_occupancy_forecasting_2p4GHz.py
        (measured labels + forecasts). If it is missing the script stops and
        says so; it never generates spectrum data.
 SIMULATED: only the secondary network (positions / path loss of the K SU
 links, seeded, identical for every method). Primary-user occupancy and
 interference are the MEASURED future of each test slot.

 PROBLEM (one slot)
   choose ch_k in {0..C-1, idle}, p_k in [0, 1] for k = 1..K
   SINR_k = p_k G_kk / (1 + INR_{ch_k} + sum_{j on ch_k} p_j G_jk)
   utility = sum_k log2(1 + SINR_k) - lambda * #(SU transmissions on a
             sub-channel that is truly occupied)          (lambda = 2 bit/s/Hz)
   decisions see only the forecast phat and the expected INR.

 METHODS
   Random                          lower reference
   Greedy-forecast                 marginal-utility channel assignment
   Oracle (true future)            greedy + WMMSE with the true future, upper reference
   Greedy+WMMSE                    Shi, Razaviyayn, Luo, He, IEEE TSP 2011
   Greedy+UWMMSE                   Chowdhury, Verma, Rao, Swami, Segarra, IEEE TWC 2021
   REGNN                           Eisen & Ribeiro, IEEE TSP 2020
   WCGCN (MPGNN)                   Shen, Shi, Zhang, Letaief, IEEE JSAC 2021
   Agg-GNN (decentralised)         Wang, Eisen, Ribeiro, arXiv:2107.01489
   Diffusion-GNN                   Uslu, Hadou, Saeedi Bidokhti, Ribeiro,
                                   arXiv:2504.20277 / NeurIPS 2025 workshop
 Forecasts used: oracle, historical mean, and the best learned forecaster of
 each training mode (chosen on validation Brier by script 1).

 OUTPUT (CSV)
   alloc_overall.csv / _agg.csv  method x forecaster x mode x scenario x horizon
                                 sum-rate, rate/link, utility, collisions, PU
                                 harm, idle-channel utilisation, access ratio,
                                 outage, Jain fairness, energy efficiency,
                                 regret vs oracle, decision latency
   alloc_per_snr.csv             PU hit rate per measured PU-SNR bin
                                 [6,8), [8,10), ... and idle utilisation
   alloc_complexity.csv          params, FLOPs per slot, latency per slot
   fl_communication.csv, run_log.txt, config_used.json
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
# Narval (Alliance Canada) layout is used automatically when it exists;
# otherwise the Windows paths. Command-line options override both.
# ---- paths -------------------------------------------------------------
# Defaults follow the repository layout: data/ and results/ next to src/.
# Override with the environment variables SPECTRUM_DATA, SPECTRUM_RESULTS
# (and SPECTRUM_PROJECT for relative paths) or with the command-line options.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NARVAL = os.environ.get("SPECTRUM_PROJECT", _REPO)   # project root (name kept from the cluster runs)
_ON_NARVAL = False
_RES = os.environ.get("SPECTRUM_RESULTS", os.path.join(_REPO, "results"))
DATA_ROOT = os.environ.get("SPECTRUM_DATA", os.path.join(_REPO, "data"))
FORECAST_DIR = os.path.join(_RES, "Forecasting")
OUT_DIR = os.path.join(_RES, "Allocation")

CONFIG: Dict = {
    "forecast_dir": FORECAST_DIR,
    "out_dir": OUT_DIR,
    "methods": ["random", "greedy_forecast", "oracle", "greedy_wmmse", "greedy_uwmmse",
                "regnn", "wcgcn", "agg_gnn", "diffusion_gnn"],
    "forecasters": ["oracle", "historical_mean", "best"],
    "modes": ["centralized", "fedavg"],
    "seeds": [0, 1, 2],
    "seed": 20261019,
    "horizons": [1, 4, 8],           # forecast step used for the slot
    # ---- secondary network (simulated, seeded) ----------------------------
    "su_links": 48,                  # K > 32 sub-channels -> reuse needed
    "su_area_m": 15.0,               # dense indoor deployment: SU-SU interference matters
    "su_link_dist_m": (2.0, 8.0),
    "su_pathloss_exp": 3.0,
    "su_shadow_db": 4.0,
    "su_snr_ref_db": 40.0,           # receive SNR at 1 m, full power
    "collision_penalty": 2.0,        # bit/s/Hz per SU transmission on a busy sub-channel
    "tx_power_min": 1e-3,
    "topo_pool_test": 64, "topo_block": 50,
    # ---- training ----------------------------------------------------------
    "train_steps": 1500, "alloc_batch": 128, "alloc_lr": 1e-3,
    "fl_rounds": 30, "fl_local_steps": 50,
    "gnn_hidden": 64, "gnn_layers": 3, "regnn_taps": 3, "agg_hops": 3,
    "gnn_decode_samples": 16,        # sampled assignments scored on the forecast
    "wmmse_iters": 30, "uwmmse_layers": 4,
    "diff_T": 100, "diff_sample_steps": 10, "diff_samples": 8,
    # ---- profiling ---------------------------------------------------------
    "prof_iters": 50, "prof_warmup": 5,
    "device": "auto",
    "verbose": True,
}

QUICK = {"train_steps": 20, "fl_rounds": 2, "fl_local_steps": 5, "seeds": [0],
         "horizons": [1], "diff_T": 10, "diff_sample_steps": 3, "diff_samples": 2,
         "prof_iters": 3, "prof_warmup": 1}

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
class Bundle:
    """Measured labels + forecasts written by script 1. Nothing is generated."""

    def __init__(self, cfg: Dict):
        d = os.path.join(cfg["forecast_dir"], "stage2")
        mp = os.path.join(d, "manifest.json")
        if not os.path.isfile(mp):
            log(f"\nNO FORECAST HAND-OFF FOUND at\n  {d}\n"
                "Run spectrum_occupancy_forecasting_2p4GHz.py first (it writes stage2/),\n"
                "or set CONFIG['forecast_dir'] / --forecast-dir to its out_dir.\n"
                "Nothing was run and no data was invented.")
            sys.exit(1)
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
        for attr in ("abar", "tau"):
            if hasattr(a, attr) and torch.is_tensor(getattr(a, attr)):
                setattr(b, attr, getattr(a, attr).to(dv))
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
#                      ALLOCATION OUTCOME (measured future)
# ==========================================================================
def evaluate_alloc(sb: SlotBatch, ch, pw, cfg: Dict) -> Dict:
    tx, chc, A, p, sinr, rate = su_side(sb, ch, pw, cfg)
    col = tx & (torch.gather(sb.y, 1, chc) > 0.5)
    sr = rate.sum(1)
    return {"rate": rate, "tx": tx, "col": col, "hit": A.sum(1) > 0, "busy": sb.y > 0.5,
            "sinr": sinr, "sum_rate": sr, "pw": p,
            "utility": sr - cfg["collision_penalty"] * col.float().sum(1)}


def soft_utility(sb: SlotBatch, logits, plog, cfg: Dict):
    """Differentiable mean-field utility of a soft allocation (training only;
    the reward uses the measured truth of TRAINING slots)."""
    K, C = sb.K, sb.C
    pi = torch.softmax(logits, -1)[..., :C]
    p = torch.sigmoid(plog)
    Gx = sb.G * (1 - torch.eye(K, device=sb.dev))
    I = torch.einsum("bjc,bj,bjk->bkc", pi, p, Gx)
    S = p * torch.diagonal(sb.G, dim1=1, dim2=2)
    r = torch.log2(1 + S[..., None] / (1 + sb.inr[:, None, :] + I))
    return (pi * (r - cfg["collision_penalty"] * sb.y[:, None, :])).sum((1, 2))


# ==========================================================================
#                         LEARNED ALLOCATION POLICIES
# ==========================================================================
class REGNNPolicy(nn.Module):
    """Eisen & Ribeiro, 'Optimal Wireless Resource Allocation With Random Edge
    Graph Neural Networks', IEEE TSP 2020. Layers are graph filters
    sum_t S^t Z A_t with the (random) channel matrix as graph shift S;
    trained without labels on the utility."""

    def __init__(self, fin, C, h=64, layers=3, taps=3):
        super().__init__()
        dims = [fin] + [h] * layers
        self.filt = nn.ModuleList([nn.ModuleList([nn.Linear(dims[l], dims[l + 1], bias=(t == 0))
                                                  for t in range(taps)]) for l in range(layers)])
        self.ch_head, self.p_head = nn.Linear(h, C + 1), nn.Linear(h, 1)

    def forward(self, sb: SlotBatch):
        z = sb.node_feats()
        S = torch.log1p(sb.G * (1 - torch.eye(sb.K, device=sb.dev)))
        St = (S / S.sum(1, keepdim=True).clamp(min=1e-6)).transpose(1, 2)
        for taps in self.filt:
            acc, zt = 0, z
            for t, lin in enumerate(taps):
                if t > 0:
                    zt = St @ zt
                acc = acc + lin(zt)
            z = F.relu(acc)
        return self.ch_head(z), self.p_head(z).squeeze(-1)


class WCGCNPolicy(nn.Module):
    """Shen, Shi, Zhang, Letaief, 'Graph Neural Networks for Scalable Radio
    Resource Management: Architecture Design and Theoretical Analysis', IEEE
    JSAC 2021 (MPGNN / WCGCN). Message passing with edge features in both
    directions and MAX aggregation."""

    def __init__(self, fin, C, h=64, layers=3):
        super().__init__()
        self.inp = nn.Linear(fin, h)
        self.msg = nn.ModuleList([nn.Sequential(nn.Linear(h + 2, h), nn.ReLU(), nn.Linear(h, h))
                                  for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, h))
                                  for _ in range(layers)])
        self.ch_head, self.p_head = nn.Linear(h, C + 1), nn.Linear(h, 1)

    def embed(self, sb: SlotBatch, z=None):
        B, K = sb.B, sb.K
        z = F.relu(self.inp(sb.node_feats())) if z is None else z
        e = sb.edge_feats()
        eye = torch.eye(K, device=sb.dev, dtype=torch.bool)
        ef = torch.stack([e, e.transpose(1, 2)], -1)
        for mlp, upd in zip(self.msg, self.upd):
            m = mlp(torch.cat([z[:, :, None, :].expand(B, K, K, z.shape[-1]), ef], -1))
            m = m.masked_fill(eye[None, :, :, None], -1e9).max(1).values
            z = z + upd(torch.cat([z, m], -1))
        return z

    def forward(self, sb: SlotBatch):
        z = self.embed(sb)
        return self.ch_head(z), self.p_head(z).squeeze(-1)


class AggGNNPolicy(nn.Module):
    """Wang, Eisen, Ribeiro, 'Learning Decentralized Wireless Resource
    Allocations with Graph Neural Networks', arXiv:2107.01489. Each link
    builds an AGGREGATION SEQUENCE [x, Sx, S^2x, ...] from neighbour exchanges
    only (decentralised), then a local network shared by all links maps it to
    its own decision."""

    def __init__(self, fin, C, h=64, hops=3):
        super().__init__()
        self.hops = hops
        self.local = nn.Sequential(nn.Linear(fin * (hops + 1), 2 * h), nn.ReLU(),
                                   nn.Linear(2 * h, h), nn.ReLU())
        self.ch_head, self.p_head = nn.Linear(h, C + 1), nn.Linear(h, 1)

    def forward(self, sb: SlotBatch):
        x = sb.node_feats()
        S = torch.log1p(sb.G * (1 - torch.eye(sb.K, device=sb.dev)))
        St = (S / S.sum(1, keepdim=True).clamp(min=1e-6)).transpose(1, 2)
        seq, z = [x], x
        for _ in range(self.hops):
            z = St @ z
            seq.append(z)
        h = self.local(torch.cat(seq, -1))
        return self.ch_head(h), self.p_head(h).squeeze(-1)


class UWMMSENet(nn.Module):
    """Chowdhury, Verma, Rao, Swami, Segarra, 'Unfolding WMMSE Using Graph
    Neural Networks for Efficient Power Allocation', IEEE TWC 2021.
    L unrolled WMMSE iterations; a 2-layer GCN on the co-channel graph sets
    per-link (a, b) in w <- a * w + b at every iteration."""

    def __init__(self, layers=4, h=16):
        super().__init__()
        self.gcn = nn.ModuleList([nn.ModuleList([nn.Linear(3, h), nn.Linear(h, 2)])
                                  for _ in range(layers)])

    def forward(self, sb: SlotBatch, ch, noise):
        C = sb.C
        tx = (ch < C).float()
        chc = ch.clamp(max=C - 1)
        A = F.one_hot(chc, C).float() * tx[..., None]
        h2 = sb.G * (A @ A.transpose(1, 2))
        hkk = torch.sqrt(torch.diagonal(sb.G, dim1=1, dim2=2))
        Ah = torch.log1p(h2) * (1 - torch.eye(sb.K, device=sb.dev))
        Ah = (Ah / Ah.sum(1, keepdim=True).clamp(min=1e-6)).transpose(1, 2)
        x = torch.stack([torch.log10(hkk ** 2 + 1e-9) / 4, torch.log10(noise) / 2, tx], -1)
        v = tx.clone()
        for l1, l2 in self.gcn:
            ab = F.softplus(l2(Ah @ F.relu(l1(Ah @ x))))
            u = hkk * v / (noise + torch.einsum("bjk,bj->bk", h2, v ** 2))
            w = ab[..., 0] / (1.0 - u * hkk * v).clamp(min=1e-6) + ab[..., 1]
            den = torch.einsum("bkj,bj->bk", h2, w * u ** 2)
            v = (w * u * hkk / den.clamp(min=1e-12)).clamp(0, 1) * tx
        return v ** 2


class DiffusionDenoiser(nn.Module):
    """Uslu, Hadou, Saeedi Bidokhti, Ribeiro, 'Generative Diffusion Models for
    Resource Allocation in Wireless Networks', arXiv:2504.20277 (NeurIPS 2025
    workshop: 'GNN-Parametrized Diffusion Policies'). eps-prediction network:
    WCGCN-style message passing conditioned on forecast and diffusion time."""

    def __init__(self, fin, C, h=64, layers=3):
        super().__init__()
        self.dx = C + 2
        self.temb = nn.Sequential(nn.Linear(16, h), nn.SiLU(), nn.Linear(h, h))
        self.inp = nn.Linear(fin + self.dx, h)
        self.gnn = WCGCNPolicy(fin, C, h, layers)
        self.out = nn.Linear(h, self.dx)

    @staticmethod
    def sincos(t, dim=16):
        f = torch.exp(-math.log(1000.0) * torch.arange(dim // 2, device=t.device) / (dim // 2))
        a = t[:, None].float() * f[None]
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(self, sb: SlotBatch, xt, t):
        z = F.relu(self.inp(torch.cat([sb.node_feats(), xt], -1))
                   + self.temb(self.sincos(t))[:, None, :])
        return self.out(self.gnn.embed(sb, z))


# ==========================================================================
#                                 METHODS
# ==========================================================================
class RandomAlloc(Allocator):
    label, ref = "Random", "lower reference"

    def act(self, sb):
        g = torch.Generator(device="cpu").manual_seed(int(sb.phat.sum().item() * 1000) % (2 ** 31))
        return (torch.randint(0, sb.C, (sb.B, sb.K), generator=g).to(sb.dev),
                torch.ones(sb.B, sb.K, device=sb.dev))


class GreedyForecast(Allocator):
    label, ref = "Greedy-forecast", "marginal-utility assignment on the forecast"

    def act(self, sb):
        return greedy_assign(sb, self.cfg)


class OracleAlloc(Allocator):
    label, ref = "Oracle (true future)", "upper reference, not causal"

    def act(self, sb):
        ch, _ = greedy_assign(sb, self.cfg, oracle=True)
        return ch, wmmse_power(sb, ch, self.cfg,
                               noise=1.0 + torch.gather(sb.inr, 1, ch.clamp(max=sb.C - 1)))


class GreedyWMMSE(Allocator):
    label, ref = "Greedy+WMMSE", "Shi et al., IEEE TSP 2011"

    def act(self, sb):
        ch, _ = greedy_assign(sb, self.cfg)
        return ch, wmmse_power(sb, ch, self.cfg)


class GreedyUWMMSE(Allocator):
    learned = True
    label, ref = "Greedy+UWMMSE", "Chowdhury et al., IEEE TWC 2021"

    def __init__(self, cfg, dev, C):
        super().__init__(cfg, dev, C)
        self.nets["net"] = UWMMSENet(cfg["uwmmse_layers"]).to(dev)

    def act(self, sb):
        ch, _ = greedy_assign(sb, self.cfg)
        with torch.no_grad():
            return ch, self.nets["net"](sb, ch, 1.0 + torch.gather(sb.inr_hat, 1, ch.clamp(max=sb.C - 1)))

    def train_steps(self, src, n, rng, client=0):
        net = self.nets["net"]; net.train(); opt = self.opt_for("net", net)
        for _ in range(n):
            sb = src.batch(rng, self.cfg["alloc_batch"])
            ch, _ = greedy_assign(sb, self.cfg)
            pw = net(sb, ch, 1.0 + torch.gather(sb.inr_hat, 1, ch.clamp(max=sb.C - 1)))
            loss = -evaluate_alloc(sb, ch, pw, self.cfg)["sum_rate"].mean() / sb.K
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()


class GNNAlloc(Allocator):
    learned = True
    arch = "wcgcn"

    def __init__(self, cfg, dev, C):
        super().__init__(cfg, dev, C)
        fin, h = 2 * C + 1, cfg["gnn_hidden"]
        net = {"regnn": lambda: REGNNPolicy(fin, C, h, cfg["gnn_layers"], cfg["regnn_taps"]),
               "agg": lambda: AggGNNPolicy(fin, C, h, cfg["agg_hops"]),
               "wcgcn": lambda: WCGCNPolicy(fin, C, h, cfg["gnn_layers"])}[self.arch]()
        self.nets["net"] = net.to(dev)
        self.steps_done = 0

    def act(self, sb):
        """Decode the soft policy: the argmax plus N samples of the per-link
        sub-channel distribution; keep the candidate with the best utility
        PREDICTED FROM THE FORECAST (a hard argmax alone lets links with
        similar features collapse onto one sub-channel)."""
        net = self.nets["net"]; net.eval()
        with torch.no_grad():
            lg, pl = net(sb)
            pw = torch.sigmoid(pl)
            N = int(self.cfg.get("gnn_decode_samples", 16))
            g = torch.Generator(device=sb.dev).manual_seed(4321)
            cands = [lg.argmax(-1)]
            prob = torch.softmax(lg, -1).reshape(-1, lg.shape[-1])
            for _ in range(N):
                cands.append(torch.multinomial(prob, 1, generator=g).reshape(lg.shape[:2]))
            U = torch.stack([expected_utility(sb, c, pw, self.cfg) for c in cands], 1)
            best = U.argmax(1)
            ch = torch.stack(cands, 1)[torch.arange(sb.B, device=sb.dev), best]
        return ch, pw

    def train_steps(self, src, n, rng, client=0):
        net = self.nets["net"]; net.train(); opt = self.opt_for("net", net)
        tot = max(1, self.cfg["train_steps"])
        for _ in range(n):
            sb = src.batch(rng, self.cfg["alloc_batch"])
            lg, pl = net(sb)
            ent = -(torch.softmax(lg, -1) * torch.log_softmax(lg, -1)).sum(-1).mean()
            beta = 0.05 * max(0.0, 1 - self.steps_done / tot)
            loss = -soft_utility(sb, lg, pl, self.cfg).mean() / sb.K - beta * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            self.steps_done += 1


class REGNNAlloc(GNNAlloc):
    arch = "regnn"
    label, ref = "REGNN", "Eisen & Ribeiro, IEEE TSP 2020"


class WCGCNAlloc(GNNAlloc):
    arch = "wcgcn"
    label, ref = "WCGCN (MPGNN)", "Shen et al., IEEE JSAC 2021"


class AggGNNAlloc(GNNAlloc):
    arch = "agg"
    label, ref = "Agg-GNN (decentralised)", "Wang, Eisen, Ribeiro, arXiv:2107.01489"


class DiffusionGNN(Allocator):
    learned = True
    label, ref = "Diffusion-GNN", "Uslu et al., arXiv:2504.20277 (NeurIPS 2025 workshop)"

    def __init__(self, cfg, dev, C):
        super().__init__(cfg, dev, C)
        self.nets["eps"] = DiffusionDenoiser(2 * C + 1, C, cfg["gnn_hidden"], cfg["gnn_layers"]).to(dev)
        self.abar = torch.cumprod(1 - torch.linspace(1e-4, 0.02, cfg["diff_T"], device=dev), 0)

    def _x0(self, sb):
        ch, _ = greedy_assign(sb, self.cfg, oracle=True)          # near-optimal expert
        pw = wmmse_power(sb, ch, self.cfg, noise=1.0 + torch.gather(sb.inr, 1, ch.clamp(max=sb.C - 1)))
        return torch.cat([F.one_hot(ch, sb.C + 1).float() * 2 - 1, (pw * 2 - 1)[..., None]], -1)

    def train_steps(self, src, n, rng, client=0):
        net = self.nets["eps"]; net.train(); opt = self.opt_for("eps", net)
        T = self.cfg["diff_T"]
        for _ in range(n):
            sb = src.batch(rng, self.cfg["alloc_batch"])
            with torch.no_grad():
                x0 = self._x0(sb)
            t = torch.as_tensor(rng.integers(0, T, sb.B), device=sb.dev)
            ab = self.abar[t][:, None, None]
            eps = torch.randn_like(x0)
            loss = F.mse_loss(net(sb, ab.sqrt() * x0 + (1 - ab).sqrt() * eps, t), eps)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()

    @torch.no_grad()
    def act(self, sb):
        net = self.nets["eps"]; net.eval()
        N, T = self.cfg["diff_samples"], self.cfg["diff_T"]
        sr = sb_repeat(sb, N)
        g = torch.Generator(device=sb.dev).manual_seed(1234)
        x = torch.randn(sr.B, sr.K, sb.C + 2, device=sb.dev, generator=g)
        ts = torch.linspace(T - 1, 0, self.cfg["diff_sample_steps"], device=sb.dev).long()
        for i, t in enumerate(ts):                       # DDIM, eta = 0
            ab = self.abar[t]
            ab_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=sb.dev)
            e = net(sr, x, t.repeat(sr.B))
            x0 = ((x - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-1, 1)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * e
        ch = x[..., :sb.C + 1].argmax(-1)
        pw = ((x[..., -1] + 1) / 2).clamp(0, 1)
        best = expected_utility(sr, ch, pw, self.cfg).reshape(sb.B, N).argmax(1)
        ar = torch.arange(sb.B, device=sb.dev)
        return ch.reshape(sb.B, N, sb.K)[ar, best], pw.reshape(sb.B, N, sb.K)[ar, best]


METHODS = {"random": RandomAlloc, "greedy_forecast": GreedyForecast, "oracle": OracleAlloc,
           "greedy_wmmse": GreedyWMMSE, "greedy_uwmmse": GreedyUWMMSE, "regnn": REGNNAlloc,
           "wcgcn": WCGCNAlloc, "agg_gnn": AggGNNAlloc, "diffusion_gnn": DiffusionGNN}


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
        sb = block_slot(blk, t, dev)
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
            "rate_per_link": S("sum_rate") / sl / K,
            "rate_per_active_link": S("sum_rate") / ntx,
            "utility_per_slot": S("utility") / sl,
            "collisions_per_slot": S("n_col") / sl,
            "collision_rate": S("n_col") / ntx,
            "pu_harm_rate": S("hit_busy") / max(S("n_busy"), 1.0),
            "idle_utilization": S("hit_idle") / max(S("n_idle"), 1.0),
            "access_ratio": S("n_tx") / (sl * K),
            "outage_rate": S("outage") / ntx,
            "jain_fairness": float(acc["jain"][mask].mean()),
            "energy_eff_bit_per_unit_power": S("sum_rate") / max(S("sum_pw"), 1e-9)}


def run(cfg, dev, bd: Bundle, out_dir):
    names, C, edges = bd.names, bd.C, bd.edges
    G_pool = gen_topologies(np.random.default_rng(cfg["seed"] + 999), cfg["topo_pool_test"], cfg)
    ROWS, SNR, CX, COMM = [], [], {}, []
    order = ["oracle"] + [m for m in cfg["methods"] if m != "oracle"]
    K = cfg["su_links"]
    pers = bd.preds("persistence", "non-learned", 0)
    for seed in cfg["seeds"]:
        for mode in cfg["modes"]:
            bm = bd.man["best_by_mode"].get(mode)
            P = bd.preds(bm, mode, seed) if bm else None
            if P is None or any("tr1" not in P[s] for s in names):
                log(f"  [{mode} seed {seed}] no forecasts in the hand-off -> skipped")
                continue
            sources = [SlotSource(s, P[s]["tr1"], bd.D[s]["y_tr1"], bd.D[s]["E_tr1"],
                                  bd.D[s]["inr_busy"], bd.D[s]["inr_idle"], cfg, dev) for s in names]
            ms = {}
            for mk in order:
                a = METHODS[mk](cfg, dev, C); a.key = mk
                set_seed(run_seed(cfg, mk, mode, seed))
                t0 = time.time()
                info = train_policy(a, sources, mode, cfg, run_seed(cfg, mk, mode, seed, "rng"))
                ms[mk] = (a, info)
                if a.learned:
                    log(f"  trained {a.label:<26s} {mode:<11s} seed {seed} ({time.time() - t0:.0f} s)")
                    COMM.append({"method": a.label, "mode": mode, "seed": seed,
                                 "comm_MB": info.get("comm_bytes", 0) / 1e6,
                                 "rounds": info.get("rounds", ""), "train_s": info["train_s"]})
            for fc in cfg["forecasters"]:
                if fc == "oracle":
                    phat_of, flab = (lambda s: bd.D[s]["y_te"]), "Oracle forecast"
                elif fc == "best":
                    phat_of, flab = (lambda s, P=P: P[s]["te"]), f"{bd.label(bm)} ({mode})"
                else:
                    Pb = bd.preds(fc, "non-learned", 0)
                    if Pb is None:
                        log(f"  forecaster {fc} not in the hand-off -> skipped"); continue
                    phat_of, flab = (lambda s, Pb=Pb: Pb[s]["te"]), bd.label(fc)
                for h in cfg["horizons"]:
                    if h > bd.H:
                        continue
                    blk = test_blocks(bd, phat_of, (lambda s: pers[s]["te"]) if pers else None,
                                      h, G_pool, cfg)
                    L = blk["PH"].shape[1]
                    U_or = {}
                    for mk in order:
                        a, info = ms[mk]
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
                                 "train_s": info.get("train_s", 0.0),
                                 "comm_MB": info.get("comm_bytes", 0) / 1e6}
                            if mk == "oracle":
                                U_or[s] = r["utility_per_slot"]
                            uo = U_or.get(s, float("nan"))
                            r["regret_vs_oracle"] = 1 - r["utility_per_slot"] / uo if uo and uo > 0 else float("nan")
                            ROWS.append(r)
                            bk = {k: r[k] for k in ("method", "method_key", "trained", "mode",
                                                    "forecaster", "forecaster_key", "seed",
                                                    "scenario", "horizon_step")}
                            SNR.append({**bk, "snr_bin": f"<= {bd.eta:g} (idle)",
                                        "n_cells": float(acc["n_idle"][mask].sum()),
                                        "idle_utilization": r["idle_utilization"]})
                            for i in range(len(edges) - 1):
                                n_ = float(sn[mask, i].sum())
                                SNR.append({**bk, "snr_bin": f"{edges[i]:g}-{edges[i + 1]:g}",
                                            "n_cells": n_,
                                            "pu_hit_rate": float(sh[mask, i].sum()) / n_ if n_ else float("nan")})
                    log(f"  evaluated {len(order)} methods | {flab} | h={h} | {mode} | seed {seed}")
            for mk, (a, _) in ms.items():
                if mk not in CX:
                    CX[mk] = profile_policy(a, C, cfg)
    if not ROWS:
        log("  nothing evaluated (see messages above)")
        return
    write_csv(os.path.join(out_dir, "alloc_overall.csv"), ROWS)
    write_csv(os.path.join(out_dir, "alloc_overall_agg.csv"),
              aggregate(ROWS, ["method", "method_key", "trained", "mode", "forecaster",
                               "forecaster_key", "scenario", "horizon_step"]))
    write_csv(os.path.join(out_dir, "alloc_per_snr.csv"),
              aggregate(SNR, ["method", "method_key", "trained", "mode", "forecaster",
                              "forecaster_key", "scenario", "horizon_step", "snr_bin"]))
    write_csv(os.path.join(out_dir, "alloc_complexity.csv"), list(CX.values()))
    write_csv(os.path.join(out_dir, "fl_communication.csv"), COMM)


# ==========================================================================
#                                   MAIN
# ==========================================================================
def main() -> None:
    global _LOG_FH
    ap = argparse.ArgumentParser(description="2.4 GHz forecast-driven spectrum allocation")
    ap.add_argument("--forecast-dir", help="out_dir of spectrum_occupancy_forecasting_2p4GHz.py")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--seeds"); ap.add_argument("--methods"); ap.add_argument("--modes")
    ap.add_argument("--device"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = CONFIG
    if a.forecast_dir:
        cfg["forecast_dir"] = a.forecast_dir
    if a.out:
        cfg["out_dir"] = a.out
    if a.seeds:
        cfg["seeds"] = [int(x) for x in a.seeds.split(",")]
    if a.methods:
        cfg["methods"] = [x.strip() for x in a.methods.split(",")]
    if a.modes:
        cfg["modes"] = [x.strip() for x in a.modes.split(",")]
    if a.device:
        cfg["device"] = a.device
    if a.quick:
        cfg.update(QUICK)
    bad = [m for m in cfg["methods"] if m not in METHODS]
    if bad:
        sys.exit(f"unknown method(s): {bad}; choose from {list(METHODS)}")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    _LOG_FH = open(os.path.join(cfg["out_dir"], "run_log.txt"), "a", encoding="utf-8")
    dev = get_device(cfg)
    log("=" * 78)
    log(f"spectrum_allocation_2p4GHz  |  device {dev}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 78)
    with open(os.path.join(cfg["out_dir"], "config_used.json"), "w") as fh:
        json.dump(cfg, fh, indent=1, default=str)
    bd = Bundle(cfg)
    run(cfg, dev, bd, cfg["out_dir"])
    log(f"\nDone. Results in {cfg['out_dir']}")


if __name__ == "__main__":
    main()
