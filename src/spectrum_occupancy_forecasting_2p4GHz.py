#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 spectrum_occupancy_forecasting_2p4GHz.py          (script 1 of 3)
 Spectrum OCCUPANCY FORECASTING, 2.4 GHz ISM, four measured scenarios
 (IEEE event, Office, Home, Campus), reported per scenario AND combined.
 Local, centralised and FEDERATED (FedAvg / FedProx) training.
 Single file, runs with no arguments. Deterministic seeds. CSV output.
 It also writes <out_dir>/stage2/ : the forecasts and measured labels read by
   spectrum_allocation_2p4GHz.py   (script 2: resource allocation)
   spectrum_sharing_2p4GHz.py      (script 3: PU-SU spectrum sharing)

 NO SYNTHETIC DATA. Every label, PSD value and SNR comes from your exported
 sessions. A missing scenario is PRINTED and skipped, never fabricated.

 ---------------------------------------------------------------- INPUT
 Per scenario, any of:
   <session>/export/<session>.h5   written by build_wifi_dataset.py
                                   (uses the gap-handled /grid/ISM_2400 grid)
   <session>.h5                    same file given directly
   <session>/                      raw session (sweeps_*.csv + psd_*.f16)
 Auto-discovery: every folder / .h5 under CONFIG["data_root"] whose path
 contains a scenario alias (ieee|event, office, home, campus) and a 2.4 GHz
 marker (wifi_2p4GHz | wifi24g | 2p4). Override on the command line:
   python spectrum_occupancy_forecasting_2p4GHz.py --data IEEE="D:\...\x.h5" --data Home="D:\...\home_wifi24g_..."

 ---------------------------------------------------------------- LABELS
   sub-channels     2.5 MHz (32 over the common 80 MHz span of all scenarios)
                    power averaged in the LINEAR domain over the 100 kHz bins
   noise floor      10th percentile (Q10) of each sub-channel's power over the
                    TRAINING span of that scenario only (no test leakage)
   occupied         excess = P - noise_floor > eta = 6 dB
   SNR of a cell    its excess (dB over the noise floor); per-SNR tables use
                    2 dB bins [6,8), [8,10), ... up to the measured maximum n
 One forecasting step = one sweep (median sweep period printed per scenario).

 ---------------------------------------------------------------- MODELS
 Forecast occupancy of the next H=8 steps from the last W=32 steps.
   Baselines   Persistence, Seasonal naive (ACF period), Historical mean
   Time series DLinear [Zeng, AAAI'23], PatchTST [Nie, ICLR'23],
               iTransformer [Liu, ICLR'24], Crossformer [Zhang, ICLR'23],
               N-HiTS [Challu, AAAI'23]
   Spectrum    NN-ResNet [Ren et al., IEEE TCCN'22] CNN + ResNet
   prediction  Bi-ConvLSTM-S2S [Wang et al., Sensors'24] ConvLSTM seq2seq
               TSB [arXiv:2405.19138, 2024] Transformer + stacked Bi-LSTM
   Spectrum    DeepSense-F [Uvaydov et al., INFOCOM'21]: the DeepSense CNN
   sensing /   applied to the time x sub-channel window (forecasting head)
   foundation  SpectrumFM-F [Liu et al., GLOBECOM'25]: conv + multi-head
               self-attention encoder, self-supervised pre-training by masked
               reconstruction + next-slot prediction, then fine-tuned
 Training modes
   local        one model per scenario, trained and tested on that scenario
   centralized  one model on the pooled training data of all scenarios
   fedavg       FedAvg [McMahan, AISTATS'17], one client per scenario
   fedprox      FedProx [Li, MLSys'20], same clients, proximal mu
 Clients are the scenarios themselves: real non-IID sites, no data sharing.

 ---------------------------------------------------------------- OUTPUT (CSV)
   dataset_card.csv                 per scenario
   forecast_overall.csv / _agg.csv  every model x mode x scenario(+Combined)
   forecast_per_horizon.csv
   forecast_per_snr_cell.csv        cells binned by their own SNR
   forecast_per_snr_window.csv      (window, sub-channel) units binned by
                                    their mean target SNR, full metric set
   forecast_complexity.csv          params, FLOPs, latency (CPU, GPU if any)
   fl_communication.csv
   stage2/                          inputs of scripts 2 and 3
   run_log.txt, config_used.json, cache/ (resume)
================================================================================
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import hashlib
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
OUT_DIR = os.path.join(_RES, "Forecasting")

CONFIG: Dict = {
    "data_root": DATA_ROOT,
    "out_dir": OUT_DIR,
    # file name (relative to data_root) or full path of each scenario's .h5 /
    # session folder; if it does not exist, or is "", auto-discovery is used
    "scenarios": {"IEEE": "ieee_event.h5", "Office": "office.h5",
                  "Home": "home.h5", "Campus": "indoor_campus.h5"},
    "scenario_aliases": {"IEEE": ["ieee", "event"], "Office": ["office"],
                         "Home": ["home"], "Campus": ["campus"]},
    "band_name": "ISM_2400",

    # ---- labels ----------------------------------------------------------
    "subchannel_hz": 2.5e6,
    "noise_quantile": 0.10,          # 10 % noise floor
    "eta_db": 6.0,                   # occupied = excess > 6 dB
    "snr_bin_start_db": 6.0,
    "snr_bin_width_db": 2.0,         # [6,8), [8,10), ... [n-2, n]

    # ---- windows / split --------------------------------------------------
    "lookback": 32, "horizon": 8, "stride": 1,
    "val_frac": 0.15, "test_frac": 0.25,   # chronological, gap = W + H
    "input_repr": "binary",          # "binary" | "excess" ((e - eta)/10, clipped)

    # ---- models / training -----------------------------------------------
    "baselines": ["persistence", "seasonal_naive", "historical_mean"],
    "forecast_models": ["dlinear", "patchtst", "itransformer", "crossformer",
                        "nhits", "nn_resnet", "biconvlstm_s2s", "tsb",
                        "deepsense_f", "spectrumfm_f"],
    "modes": ["local", "centralized", "fedavg", "fedprox"],
    "seeds": [0, 1, 2],
    "seed": 20261019,
    "epochs": 40, "batch_size": 256, "lr": 1e-3, "weight_decay": 1e-4,
    "patience": 6, "d_model": 64, "dropout": 0.1, "grad_clip": 1.0,
    "fl_rounds": 40, "fl_local_epochs": 2, "fedprox_mu": 0.01,
    "fl_patience": 8,
    "fm_pretrain_epochs": 10, "fm_mask_ratio": 0.30,
    "decision_threshold": 0.5,       # also reported: F1-optimal threshold on VAL

    # ---- profiling --------------------------------------------------------
    "prof_iters": 50, "prof_warmup": 5, "prof_batch": 256,
    "device": "auto",
    "resume": True,
    "verbose": True,
}

QUICK = {   # --quick: smoke test of the code path only (results meaningless)
    "epochs": 2, "fl_rounds": 2, "fl_local_epochs": 1, "fm_pretrain_epochs": 1,
    "seeds": [0], "prof_iters": 3, "prof_warmup": 1, "patience": 1, "fl_patience": 1,
}

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
    """Deterministic seed per (model, mode, seed, ...), independent of order."""
    key = "|".join(str(p) for p in parts)
    return int((int(cfg["seed"]) + zlib.crc32(key.encode())) % (2 ** 31 - 1))


def get_device(cfg: Dict) -> torch.device:
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


# ==========================================================================
#                              DATA DISCOVERY
# ==========================================================================
_24_MARKERS = ("wifi_2p4ghz", "wifi24g", "2p4", "2400")


def _is_24(p: str) -> bool:
    q = p.lower().replace("\\", "/")
    return any(m in q for m in _24_MARKERS)


def _is_session_dir(d: str) -> bool:
    return (os.path.isfile(os.path.join(d, "session_meta.json"))
            and bool(glob.glob(os.path.join(d, "sweeps_*.csv"))))


def discover(cfg: Dict) -> Dict[str, str]:
    """Scenario -> source path. Explicit paths win; otherwise the newest
    matching .h5 export, else the newest matching raw session folder."""
    found: Dict[str, str] = {}
    root = cfg["data_root"]
    h5s, sess = [], []
    if os.path.isdir(root):
        for dp, dn, fn in os.walk(root):
            if _is_session_dir(dp):
                sess.append(dp)
            for f in fn:
                if f.lower().endswith(".h5"):
                    h5s.append(os.path.join(dp, f))
    for name, path in cfg["scenarios"].items():
        if path:
            full = path if os.path.isabs(path) else os.path.join(root, path)
            if os.path.exists(full):
                found[name] = full
                continue
            log(f"  [{name}] {full} not found -> trying auto-discovery")
        al = [a.lower() for a in cfg["scenario_aliases"].get(name, [name.lower()])]

        def match(p):
            q = p.lower().replace("\\", "/")
            return any(a in q for a in al) and _is_24(q)
        c = [p for p in h5s if match(p)] or [p for p in sess if match(p)]
        if c:
            c.sort(key=os.path.getmtime)
            found[name] = c[-1]
        else:
            log(f"  [{name}] nothing found under {root} (aliases {al}, needs a "
                f"2.4 GHz marker {list(_24_MARKERS)}) -> scenario skipped")
    return found


# ==========================================================================
#                                 LOADING
# ==========================================================================
def _dec(v):
    return v.decode("utf-8", "replace") if isinstance(v, (bytes, np.bytes_)) else str(v)


def _pick_band(dw: Dict, band: str):
    """Keep the dwells of the 2.4 GHz band; tolerate other band labels."""
    names = sorted(set(dw["band"].tolist()))
    if band in names:
        m = dw["band"] == band
    else:
        hits = [b for b in names if "2400" in b or "2p4" in b.lower() or "24g" in b.lower()]
        if len(hits) == 1 or len(names) == 1:
            use = hits[0] if hits else names[0]
            log(f"    bands in file {names}; using '{use}' as the 2.4 GHz band")
            m = dw["band"] == use
        else:
            return None, f"no band '{band}' among {names}; set CONFIG['band_name']"
    out = {k: v[m] for k, v in dw.items()}
    out["band"] = np.array([band] * int(m.sum()))
    return out, None


def _load_h5(path: str, band: str) -> Dict:
    """build_wifi_dataset.py export. Prefers the gap-handled /grid/<band>;
    falls back to per-dwell /sweeps + /psd. Refuses simulated rows."""
    try:
        import h5py
    except ImportError:
        sys.exit("h5py is required for .h5 input:  pip install h5py")
    with h5py.File(path, "r") as f:
        if "sweeps" in f and "simulated" in f["sweeps"]:
            sim = np.asarray(f["sweeps"]["simulated"][()], dtype=np.float64)
            if np.nansum(sim) > 0:
                return {"error": f"{int(np.nansum(sim))} dwells flagged simulated=1 -> refused"}
        top = sorted(f.keys())
        gname = None
        if "grid" in f:
            gk = sorted(f["grid"].keys())
            if band in gk:
                gname = band
            else:
                hits = [k for k in gk if "2400" in k or "24" in k.replace(".", "")]
                gname = hits[0] if len(hits) == 1 else (gk[0] if len(gk) == 1 else None)
                if gname:
                    log(f"    /grid has {gk}; using '{gname}' as the 2.4 GHz band")
                else:
                    return {"error": f"/grid has {gk}, none is '{band}'. Set CONFIG['band_name']."}
        if gname is not None:
            g = f["grid"][gname]
            miss = [k for k in ("psd_db", "freq_hz", "utc") if k not in g]
            if miss:
                return {"error": f"/grid/{gname} lacks {miss} (has {sorted(g.keys())})"}
            psd = np.asarray(g["psd_db"][()], dtype=np.float32)
            st = np.asarray(g["status"][()]) if "status" in g else None
            return {"kind": "grid", "psd_bins": psd,
                    "freq_hz": np.asarray(g["freq_hz"][()], dtype=np.float64),
                    "utc": np.asarray(g["utc"][()], dtype=np.float64),
                    "filled_frac": float((st == 1).mean()) if st is not None else float("nan"),
                    "src": path}
        if "sweeps" not in f or "psd" not in f or "psd_db" not in f["psd"]:
            return {"error": f"needs /grid/{band} or /sweeps + /psd/psd_db; file has {top}"}
        gs = f["sweeps"]
        need = ["sweep", "utc_start", "f_lo_hz", "f_hi_hz", "samp_rate_hz", "center_hz"]
        if "row_index" not in f["psd"]:
            need.append("psd_row")
        miss = [k for k in need if k not in gs]
        if miss:
            return {"error": f"/sweeps lacks {miss} (has {sorted(gs.keys())[:30]})"}
        n = int(gs["sweep"].shape[0])
        num = lambda k: np.asarray(gs[k][()], dtype=np.float64).reshape(-1)[:n]
        bn = np.array([_dec(x) for x in gs["band_name"][()]])[:n] if "band_name" in gs \
            else np.array([band] * n)
        prow = (np.asarray(f["psd"]["row_index"][()], dtype=np.int64)[:n]
                if "row_index" in f["psd"] else num("psd_row").astype(np.int64))
        psd = np.asarray(f["psd"]["psd_db"][()], dtype=np.float32)
        dw = {"band": bn, "sweep": num("sweep"), "utc": num("utc_start"),
              "f_lo": num("f_lo_hz"), "f_hi": num("f_hi_hz"),
              "rate": num("samp_rate_hz"), "ctr": num("center_hz"), "prow": prow}
    dw, err = _pick_band(dw, band)
    if err:
        return {"error": err}
    return {"kind": "dwells", "dw": dw, "psd_rows": psd, "src": path}


def _load_session(d: str, band: str) -> Dict:
    meta = json.load(open(os.path.join(d, "session_meta.json")))
    bins = int(meta.get("config", {}).get("psd_bins", 1024))
    rows, parts, off = [], [], 0
    for cp in sorted(glob.glob(os.path.join(d, "sweeps_*.csv"))):
        key = os.path.basename(cp)[len("sweeps_"):-len(".csv")]
        pp = os.path.join(d, f"psd_{key}.f16")
        base = off
        if os.path.exists(pp):
            raw = np.fromfile(pp, dtype=np.float16)
            nr = raw.size // bins
            parts.append(raw[:nr * bins].reshape(nr, bins).astype(np.float32))
            off += nr
        with open(cp, newline="") as fh:
            for r in csv.DictReader(fh):
                if str(r.get("simulated", "0")).strip() not in ("", "0", "0.0", "False"):
                    return {"error": "session contains simulated=1 dwells -> refused"}
                try:
                    pr = base + int(float(r["psd_row"]))
                except (KeyError, ValueError):
                    continue
                rows.append((r.get("band_name", band), float(r["sweep"]),
                             float(r["utc_start"]), float(r["f_lo_hz"]),
                             float(r["f_hi_hz"]), float(r["samp_rate_hz"]),
                             float(r["center_hz"]), pr))
    if not rows or not parts:
        return {"error": "no sweeps rows / psd rows"}
    a = list(zip(*rows))
    dw = {"band": np.array(a[0]), "sweep": np.array(a[1]), "utc": np.array(a[2]),
          "f_lo": np.array(a[3]), "f_hi": np.array(a[4]), "rate": np.array(a[5]),
          "ctr": np.array(a[6]), "prow": np.array(a[7], dtype=np.int64)}
    dw, err = _pick_band(dw, band)
    if err:
        return {"error": err}
    return {"kind": "dwells", "dw": dw, "psd_rows": np.concatenate(parts, 0), "src": d}


def load_source(path: str, band: str) -> Dict:
    if os.path.isdir(path):
        ex = sorted(glob.glob(os.path.join(path, "export", "*.h5")), key=os.path.getmtime)
        if ex:
            return _load_h5(ex[-1], band)
        if _is_session_dir(path):
            return _load_session(path, band)
        return {"error": "folder has neither export/*.h5 nor a raw session"}
    return _load_h5(path, band)


def freq_span(src: Dict, band: str) -> Tuple[float, float]:
    if src["kind"] == "grid":
        f = src["freq_hz"]
        df = float(np.median(np.diff(f))) if f.size > 1 else 0.0
        return float(f.min() - df / 2), float(f.max() + df / 2)
    dw = src["dw"]
    m = dw["band"] == band
    return float(dw["f_lo"][m].min()), float(dw["f_hi"][m].max())


def to_subchannels(src: Dict, band: str, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """-> P[T, C] dB (linear-domain mean per sub-channel), utc[T], #interpolated."""
    C = len(edges) - 1
    if src["kind"] == "grid":
        P, f, t = src["psd_bins"], src["freq_hz"], src["utc"]
        which = np.searchsorted(edges, f, "right") - 1
        out = np.full((P.shape[0], C), np.nan, np.float32)
        lin = np.power(10.0, P.astype(np.float64) / 10.0)
        for c in range(C):
            m = which == c
            if m.any():
                with np.errstate(invalid="ignore"):
                    out[:, c] = 10 * np.log10(np.nanmean(lin[:, m], 1) + 1e-20)
        o = np.argsort(t)
        out, t = out[o], t[o]
    else:
        dw = src["dw"]
        m = dw["band"] == band
        dw = {k: v[m] for k, v in dw.items()}
        R = src["psd_rows"]
        sw, si = np.unique(dw["sweep"], return_inverse=True)
        out = np.full((len(sw), C), np.nan, np.float32)
        t = np.bincount(si, weights=dw["utc"]) / np.maximum(np.bincount(si), 1)
        for i in range(len(dw["sweep"])):
            pr = dw["prow"][i]
            if not 0 <= pr < R.shape[0]:
                continue
            row = R[pr]
            nb = row.size
            bf = dw["ctr"][i] + (np.arange(nb) - nb / 2.0) * (dw["rate"][i] / nb)
            k = (bf >= dw["f_lo"][i]) & (bf <= dw["f_hi"][i]) & np.isfinite(row)
            if k.sum() < 4:
                continue
            wc = np.searchsorted(edges, bf[k], "right") - 1
            lin = np.power(10.0, row[k].astype(np.float64) / 10.0)
            ok = (wc >= 0) & (wc < C)
            s = np.bincount(wc[ok], weights=lin[ok], minlength=C)
            n = np.bincount(wc[ok], minlength=C)
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.where(n > 0, 10 * np.log10(s / np.maximum(n, 1) + 1e-20), np.nan)
            out[si[i]] = np.fmax(out[si[i]], v)
        o = np.argsort(t)
        out, t = out[o], t[o]
    n_interp = int((~np.isfinite(out)).sum())
    for c in range(C):                      # residual holes: interpolate in time
        col = out[:, c]
        bad = ~np.isfinite(col)
        if bad.all():
            col[:] = np.nanmin(out) if np.isfinite(out).any() else -100.0
        elif bad.any():
            ok = np.flatnonzero(~bad)
            col[bad] = np.interp(np.flatnonzero(bad), ok, col[ok])
    return out, t, n_interp


def _runs(col: np.ndarray, val: int) -> List[int]:
    out, n = [], 0
    for v in col:
        if v == val:
            n += 1
        elif n:
            out.append(n); n = 0
    if n:
        out.append(n)
    return out


def build_scenario(name: str, P: np.ndarray, t: np.ndarray, edges: np.ndarray,
                   cfg: Dict, n_interp: int, src_path: str) -> Dict:
    T, C = P.shape
    W, H = cfg["lookback"], cfg["horizon"]
    n_fit = max(W + H + 1, int(T * (1 - cfg["val_frac"] - cfg["test_frac"])))
    nf = np.quantile(P[:n_fit], cfg["noise_quantile"], axis=0)       # Q10, train only
    E = (P - nf[None, :]).astype(np.float32)                          # excess = SNR
    Y = (E > cfg["eta_db"]).astype(np.float32)
    occ_runs = [r for c in range(C) for r in _runs(Y[:, c].astype(int), 1)]
    idle_runs = [r for c in range(C) for r in _runs(Y[:, c].astype(int), 0)]
    s = Y.mean(1) - Y.mean()
    ac1 = float(np.corrcoef(s[:-1], s[1:])[0, 1]) if s.std() > 1e-9 else float("nan")
    step = float(np.median(np.diff(t))) if T > 1 else float("nan")
    sc = {"name": name, "P": P, "E": E, "Y": Y, "utc": t, "noise_floor": nf,
          "edges": edges, "T": T, "C": C, "n_fit": n_fit, "step_s": step}
    sc["card"] = {
        "scenario": name, "source": src_path, "steps": T, "subchannels": C,
        "f_lo_MHz": edges[0] / 1e6, "f_hi_MHz": edges[-1] / 1e6,
        "subchannel_MHz": cfg["subchannel_hz"] / 1e6,
        "step_s_median": step, "duration_min": float((t[-1] - t[0]) / 60.0),
        "cells_interpolated": n_interp,
        "noise_quantile": cfg["noise_quantile"], "eta_db": cfg["eta_db"],
        "noise_floor_mean_db": float(nf.mean()), "noise_floor_std_db": float(nf.std()),
        "positive_rate": float(Y.mean()),
        "positive_rate_train": float(Y[:n_fit].mean()),
        "positive_rate_test": float(Y[int(T * (1 - cfg["test_frac"])):].mean()),
        "on_run_mean_steps": float(np.mean(occ_runs)) if occ_runs else 0.0,
        "off_run_mean_steps": float(np.mean(idle_runs)) if idle_runs else 0.0,
        "lag1_acf": ac1,
        "excess_max_db": float(E.max()),
        "cells_within_1dB_of_eta": float((np.abs(E - cfg["eta_db"]) < 1).mean()),
    }
    log(f"  [{name}] T={T} steps (median {step:.2f} s/step, "
        f"{sc['card']['duration_min']:.1f} min), C={C}, occupancy "
        f"{Y.mean()*100:.1f}%, NF {nf.mean():.1f} dB, max SNR {E.max():.1f} dB")
    return sc


def estimate_period(Y: np.ndarray, max_lag: int) -> int:
    s = Y.mean(1)
    s = s - s.mean()
    if s.std() < 1e-9 or s.size < 8:
        return 1
    n = min(max_lag, s.size // 2)
    ac = np.nan_to_num(np.array([np.corrcoef(s[:-l], s[l:])[0, 1] for l in range(1, n)]))
    if ac.size < 2:
        return 1
    k = int(np.argmax(ac[1:]) + 2)
    return k if ac[k - 1] > 0.2 else 1


def make_windows(sc: Dict, cfg: Dict) -> Dict:
    """Sliding windows + chronological split with a W+H gap on each side."""
    W, H, S = cfg["lookback"], cfg["horizon"], cfg["stride"]
    Y, E = sc["Y"], sc["E"]
    T = Y.shape[0]
    starts = np.arange(0, T - W - H + 1, S)
    if cfg["input_repr"] == "excess":
        Xsrc = np.clip((E - cfg["eta_db"]) / 10.0, -2.0, 4.0).astype(np.float32)
    else:
        Xsrc = Y
    idx_in = starts[:, None] + np.arange(W)[None, :]
    idx_out = starts[:, None] + W + np.arange(H)[None, :]
    d = {"X": Xsrc[idx_in], "Xbin": Y[idx_in], "Y": Y[idx_out], "E": E[idx_out],
         "t0": sc["utc"][starts + W], "start": starts}
    n = len(starts)
    gap = W + H
    n_te = int(round(cfg["test_frac"] * n))
    n_va = int(round(cfg["val_frac"] * n))
    n_tr = n - n_va - n_te - 2 * gap
    if n_tr < 16:
        sys.exit(f"[{sc['name']}] only {n} windows: too few for W={W}, H={H}")
    d["tr"] = np.arange(0, n_tr)
    d["va"] = np.arange(n_tr + gap, n_tr + gap + n_va)
    d["te"] = np.arange(n_tr + n_va + 2 * gap, n)
    d["period"] = estimate_period(Y[:sc["n_fit"]], W)
    # INR statistics per sub-channel on the training span (allocation stage)
    inr = np.maximum(np.power(10.0, E[:sc["n_fit"]] / 10.0) - 1.0, 0.0)
    yb = Y[:sc["n_fit"]] > 0.5
    d["inr_busy"] = np.array([inr[yb[:, c], c].mean() if yb[:, c].any() else 10.0
                              for c in range(Y.shape[1])], np.float32)
    d["inr_idle"] = np.array([inr[~yb[:, c], c].mean() if (~yb[:, c]).any() else 0.0
                              for c in range(Y.shape[1])], np.float32)
    log(f"  [{sc['name']}] windows train/val/test = {len(d['tr'])}/{len(d['va'])}/"
        f"{len(d['te'])}  seasonal period = {d['period']} steps")
    return d


# ==========================================================================
#                    STAGE 1 - NON-LEARNED BASELINES
# ==========================================================================
def baseline_forecast(name: str, X: np.ndarray, H: int, period: int) -> np.ndarray:
    """X: binary [N, W, C] -> occupancy probability [N, H, C]."""
    if name == "persistence":
        return np.repeat(X[:, -1:, :], H, axis=1)
    if name == "historical_mean":
        return np.repeat(X.mean(axis=1, keepdims=True), H, axis=1)
    if name == "seasonal_naive":
        m, W = max(1, period), X.shape[1]
        out = np.empty((X.shape[0], H, X.shape[2]), dtype=X.dtype)
        for h in range(H):
            out[:, h, :] = X[:, max(0, min(W - m + (h % m), W - 1)), :]
        return out
    raise ValueError(name)


# ==========================================================================
#                 STAGE 1 - LEARNED FORECASTERS (logit outputs)
# ==========================================================================
class SeriesDecomp(nn.Module):
    def __init__(self, kernel: int = 25):
        super().__init__()
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel, stride=1, padding=0)

    def forward(self, x):
        p = (self.kernel - 1) // 2
        xp = torch.cat([x[:, :1].repeat(1, p, 1), x,
                        x[:, -1:].repeat(1, self.kernel - 1 - p, 1)], dim=1)
        trend = self.avg(xp.permute(0, 2, 1)).permute(0, 2, 1)
        return x - trend, trend


class DLinear(nn.Module):
    """Zeng et al., 'Are Transformers Effective for Time Series Forecasting?',
    AAAI 2023. Trend/seasonal decomposition + one linear map each."""

    def __init__(self, W, H, C, **kw):
        super().__init__()
        self.decomp = SeriesDecomp(min(25, W - (1 - W % 2)))
        self.lin_s, self.lin_t = nn.Linear(W, H), nn.Linear(W, H)

    def forward(self, x):
        s, t = self.decomp(x)
        return (self.lin_s(s.permute(0, 2, 1)) + self.lin_t(t.permute(0, 2, 1))).permute(0, 2, 1)


class NHiTS(nn.Module):
    """Challu et al., 'N-HiTS: Neural Hierarchical Interpolation for Time
    Series Forecasting', AAAI 2023. Multi-rate pooling + hierarchical
    interpolation of backcast/forecast."""

    def __init__(self, W, H, C, d_model=64, dropout=0.1, pools=(4, 2, 1), **kw):
        super().__init__()
        self.W, self.H, self.C, self.pools = W, H, C, pools
        self.blocks = nn.ModuleList()
        for p in pools:
            wp = max(1, W // p)
            self.blocks.append(nn.Sequential(
                nn.Linear(wp * C, 4 * d_model), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(4 * d_model, 4 * d_model), nn.ReLU(),
                nn.Linear(4 * d_model, (wp + H) * C)))

    def forward(self, x):
        B = x.shape[0]
        res = x
        out = torch.zeros(B, self.H, self.C, device=x.device, dtype=x.dtype)
        for p, blk in zip(self.pools, self.blocks):
            xp = res.permute(0, 2, 1)
            xp = F.avg_pool1d(xp, p, stride=p) if p > 1 else xp
            wp = xp.shape[-1]
            h = blk(xp.permute(0, 2, 1).reshape(B, -1))
            back = h[:, :wp * self.C].reshape(B, wp, self.C)
            res = res - F.interpolate(back.permute(0, 2, 1), size=self.W, mode="linear",
                                      align_corners=False).permute(0, 2, 1)
            out = out + h[:, wp * self.C:].reshape(B, self.H, self.C)
        return out


class PatchTST(nn.Module):
    """Nie et al., 'A Time Series is Worth 64 Words', ICLR 2023.
    Channel-independent patches + Transformer encoder."""

    def __init__(self, W, H, C, d_model=64, dropout=0.1, patch=8, stride=4,
                 nhead=4, layers=2, **kw):
        super().__init__()
        self.H, self.patch, self.stride = H, patch, stride
        self.n_patch = max(1, (W - patch) // stride + 1)
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, 2 * d_model, dropout=dropout,
                                         batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(self.n_patch * d_model, H)

    def forward(self, x):
        B, W, C = x.shape
        z = x.permute(0, 2, 1).reshape(B * C, W)
        idx = (torch.arange(self.patch, device=x.device)[None, :]
               + self.stride * torch.arange(self.n_patch, device=x.device)[:, None]).clamp(max=W - 1)
        h = self.tr(self.embed(z[:, idx]) + self.pos)
        return self.head(h.reshape(B * C, -1)).reshape(B, C, self.H).permute(0, 2, 1)


class ITransformer(nn.Module):
    """Liu et al., 'iTransformer: Inverted Transformers Are Effective for Time
    Series Forecasting', ICLR 2024. Each sub-channel is one token."""

    def __init__(self, W, H, C, d_model=64, dropout=0.1, nhead=4, layers=2, **kw):
        super().__init__()
        self.embed = nn.Linear(W, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, 2 * d_model, dropout=dropout,
                                         batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, H)

    def forward(self, x):
        return self.head(self.tr(self.embed(x.permute(0, 2, 1)))).permute(0, 2, 1)


class Crossformer(nn.Module):
    """Zhang & Yan, 'Crossformer: Transformer Utilizing Cross-Dimension
    Dependency', ICLR 2023. Two-stage attention: time, then sub-channels."""

    def __init__(self, W, H, C, d_model=64, dropout=0.1, seg=4, nhead=4, **kw):
        super().__init__()
        self.H, self.seg, self.n_seg = H, seg, max(1, W // seg)
        self.embed = nn.Linear(seg, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_seg, d_model))
        self.t_att = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.c_att = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.head = nn.Linear(self.n_seg * d_model, H)

    def forward(self, x):
        B, W, C = x.shape
        z = x[:, W - self.n_seg * self.seg:].permute(0, 2, 1).reshape(B * C, self.n_seg, self.seg)
        z = self.embed(z) + self.pos
        a, _ = self.t_att(z, z, z, need_weights=False); z = self.n1(z + a)
        z = z.reshape(B, C, self.n_seg, -1).permute(0, 2, 1, 3).reshape(B * self.n_seg, C, -1)
        a, _ = self.c_att(z, z, z, need_weights=False); z = self.n2(z + a)
        z = self.n3(z + self.ff(z))
        z = z.reshape(B, self.n_seg, C, -1).permute(0, 2, 1, 3).reshape(B * C, -1)
        return self.head(z).reshape(B, C, self.H).permute(0, 2, 1)


class DeepSenseF(nn.Module):
    """Uvaydov, D'Oro, Restuccia, Melodia, 'DeepSense: Fast Wideband Spectrum
    Sensing Through Real-Time In-the-Loop Deep Learning', INFOCOM 2021.
    ADAPTATION: DeepSense classifies sub-band occupancy from raw I/Q with a
    small CNN (conv blocks + dense). Here the same CNN design reads the
    (time x sub-channel) occupancy window and its dense head forecasts the
    next H steps. Architecture family kept; input and head differ."""

    def __init__(self, W, H, C, d_model=64, dropout=0.1, **kw):
        super().__init__()
        self.H, self.C = H, C
        self.feat = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, C)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 4 * C, 4 * d_model),
                                  nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(4 * d_model, H * C))

    def forward(self, x):
        return self.head(self.feat(x.unsqueeze(1))).reshape(-1, self.H, self.C)


class _ConfBlock(nn.Module):
    """MHSA + depthwise-conv module + FFN (conv + attention encoder block)."""

    def __init__(self, d, nhead, dropout):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.dw = nn.Conv1d(d, d, 5, padding=2, groups=d)
        self.pw = nn.Conv1d(d, d, 1)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, z):
        h = self.n1(z)
        z = z + self.att(h, h, h, need_weights=False)[0]
        h = self.n2(z).transpose(1, 2)
        z = z + self.pw(F.gelu(self.dw(h))).transpose(1, 2)
        return z + self.ff(self.n3(z))


class SpectrumFMF(nn.Module):
    """Liu, Zhang, Wu, Zhou, Wu, Ng, Chae, 'SpectrumFM: Redefining Spectrum
    Cognition via Foundation Modeling', IEEE GLOBECOM 2025.
    Conv + multi-head self-attention encoder, self-supervised pre-training by
    masked reconstruction and next-slot prediction, then fine-tuned.
    ADAPTATION: tokens are the W time steps of the sub-channel vector (the
    paper tokenises I/Q); model size is reduced for this dataset; no external
    pre-training corpus -- pre-training uses only this run's TRAINING split."""

    def __init__(self, W, H, C, d_model=64, dropout=0.1, nhead=4, layers=3, **kw):
        super().__init__()
        self.W, self.H, self.C = W, H, C
        self.stem = nn.Sequential(nn.Conv1d(C, d_model, 3, padding=1), nn.GELU(),
                                  nn.Conv1d(d_model, d_model, 3, padding=1))
        self.pos = nn.Parameter(torch.zeros(1, W, d_model))
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([_ConfBlock(d_model, nhead, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.recon = nn.Linear(d_model, C)          # pre-training head
        self.head = nn.Linear(2 * d_model, H * C)   # forecasting head
        self.mask_ratio = 0.3
        self.binary_input = True

    def encode(self, x, mask=None):
        z = self.stem(x.transpose(1, 2)).transpose(1, 2)
        if mask is not None:
            z = torch.where(mask[..., None], self.mask_tok.expand_as(z), z)
        z = z + self.pos
        for b in self.blocks:
            z = b(z)
        return self.norm(z)

    def forward(self, x):
        z = self.encode(x)
        return self.head(torch.cat([z.mean(1), z[:, -1]], -1)).reshape(-1, self.H, self.C)

    def pretrain_loss(self, x):
        B, W, _ = x.shape
        mask = torch.rand(B, W, device=x.device) < self.mask_ratio
        mask[:, -1] = True                          # next-slot prediction
        rec = self.recon(self.encode(x, mask))
        if self.binary_input:
            l = F.binary_cross_entropy_with_logits(rec, x, reduction="none")
        else:
            l = (rec - x) ** 2
        return (l.mean(-1) * mask).sum() / mask.sum().clamp(min=1)


class _ResBlock2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
                               nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch))

    def forward(self, x):
        return F.relu(x + self.f(x))


class NNResNet(nn.Module):
    """Ren, Mosavat-Jahromi, Cai, Kidston, 'Spatio-Temporal Spectrum Load
    Prediction Using Convolutional Neural Network and ResNet', IEEE TCCN 2022.
    CNN stem + residual blocks over the (time x channel) spectrum matrix.
    ADAPTATION: the paper's spatial axis (sensing locations) is the frequency
    axis here (one sensor); the output head predicts H steps."""

    def __init__(self, W, H, C, d_model=64, dropout=0.1, ch=32, blocks=3, **kw):
        super().__init__()
        self.H, self.C = H, C
        self.stem = nn.Sequential(nn.Conv2d(1, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU())
        self.res = nn.Sequential(*[_ResBlock2d(ch) for _ in range(blocks)])
        self.pool = nn.AdaptiveAvgPool2d((4, C))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(ch * 4 * C, H * C))

    def forward(self, x):
        return self.head(self.pool(self.res(self.stem(x.unsqueeze(1))))).reshape(-1, self.H, self.C)


class _ConvLSTMCell1d(nn.Module):
    """ConvLSTM cell whose convolution runs along the frequency axis."""

    def __init__(self, cin, hid, k=3):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv1d(cin + hid, 4 * hid, k, padding=k // 2)

    def forward(self, x, h, c):                       # x [B,cin,C], h,c [B,hid,C]
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class BiConvLSTMSeq2Seq(nn.Module):
    """Wang, Hu, Jiang, Chen, 'A Deep Long-Term Joint Temporal-Spectral Network
    for Spectrum Prediction', Sensors 24(5):1498, 2024. Bidirectional ConvLSTM
    encoder over time (convolution across frequency) + ConvLSTM seq2seq
    decoder that feeds back its own prediction for H steps."""

    def __init__(self, W, H, C, d_model=64, dropout=0.1, hid=16, **kw):
        super().__init__()
        self.H, self.hid = H, hid
        self.fw, self.bw = _ConvLSTMCell1d(1, hid), _ConvLSTMCell1d(1, hid)
        self.dec = _ConvLSTMCell1d(1, 2 * hid)
        self.out = nn.Conv1d(2 * hid, 1, 1)

    def forward(self, x):
        B, W, C = x.shape
        z = lambda n: torch.zeros(B, n, C, device=x.device, dtype=x.dtype)
        hf, cf, hb, cb = z(self.hid), z(self.hid), z(self.hid), z(self.hid)
        for t in range(W):
            hf, cf = self.fw(x[:, t:t + 1], hf, cf)
            hb, cb = self.bw(x[:, W - 1 - t:W - t], hb, cb)
        h, c = torch.cat([hf, hb], 1), torch.cat([cf, cb], 1)
        inp, outs = x[:, -1:], []
        for _ in range(self.H):
            h, c = self.dec(inp, h, c)
            o = self.out(h)                           # logits [B,1,C]
            outs.append(o)
            inp = torch.sigmoid(o)
        return torch.cat(outs, 1)


class TSB(nn.Module):
    """'Multi-Channel Multi-Step Spectrum Prediction Using Transformer and
    Stacked Bi-LSTM' (TSB), arXiv:2405.19138, 2024. Multi-head attention
    encoder followed by stacked Bi-LSTMs in the encoder and the decoder."""

    def __init__(self, W, H, C, d_model=64, dropout=0.1, nhead=4, **kw):
        super().__init__()
        self.H, self.C = H, C
        self.embed = nn.Linear(C, d_model)
        self.pos = nn.Parameter(torch.zeros(1, W, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, 2 * d_model, dropout=dropout,
                                         batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, 2, enable_nested_tensor=False)
        self.enc_lstm = nn.LSTM(d_model, d_model // 2, 2, batch_first=True, bidirectional=True,
                                dropout=dropout)
        self.query = nn.Parameter(torch.zeros(1, H, d_model))
        self.dec_lstm = nn.LSTM(d_model, d_model // 2, 2, batch_first=True, bidirectional=True,
                                dropout=dropout)
        self.head = nn.Linear(d_model, C)

    def forward(self, x):
        z, _ = self.enc_lstm(self.tr(self.embed(x) + self.pos))
        q = self.query.expand(x.shape[0], -1, -1) + z[:, -1:, :]
        d, _ = self.dec_lstm(torch.cat([z, q], 1))
        return self.head(d[:, -self.H:])


MODEL_REGISTRY = {"dlinear": DLinear, "nhits": NHiTS, "patchtst": PatchTST,
                  "itransformer": ITransformer, "crossformer": Crossformer,
                  "deepsense_f": DeepSenseF, "spectrumfm_f": SpectrumFMF,
                  "nn_resnet": NNResNet, "biconvlstm_s2s": BiConvLSTMSeq2Seq, "tsb": TSB}
MODEL_LABEL = {"persistence": "Persistence", "seasonal_naive": "Seasonal naive",
               "historical_mean": "Historical mean", "dlinear": "DLinear",
               "nhits": "N-HiTS", "patchtst": "PatchTST", "itransformer": "iTransformer",
               "crossformer": "Crossformer", "deepsense_f": "DeepSense-F",
               "spectrumfm_f": "SpectrumFM-F", "nn_resnet": "NN-ResNet",
               "biconvlstm_s2s": "Bi-ConvLSTM-S2S", "tsb": "TSB"}


def build_model(name: str, cfg: Dict, C: int) -> nn.Module:
    m = MODEL_REGISTRY[name](W=cfg["lookback"], H=cfg["horizon"], C=C,
                             d_model=cfg["d_model"], dropout=cfg["dropout"])
    if isinstance(m, SpectrumFMF):
        m.mask_ratio = cfg["fm_mask_ratio"]
        m.binary_input = cfg["input_repr"] == "binary"
    return m


# ==========================================================================
#                          TRAINING (centralised)
# ==========================================================================
def _tt(a, dev):
    return torch.as_tensor(a, dtype=torch.float32, device=dev)


def _objective(model, xb, yb, objective):
    if objective == "pretrain":
        return model.pretrain_loss(xb)
    return F.binary_cross_entropy_with_logits(model(xb), yb)


def train_epochs(model, X, Y, cfg, dev, epochs, objective="forecast",
                 prox_ref=None, mu=0.0, rng=None) -> float:
    """Plain local training: no early stopping (used by FL clients too)."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n, bs = X.shape[0], cfg["batch_size"]
    rng = rng or np.random.default_rng(0)
    last = float("nan")
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(n)
        tot, cnt = 0.0, 0
        for a in range(0, n, bs):
            j = perm[a:a + bs]
            if len(j) < 2:
                continue
            xb, yb = X[j], Y[j]
            loss = _objective(model, xb, yb, objective)
            if prox_ref is not None and mu > 0:
                loss = loss + 0.5 * mu * sum(((p - q) ** 2).sum()
                                             for p, q in zip(model.parameters(), prox_ref))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            tot += float(loss) * len(j); cnt += len(j)
        last = tot / max(cnt, 1)
    return last


@torch.no_grad()
def eval_loss(model, X, Y, objective="forecast", bs=1024) -> float:
    model.eval()
    tot, n = 0.0, 0
    for a in range(0, X.shape[0], bs):
        tot += float(_objective(model, X[a:a + bs], Y[a:a + bs], objective)) * len(X[a:a + bs])
        n += len(X[a:a + bs])
    return tot / max(n, 1)


def fit_central(model, Xtr, Ytr, Xva, Yva, cfg, dev, objective="forecast",
                epochs=None, seed=0) -> Dict:
    """Early-stopped training on (pooled) data; returns history."""
    epochs = epochs or cfg["epochs"]
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        tr = train_epochs(model, Xtr, Ytr, cfg, dev, 1, objective, rng=rng)
        va = eval_loss(model, Xva, Yva, objective)
        hist.append((tr, va))
        if va < best - 1e-5:
            best, bad = va, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"epochs_run": len(hist), "best_val_loss": best, "train_s": time.time() - t0}


# ==========================================================================
#                 FEDERATED TRAINING (FedAvg / FedProx)
# ==========================================================================
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
            out[k] = states[0][k].clone()      # e.g. BatchNorm counters
    return out


def fit_federated(model, clients: List[Tuple], val_sets: List[Tuple], cfg, dev,
                  algo: str = "fedavg", objective="forecast", rounds=None,
                  seed=0) -> Dict:
    """clients: [(X, Y)] tensors per scenario. Sample-weighted averaging."""
    rounds = rounds or cfg["fl_rounds"]
    mu = cfg["fedprox_mu"] if algo == "fedprox" else 0.0
    rng = np.random.default_rng(seed)
    g = copy.deepcopy(model.state_dict())
    best, best_state, bad, hist = float("inf"), None, 0, []
    nval = sum(v[0].shape[0] for v in val_sets)
    t0 = time.time()
    for r in range(rounds):
        states, ws = [], []
        for (X, Y) in clients:
            model.load_state_dict(g)
            ref = [p.detach().clone() for p in model.parameters()] if mu > 0 else None
            train_epochs(model, X, Y, cfg, dev, cfg["fl_local_epochs"], objective,
                         prox_ref=ref, mu=mu, rng=rng)
            states.append(copy.deepcopy(model.state_dict()))
            ws.append(X.shape[0])
        g = _avg_states(states, ws)
        model.load_state_dict(g)
        va = sum(eval_loss(model, Xv, Yv, objective) * Xv.shape[0] for Xv, Yv in val_sets) / max(nval, 1)
        hist.append(va)
        if va < best - 1e-5:
            best, bad, best_state = va, 0, copy.deepcopy(g)
        else:
            bad += 1
            if bad >= cfg["fl_patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    nb = model_bytes(model)
    return {"rounds_run": len(hist), "best_val_loss": best, "train_s": time.time() - t0,
            "comm_bytes": 2 * nb * len(clients) * len(hist), "model_bytes": nb,
            "n_clients": len(clients)}


@torch.no_grad()
def predict_proba(model, X, bs=1024) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(X[a:a + bs])).float().cpu().numpy() for a in range(0, X.shape[0], bs)]
    return np.concatenate(out, 0)


# ==========================================================================
#                          STAGE 1 - METRICS
# ==========================================================================
def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    _, first, cnt = np.unique(ps, return_index=True, return_counts=True)
    ranks = np.empty(len(p), np.float64)
    ranks[order] = np.repeat(first + (cnt - 1) / 2.0 + 1.0, cnt)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _auprc(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision (step-wise PR area), ties grouped."""
    npos = float(y.sum())
    if npos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order].astype(np.float64), p[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]
    tp, fp = tp[last], fp[last]
    rec, prec = tp / npos, tp / np.maximum(tp + fp, 1e-12)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def _ece(y, p, bins=15) -> float:
    e, n = 0.0, len(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def bin_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5, full: bool = True) -> Dict:
    y = y.reshape(-1).astype(np.float64)
    p = np.clip(p.reshape(-1).astype(np.float64), 1e-7, 1 - 1e-7)
    yb, pb = y > 0.5, p >= thr
    tp = int(np.sum(yb & pb)); tn = int(np.sum(~yb & ~pb))
    fp = int(np.sum(~yb & pb)); fn = int(np.sum(yb & ~pb))
    n = max(len(y), 1)
    rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    out = {"n_cells": int(len(y)), "accuracy": (tp + tn) / n, "precision": prec,
           "recall_pd": rec, "specificity": spec, "false_alarm_rate": 1 - spec,
           "miss_rate": 1 - rec,
           "f1": 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0,
           "balanced_accuracy": 0.5 * (rec + spec),
           "mcc": (tp * tn - fp * fn) / den,
           "brier": float(np.mean((p - y) ** 2)),
           "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
           "mae": float(np.mean(np.abs(p - y))),
           "rmse": float(math.sqrt(np.mean((p - y) ** 2))),
           "positive_rate": float(y.mean()), "pred_positive_rate": float(pb.mean())}
    if full:
        out.update({"auroc": _auroc(y, p), "auprc": _auprc(y, p), "ece": _ece(y, p)})
    return out


def f1_opt_threshold(y, p) -> float:
    y = y.reshape(-1) > 0.5
    p = p.reshape(-1)
    best, bt = -1.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        pb = p >= t
        tp = np.sum(y & pb); fp = np.sum(~y & pb); fn = np.sum(y & ~pb)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best:
            best, bt = f1, float(t)
    return bt


def snr_edges(emax: float, cfg: Dict) -> np.ndarray:
    lo, w = cfg["snr_bin_start_db"], cfg["snr_bin_width_db"]
    hi = lo + w * max(1, math.ceil((emax - lo) / w))
    return np.arange(lo, hi + 1e-9, w)


def _bin_mask(E, lo, hi, last, eta):
    lower = (E > lo) if abs(lo - eta) < 1e-9 else (E >= lo)
    return lower & ((E <= hi) if last else (E < hi))


def forecast_rows(model: str, mode: str, seed: int, scen: str, y, p, E, base: Dict,
                  thr: float, thr_val: float, edges: np.ndarray, cfg: Dict,
                  extra: Dict) -> Tuple[Dict, List[Dict], List[Dict], List[Dict]]:
    """-> overall row, per-horizon rows, per-SNR cell rows, per-SNR unit rows."""
    eta = cfg["eta_db"]
    key = {"model": MODEL_LABEL.get(model, model), "model_key": model, "mode": mode,
           "seed": seed, "scenario": scen}
    ov = {**key, **bin_metrics(y, p, thr)}
    vt = bin_metrics(y, p, thr_val, full=False)
    ov.update({"threshold_val_f1opt": thr_val, "f1_at_val_thr": vt["f1"],
               "accuracy_at_val_thr": vt["accuracy"], "recall_at_val_thr": vt["recall_pd"],
               "far_at_val_thr": vt["false_alarm_rate"]})
    bb = {k: float(np.mean((np.clip(v, 1e-7, 1 - 1e-7) - y) ** 2)) for k, v in base.items()}
    best_b = min(bb.values()) if bb else float("nan")
    ov["brier_skill_vs_persistence"] = 1 - ov["brier"] / bb["persistence"] if "persistence" in bb else float("nan")
    ov["brier_skill_vs_hist_mean"] = 1 - ov["brier"] / bb["historical_mean"] if "historical_mean" in bb else float("nan")
    ov["brier_skill_vs_best_baseline"] = 1 - ov["brier"] / best_b if bb else float("nan")
    ov["beats_hist_mean"] = bool(ov["brier"] < bb.get("historical_mean", -1))
    ov["beats_all_baselines"] = bool(ov["brier"] < best_b)
    ov.update(extra)

    hz = []
    for h in range(y.shape[1]):
        r = bin_metrics(y[:, h], p[:, h], thr)
        hz.append({**key, "horizon_step": h + 1, **{k: r[k] for k in
                   ("accuracy", "precision", "recall_pd", "false_alarm_rate", "f1",
                    "mcc", "auroc", "auprc", "brier", "logloss")}})

    cells = []
    nb = len(edges) - 1
    idle = E <= eta
    if idle.any():
        r = bin_metrics(y[idle], p[idle], thr, full=False)
        cells.append({**key, "snr_bin": f"<= {eta:g} (idle)", "snr_lo_db": float("-inf"),
                      "snr_hi_db": eta, "n_cells": int(idle.sum()),
                      "false_alarm_rate": r["false_alarm_rate"], "specificity": r["specificity"],
                      "recall_pd": float("nan"), "miss_rate": float("nan"),
                      "mean_prob": float(p[idle].mean()), "brier": r["brier"],
                      "logloss": r["logloss"],
                      "brier_best_baseline": min(float(np.mean((v[idle] - y[idle]) ** 2)) for v in base.values()) if base else float("nan")})
    for i in range(nb):
        lo, hi = float(edges[i]), float(edges[i + 1])
        m = _bin_mask(E, lo, hi, i == nb - 1, eta)
        if not m.any():
            cells.append({**key, "snr_bin": f"{lo:g}-{hi:g}", "snr_lo_db": lo, "snr_hi_db": hi,
                          "n_cells": 0})
            continue
        pm, ym = p[m], y[m]
        bbm = min(float(np.mean((v[m] - ym) ** 2)) for v in base.values()) if base else float("nan")
        br = float(np.mean((pm - ym) ** 2))
        cells.append({**key, "snr_bin": f"{lo:g}-{hi:g}", "snr_lo_db": lo, "snr_hi_db": hi,
                      "n_cells": int(m.sum()),
                      "recall_pd": float(np.mean(pm >= thr)), "miss_rate": float(np.mean(pm < thr)),
                      "recall_pd_at_val_thr": float(np.mean(pm >= thr_val)),
                      "mean_prob": float(pm.mean()), "brier": br,
                      "logloss": float(-np.mean(np.log(np.clip(pm, 1e-7, 1)))),
                      "brier_best_baseline": bbm,
                      "brier_skill_vs_best_baseline": 1 - br / bbm if bbm > 0 else float("nan")})

    units = []
    U = E.mean(1)                                     # [N, C]: mean target SNR
    for i in range(-1, nb):
        if i < 0:
            m, lab, lo, hi = U <= eta, f"<= {eta:g}", float("-inf"), eta
        else:
            lo, hi = float(edges[i]), float(edges[i + 1])
            m, lab = _bin_mask(U, lo, hi, i == nb - 1, eta), f"{lo:g}-{hi:g}"
        if not m.any():
            units.append({**key, "snr_bin": lab, "snr_lo_db": lo, "snr_hi_db": hi, "n_units": 0})
            continue
        ym = y.transpose(0, 2, 1)[m]                  # [n_units, H]
        pm = p.transpose(0, 2, 1)[m]
        r = bin_metrics(ym, pm, thr)
        bbm = min(float(np.mean((v.transpose(0, 2, 1)[m] - ym) ** 2)) for v in base.values()) if base else float("nan")
        units.append({**key, "snr_bin": lab, "snr_lo_db": lo, "snr_hi_db": hi,
                      "n_units": int(m.sum()), **r,
                      "brier_skill_vs_best_baseline": 1 - r["brier"] / bbm if bbm > 0 else float("nan")})
    return ov, hz, cells, units


# ==========================================================================
#                          COMPLEXITY PROFILING
# ==========================================================================
def count_flops(fn, *args) -> float:
    """FLOPs of one call, counted by torch's FlopCounterMode (matmul, conv,
    bmm, attention). Element-wise ops are not counted (standard practice)."""
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


def profile_forecaster(name: str, cfg: Dict, C: int, model: Optional[nn.Module] = None) -> Dict:
    W, H = cfg["lookback"], cfg["horizon"]
    row = {"model": MODEL_LABEL.get(name, name), "model_key": name}
    if model is None:                       # non-learned baseline
        X = np.random.default_rng(0).integers(0, 2, (cfg["prof_batch"], W, C)).astype(np.float32)
        x1 = X[:1]
        f1 = lambda: baseline_forecast(name, x1, H, 4)
        fb = lambda: baseline_forecast(name, X, H, 4)
        fl = {"persistence": 0, "seasonal_naive": 0, "historical_mean": W * C}[name]
        row.update({"params": 0, "params_M": 0.0, "size_MB": 0.0,
                    "flops_per_sample": float(fl), "mflops_per_sample": fl / 1e6,
                    "macs_per_sample": fl / 2.0,
                    "cpu_latency_ms_batch1": _time_call(f1, cfg["prof_iters"], cfg["prof_warmup"]) * 1e3,
                    "cpu_latency_ms_per_sample_b%d" % cfg["prof_batch"]:
                        _time_call(fb, cfg["prof_iters"], cfg["prof_warmup"]) * 1e3 / cfg["prof_batch"]})
        return row
    m = copy.deepcopy(model).cpu()
    n_par = sum(p.numel() for p in m.parameters())
    m.train()                                # disables the fused MHA fast path (uncountable)
    fl = count_flops(m, torch.zeros(2, W, C)) / 2.0
    m.eval()
    x1, xb = torch.zeros(1, W, C), torch.zeros(cfg["prof_batch"], W, C)
    with torch.inference_mode():
        t1 = _time_call(lambda: m(x1), cfg["prof_iters"], cfg["prof_warmup"])
        tb = _time_call(lambda: m(xb), max(3, cfg["prof_iters"] // 5), cfg["prof_warmup"])
    row.update({"params": n_par, "params_M": n_par / 1e6,
                "size_MB": sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6,
                "flops_per_sample": fl, "mflops_per_sample": fl / 1e6, "macs_per_sample": fl / 2.0,
                "cpu_latency_ms_batch1": t1 * 1e3,
                "cpu_latency_ms_per_sample_b%d" % cfg["prof_batch"]: tb * 1e3 / cfg["prof_batch"]})
    if torch.cuda.is_available():
        g = copy.deepcopy(model).cuda().eval()
        x1g, xbg = x1.cuda(), xb.cuda()
        with torch.inference_mode():
            row["gpu_latency_ms_batch1"] = _time_call(lambda: g(x1g), cfg["prof_iters"],
                                                      cfg["prof_warmup"], torch.cuda.synchronize) * 1e3
            row["gpu_latency_ms_per_sample_b%d" % cfg["prof_batch"]] = _time_call(
                lambda: g(xbg), cfg["prof_iters"], cfg["prof_warmup"],
                torch.cuda.synchronize) * 1e3 / cfg["prof_batch"]
        del g
    return row


# ==========================================================================
#                               CSV / CACHE
# ==========================================================================
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
                o[k] = float(np.mean(v))
                o[k + "_std"] = float(np.std(v))
            elif k not in o:
                o[k] = rs[0][k]
        out.append(o)
    return out


def fingerprint(cfg: Dict, data_fp: str, *parts) -> str:
    keys = ["subchannel_hz", "noise_quantile", "eta_db", "lookback", "horizon", "stride",
            "val_frac", "test_frac", "input_repr", "epochs", "batch_size", "lr",
            "weight_decay", "patience", "d_model", "dropout", "fl_rounds",
            "fl_local_epochs", "fedprox_mu", "fl_patience", "fm_pretrain_epochs",
            "fm_mask_ratio", "seed"]
    s = json.dumps({k: cfg[k] for k in keys}, sort_keys=True, default=str) + data_fp + "|".join(map(str, parts))
    return hashlib.sha1(s.encode()).hexdigest()[:16]


# ==========================================================================
#                          STAGE 1 - ORCHESTRATION
# ==========================================================================
def _temporal_shards(X, Y, n=4):
    cut = np.linspace(0, X.shape[0], n + 1).astype(int)
    return [(X[a:b], Y[a:b]) for a, b in zip(cut[:-1], cut[1:]) if b - a > 8]


def train_one(mname: str, mode: str, seed: int, D: Dict, names: List[str], cfg, dev, C) -> Tuple[Dict, Dict]:
    """Returns {scenario: {'te','va','tr1'}} predictions and training info."""
    preds, info = {}, {"model": mname, "mode": mode, "seed": seed}
    fm = mname == "spectrumfm_f"

    def new_model(tag):
        set_seed(run_seed(cfg, mname, mode, seed, tag))
        return build_model(mname, cfg, C).to(dev)

    def pred_all(model, scen_list):
        for s in scen_list:
            preds[s] = {"te": predict_proba(model, D[s]["Xte"]),
                        "va": predict_proba(model, D[s]["Xva"]),
                        "tr1": predict_proba(model, D[s]["Xtr"])[:, 0, :]}

    if mode == "local":
        tot = 0.0
        for s in names:
            m = new_model(s)
            if fm:
                fit_central(m, D[s]["Xtr"], D[s]["Xtr"], D[s]["Xva"], D[s]["Xva"], cfg, dev,
                            "pretrain", cfg["fm_pretrain_epochs"], seed)
            r = fit_central(m, D[s]["Xtr"], D[s]["Ytr"], D[s]["Xva"], D[s]["Yva"], cfg, dev, seed=seed)
            tot += r["train_s"]
            pred_all(m, [s])
        info.update({"train_s": tot, "comm_bytes": 0})
    elif mode == "centralized":
        Xtr = torch.cat([D[s]["Xtr"] for s in names]); Ytr = torch.cat([D[s]["Ytr"] for s in names])
        Xva = torch.cat([D[s]["Xva"] for s in names]); Yva = torch.cat([D[s]["Yva"] for s in names])
        m = new_model("all")
        if fm:
            fit_central(m, Xtr, Xtr, Xva, Xva, cfg, dev, "pretrain", cfg["fm_pretrain_epochs"], seed)
        r = fit_central(m, Xtr, Ytr, Xva, Yva, cfg, dev, seed=seed)
        info.update({"train_s": r["train_s"], "epochs_run": r["epochs_run"], "comm_bytes": 0})
        pred_all(m, names)
    else:                                              # fedavg / fedprox
        if len(names) >= 2:
            clients = [(D[s]["Xtr"], D[s]["Ytr"]) for s in names]
            vals = [(D[s]["Xva"], D[s]["Yva"]) for s in names]
            ctype = "scenario"
        else:
            s = names[0]
            clients = _temporal_shards(D[s]["Xtr"], D[s]["Ytr"])
            vals = [(D[s]["Xva"], D[s]["Yva"])]
            ctype = "temporal (single scenario)"
        m = new_model("fl")
        comm = 0
        if fm:
            rp = fit_federated(m, [(x, x) for x, _ in clients], [(x, x) for x, _ in vals], cfg, dev,
                               mode, "pretrain", cfg["fm_pretrain_epochs"], seed)
            comm += rp["comm_bytes"]
        r = fit_federated(m, clients, vals, cfg, dev, mode, seed=seed)
        comm += r["comm_bytes"]
        info.update({"train_s": r["train_s"], "rounds_run": r["rounds_run"], "comm_bytes": comm,
                     "model_bytes": r["model_bytes"], "n_clients": r["n_clients"],
                     "client_type": ctype})
        pred_all(m, names)
    return preds, info


def _pool(names, D, P, key):
    return np.concatenate([P[s][key] for s in names], 0)


def run_forecasting(cfg, dev, D, names, C, edges, out_dir, data_fp):
    cache = os.path.join(out_dir, "cache"); os.makedirs(cache, exist_ok=True)
    thr = cfg["decision_threshold"]
    OV, HZ, CELL, UNIT, COMM = [], [], [], [], []
    store: Dict[Tuple, Dict] = {}
    base = {s: {b: {"te": baseline_forecast(b, D[s]["Xbin_te"], cfg["horizon"], D[s]["period"]),
                    "va": baseline_forecast(b, D[s]["Xbin_va"], cfg["horizon"], D[s]["period"])}
                for b in cfg["baselines"]} for s in names}

    def emit(model, mode, seed, P, extra):
        # per scenario
        for s in names + (["Combined"] if len(names) > 1 else []):
            sc = names if s == "Combined" else [s]
            y = np.concatenate([D[k]["y_te"] for k in sc]); E = np.concatenate([D[k]["E_te"] for k in sc])
            p = np.concatenate([P[k]["te"] for k in sc])
            yv = np.concatenate([D[k]["y_va"] for k in sc]); pv = np.concatenate([P[k]["va"] for k in sc])
            bp = {b: np.concatenate([base[k][b]["te"] for k in sc]) for b in cfg["baselines"]}
            ov, hz, ce, un = forecast_rows(model, mode, seed, s, y, p, E, bp, thr,
                                           f1_opt_threshold(yv, pv), edges, cfg, extra)
            ov["val_brier"] = float(np.mean((pv - yv) ** 2))
            OV.append(ov); HZ.extend(hz); CELL.extend(ce); UNIT.extend(un)

    log("\n[stage 1] baselines")
    for b in cfg["baselines"]:
        emit(b, "non-learned", 0, {s: base[s][b] for s in names}, {"train_s": 0.0})
        store[(b, "non-learned", 0)] = {s: {"te": base[s][b]["te"], "va": base[s][b]["va"]} for s in names}

    for seed in cfg["seeds"]:
        for mode in cfg["modes"]:
            for mname in cfg["forecast_models"]:
                fp = fingerprint(cfg, data_fp, mname, mode, seed)
                cp = os.path.join(cache, f"{mname}__{mode}__s{seed}__{fp}.npz")
                t0 = time.time()
                if cfg["resume"] and os.path.exists(cp):
                    z = np.load(cp, allow_pickle=True)
                    P = {s: {k: z[f"{s}__{k}"] for k in ("te", "va", "tr1")} for s in names}
                    info = json.loads(str(z["info"]))
                    log(f"  cached  {MODEL_LABEL[mname]:<14s} {mode:<12s} seed {seed}")
                else:
                    P, info = train_one(mname, mode, seed, D, names, cfg, dev, C)
                    np.savez_compressed(cp, info=json.dumps(info),
                                        **{f"{s}__{k}": v for s, d in P.items() for k, v in d.items()})
                    log(f"  trained {MODEL_LABEL[mname]:<14s} {mode:<12s} seed {seed} "
                        f"({time.time() - t0:.0f} s)")
                store[(mname, mode, seed)] = P
                emit(mname, mode, seed, P, {"train_s": info.get("train_s", float("nan")),
                                            "comm_MB": info.get("comm_bytes", 0) / 1e6})
                COMM.append({"model": MODEL_LABEL[mname], "mode": mode, "seed": seed,
                             "comm_MB": info.get("comm_bytes", 0) / 1e6,
                             "model_MB": info.get("model_bytes", 0) / 1e6,
                             "rounds_run": info.get("rounds_run", ""),
                             "n_clients": info.get("n_clients", ""),
                             "client_type": info.get("client_type", ""),
                             "train_s": info.get("train_s", float("nan"))})
    log("\n[stage 1] complexity")
    CX = [profile_forecaster(b, cfg, C) for b in cfg["baselines"]]
    for mname in cfg["forecast_models"]:
        CX.append(profile_forecaster(mname, cfg, C, build_model(mname, cfg, C)))
        log(f"  {MODEL_LABEL[mname]:<14s} params {CX[-1]['params']:>9,d}  "
            f"{CX[-1]['mflops_per_sample']:8.3f} MFLOPs  CPU b1 {CX[-1]['cpu_latency_ms_batch1']:.2f} ms")

    write_csv(os.path.join(out_dir, "forecast_overall.csv"), OV)
    write_csv(os.path.join(out_dir, "forecast_overall_agg.csv"),
              aggregate(OV, ["model", "model_key", "mode", "scenario"]))
    write_csv(os.path.join(out_dir, "forecast_per_horizon.csv"),
              aggregate(HZ, ["model", "model_key", "mode", "scenario", "horizon_step"]))
    write_csv(os.path.join(out_dir, "forecast_per_snr_cell.csv"),
              aggregate(CELL, ["model", "model_key", "mode", "scenario", "snr_bin"]))
    write_csv(os.path.join(out_dir, "forecast_per_snr_window.csv"),
              aggregate(UNIT, ["model", "model_key", "mode", "scenario", "snr_bin"]))
    write_csv(os.path.join(out_dir, "forecast_complexity.csv"), CX)
    write_csv(os.path.join(out_dir, "fl_communication.csv"), COMM)
    return store, OV


# ==========================================================================
#            HAND-OFF TO SCRIPTS 2 (ALLOCATION) AND 3 (SHARING)
# ==========================================================================
def write_stage2(cfg, out_dir, D, names, C, edges_snr, store, OV) -> None:
    """<out_dir>/stage2/: measured labels of every scenario + the forecasts the
    allocation / sharing scripts need (baselines and the best learned model of
    each training mode, chosen on VALIDATION Brier of the Combined set)."""
    d = os.path.join(out_dir, "stage2")
    os.makedirs(d, exist_ok=True)
    lab = {"names": json.dumps(names), "C": C, "edges_snr": edges_snr,
           "eta_db": cfg["eta_db"], "horizon": cfg["horizon"]}
    for s in names:
        for k in ("y_te", "E_te", "y_tr1", "E_tr1", "inr_busy", "inr_idle"):
            lab[f"{s}__{k}"] = D[s][k]
    np.savez_compressed(os.path.join(d, "labels.npz"), **lab)
    scope = "Combined" if len(names) > 1 else names[0]
    best = {}
    for mode in cfg["modes"]:
        cand = [(np.mean([r["val_brier"] for r in OV if r["model_key"] == m and r["mode"] == mode
                          and r["scenario"] == scope]), m)
                for m in cfg["forecast_models"]
                if any(r["model_key"] == m and r["mode"] == mode for r in OV)]
        if cand:
            best[mode] = min(cand)[1]
    files = []
    for (m, mode, seed), P in store.items():
        if not (mode == "non-learned" or best.get(mode) == m):
            continue
        fn = f"preds__{m}__{mode}__s{seed}.npz"
        arr = {}
        for s in names:
            arr[f"{s}__te"] = P[s]["te"].astype(np.float16)
            if "tr1" in P[s]:
                arr[f"{s}__tr1"] = P[s]["tr1"].astype(np.float16)
        np.savez_compressed(os.path.join(d, fn), **arr)
        files.append(fn)
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "scenarios": names, "C": C,
           "best_by_mode": best, "seeds": cfg["seeds"], "baselines": cfg["baselines"],
           "files": files, "labels": {k: cfg[k] for k in ("subchannel_hz", "noise_quantile",
                                                          "eta_db", "lookback", "horizon")},
           "model_labels": MODEL_LABEL, "cache_dir": os.path.join(out_dir, "cache")}
    with open(os.path.join(d, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    log(f"\n  stage-2 hand-off -> {d}")
    for mode, m in best.items():
        log(f"    best forecaster ({mode}, validation Brier): {MODEL_LABEL[m]}")


# ==========================================================================
#                                   MAIN
# ==========================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="2.4 GHz spectrum occupancy forecasting")
    ap.add_argument("--data", action="append", default=[], help="NAME=PATH (repeatable)")
    ap.add_argument("--root", help="data_root for auto-discovery")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--seeds", help="comma list, e.g. 0,1,2")
    ap.add_argument("--models", help="comma list of forecasting models")
    ap.add_argument("--modes", help="comma list: local,centralized,fedavg,fedprox")
    ap.add_argument("--device", help="auto | cpu | cuda")
    ap.add_argument("--quick", action="store_true", help="tiny run to test the code path")
    return ap.parse_args()


def main() -> None:
    global _LOG_FH
    a = parse_args()
    cfg = CONFIG
    if a.root:
        cfg["data_root"] = a.root
    if a.out:
        cfg["out_dir"] = a.out
    for kv in a.data:
        if "=" not in kv:
            sys.exit(f"--data expects NAME=PATH, got {kv!r}")
        k, v = kv.split("=", 1)
        cfg["scenarios"][k.strip()] = v.strip().strip('"')
    if a.seeds:
        cfg["seeds"] = [int(x) for x in a.seeds.split(",")]
    if a.models:
        cfg["forecast_models"] = [x.strip() for x in a.models.split(",")]
    if a.modes:
        cfg["modes"] = [x.strip() for x in a.modes.split(",")]
    if a.device:
        cfg["device"] = a.device
    if a.quick:
        cfg.update(QUICK)
    bad = [m for m in cfg["forecast_models"] if m not in MODEL_REGISTRY]
    if bad:
        sys.exit(f"unknown model name(s): {bad}")
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    _LOG_FH = open(os.path.join(out_dir, "run_log.txt"), "a", encoding="utf-8")
    dev = get_device(cfg)
    log("=" * 78)
    log(f"spectrum_occupancy_forecasting_2p4GHz  |  device {dev}  |  "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"labels: Q{cfg['noise_quantile'] * 100:g} noise floor (train span), eta = {cfg['eta_db']:g} dB, "
        f"{cfg['subchannel_hz'] / 1e6:g} MHz sub-channels; W={cfg['lookback']} H={cfg['horizon']}")
    log("=" * 78)
    with open(os.path.join(out_dir, "config_used.json"), "w") as fh:
        json.dump(cfg, fh, indent=1, default=str)

    log("\n[data] discovery")
    found = discover(cfg)
    srcs = {}
    for name, path in found.items():
        log(f"  [{name}] {path}")
        s = load_source(path, cfg["band_name"])
        if "error" in s:
            log(f"  [{name}] NOT USED: {s['error']}")
            continue
        srcs[name] = s
    if not srcs:
        log("\nNO SCENARIO DATA FOUND. Nothing was trained and no data was invented.\n"
            "Point the script at your exports, e.g.\n"
            "  python spectrum_occupancy_forecasting_2p4GHz.py --data IEEE=\"...\\IEEE\\export\\x.h5\" "
            "--data Office=\"...\" --data Home=\"...\" --data Campus=\"...\"\n"
            f"or set CONFIG['data_root'] (now {cfg['data_root']}).")
        sys.exit(1)
    spans = {n: freq_span(s, cfg["band_name"]) for n, s in srcs.items()}
    lo, hi = max(v[0] for v in spans.values()), min(v[1] for v in spans.values())
    sub = cfg["subchannel_hz"]
    C = int(math.floor((hi - lo) / sub + 1e-9))
    if C < 2:
        sys.exit(f"common span {lo / 1e6:.1f}-{hi / 1e6:.1f} MHz is too narrow")
    off = (hi - lo - C * sub) / 2
    edges = lo + off + np.arange(C + 1) * sub
    log(f"  common span {edges[0] / 1e6:.2f}-{edges[-1] / 1e6:.2f} MHz -> C = {C} sub-channels")

    D, cards, fp_parts = {}, [], []
    names = [n for n in cfg["scenarios"] if n in srcs] + [n for n in srcs if n not in cfg["scenarios"]]
    for n in names:
        P, t, ni = to_subchannels(srcs[n], cfg["band_name"], edges)
        sc = build_scenario(n, P, t, edges, cfg, ni, srcs[n]["src"])
        w = make_windows(sc, cfg)
        cards.append(sc["card"])
        tt = lambda a: _tt(a, dev)
        D[n] = {"Xtr": tt(w["X"][w["tr"]]), "Ytr": tt(w["Y"][w["tr"]]),
                "Xva": tt(w["X"][w["va"]]), "Yva": tt(w["Y"][w["va"]]),
                "Xte": tt(w["X"][w["te"]]),
                "Xbin_te": w["Xbin"][w["te"]], "Xbin_va": w["Xbin"][w["va"]],
                "y_te": w["Y"][w["te"]], "E_te": w["E"][w["te"]], "y_va": w["Y"][w["va"]],
                "y_tr1": w["Y"][w["tr"]][:, 0, :], "E_tr1": w["E"][w["tr"]][:, 0, :],
                "period": w["period"], "inr_busy": w["inr_busy"], "inr_idle": w["inr_idle"]}
        fp_parts.append(f"{n}:{sc['T']}:{float(sc['Y'].sum()):.0f}:{float(sc['E'].sum()):.3f}")
        del srcs[n]
    data_fp = hashlib.sha1("|".join(fp_parts).encode()).hexdigest()[:12]
    emax = max(float(D[n]["E_te"].max()) for n in names)
    edges_snr = snr_edges(emax, cfg)
    log(f"  SNR bins (dB): {', '.join(f'[{a:g},{b:g})' for a, b in zip(edges_snr[:-1], edges_snr[1:]))}"
        f"   (max test SNR {emax:.1f} dB)")
    write_csv(os.path.join(out_dir, "dataset_card.csv"), cards)

    log("\n[stage 1] occupancy forecasting")
    store, OV = run_forecasting(cfg, dev, D, names, C, edges_snr, out_dir, data_fp)
    write_stage2(cfg, out_dir, D, names, C, edges_snr, store, OV)
    log(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
