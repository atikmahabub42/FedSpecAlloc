#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 novel_forecasting_2p4GHz.py
 NOVEL spectrum-occupancy forecasters, 2.4 GHz ISM, four measured scenarios
 (IEEE, Office, Home, Campus), per scenario AND combined; local, centralised,
 FedAvg and FedProx. Same data, labels, splits, metrics and CSV schema as
 spectrum_occupancy_forecasting_2p4GHz.py, so rows can be merged with the
 SOTA results directly (Q10 noise floor on the training span, eta = 6 dB,
 2.5 MHz sub-channels, W = 32, H = 8, chronological split with W+H gaps).
 Single file, no arguments. Deterministic seeds. CSV output. NO SYNTHETIC DATA.

 WHY THESE DESIGNS (from the SOTA results)
   * The window mean (historical mean) is the hardest baseline: the best SOTA
     model beats it by 2.5 % Brier (centralised) and 1.4 % (FedAvg).
     -> every novel model predicts a RESIDUAL on the logit of the
        (Krichevsky-Trofimov smoothed) window mean, with a zero-initialised
        output layer: training starts exactly at the strongest baseline.
   * Binary inputs throw away how far a cell is above/below eta; persistence
     is weak (Brier 0.32) because single samples flip, not because the level
     is uninformative.  -> the excess-power MARGIN (e - eta)/10 is a second
     input channel, plus multi-scale statistics.
   * One 20 MHz Wi-Fi channel covers ~8 sub-channels (2.5 MHz) and channels
     sit 5 MHz (2 sub-channels) apart. -> explicit frequency coupling.
   * Sites differ (occupancy 44-70 %) and drift (Campus 42 % -> 75 % from
     train to test). FedAvg lost most of the centralised gain.
     -> a per-scenario calibration layer (bias per sub-channel x horizon and a
        temperature) that is NEVER averaged in federated training (it stays at
        its client), and an optional streaming correction (DCS) that uses the
        labels which become observable as time advances (causal).

 MODELS (all novel compositions designed for this problem)
   MARS   Margin-Aware Residual-on-Statistics forecaster. Per sub-channel
          multi-scale occupancy/margin statistics (1..32 steps), run length,
          transition rate, margin volatility; depthwise convolution across
          sub-channels (Wi-Fi channel width); shared MLP -> residual logits.
   FRAT   Frequency-Relative Attention Transformer. Inverted tokens (one per
          sub-channel: binary history + margin history + statistics) with a
          LEARNED ATTENTION BIAS AS A FUNCTION OF SUB-CHANNEL DISTANCE (per
          head), i.e. Wi-Fi channel overlap is learned explicitly.
   VGM    Volatility-Gated Mixture. A fast expert (last 4 steps, margins) and a
          slow expert (multi-scale levels) mixed per sub-channel and horizon by
          a gate driven by the local volatility (flip rate, margin spread).
   +DCS   Drift-Compensated Streaming inference, applied to every model above
          (reported as MARS-DCS, FRAT-DCS, VGM-DCS). The prediction made at
          window j for horizon h is verified at window j + h; a per-scenario,
          per-horizon, per-sub-channel logit offset takes one online logistic
          step on that verified residual. Step size chosen on VALIDATION; the
          state is carried from validation into test (both are in the past of
          every test forecast). Uses only labels observed BEFORE the forecast
          time: causal, no test leakage. Costs no communication in FL.
 FL: all parameters are averaged except the per-scenario calibration layer
     ('loc_' parameters), which every client keeps (personalised FL).

 OUTPUT (same schema as the SOTA script; merge on model/mode/scenario)
   forecast_overall.csv / _agg.csv, forecast_per_horizon.csv,
   forecast_per_snr_cell.csv, forecast_per_snr_window.csv,
   forecast_complexity.csv, fl_communication.csv, dataset_card.csv,
   reference_baselines.csv  (persistence / seasonal naive / historical mean,
                             recomputed: must equal your SOTA run -> proof
                             that data and labels are identical)
   stage2/  hand-off for novel_allocation_2p4GHz.py / novel_sharing_2p4GHz.py
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
# ---- paths -------------------------------------------------------------
# Defaults follow the repository layout: data/ and results/ next to src/.
# Override with the environment variables SPECTRUM_DATA, SPECTRUM_RESULTS
# (and SPECTRUM_PROJECT for relative paths) or with the command-line options.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NARVAL = os.environ.get("SPECTRUM_PROJECT", _REPO)   # project root (name kept from the cluster runs)
_ON_NARVAL = False
_RES = os.environ.get("SPECTRUM_RESULTS", os.path.join(_REPO, "results"))
DATA_ROOT = os.environ.get("SPECTRUM_DATA", os.path.join(_REPO, "data"))
OUT_DIR = os.path.join(_RES, "Novel_Forecasting")

CONFIG: Dict = {
    "data_root": DATA_ROOT,
    "out_dir": OUT_DIR,
    "scenarios": {"IEEE": "ieee_event.h5", "Office": "office.h5",
                  "Home": "home.h5", "Campus": "indoor_campus.h5"},
    "scenario_aliases": {"IEEE": ["ieee", "event"], "Office": ["office"],
                         "Home": ["home"], "Campus": ["campus"]},
    "band_name": "ISM_2400",
    # ---- labels (identical to the SOTA run) --------------------------------
    "subchannel_hz": 2.5e6,
    "noise_quantile": 0.10,
    "eta_db": 6.0,
    "snr_bin_start_db": 6.0,
    "snr_bin_width_db": 2.0,
    "lookback": 32, "horizon": 8, "stride": 1,
    "val_frac": 0.15, "test_frac": 0.25,
    "input_repr": "binary",
    "baselines": ["persistence", "seasonal_naive", "historical_mean"],
    # ---- novel models -------------------------------------------------------
    "forecast_models": ["mars", "frat", "vgm"],
    "dcs": True,                      # also report the -DCS streaming variants
    "dcs_etas": [0.0, 0.01, 0.03, 0.1, 0.3],
    "modes": ["local", "centralized", "fedavg", "fedprox"],
    "seeds": [0, 1, 2],
    "seed": 20261019,
    "epochs": 40, "batch_size": 256, "lr": 1e-3, "weight_decay": 1e-4,
    "patience": 6, "d_model": 64, "dropout": 0.1, "grad_clip": 1.0,
    "fl_rounds": 40, "fl_local_epochs": 2, "fedprox_mu": 0.01, "fl_patience": 8,
    "decision_threshold": 0.5,
    # ---- profiling -----------------------------------------------------------
    "prof_iters": 50, "prof_warmup": 5, "prof_batch": 256,
    "device": "auto",
    "resume": True,
    "verbose": True,
}

QUICK = {"epochs": 2, "fl_rounds": 2, "fl_local_epochs": 1, "seeds": [0],
         "prof_iters": 3, "prof_warmup": 1, "patience": 1, "fl_patience": 1}

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
            p = os.path.expanduser(path)
            cands = [p] if os.path.isabs(p) else [os.path.abspath(p), os.path.join(root, p)]
            hit = next((c for c in cands if os.path.exists(c)), None)
            if hit:
                found[name] = hit
                continue
            log(f"  [{name}] not found at {' or '.join(cands)} -> trying auto-discovery")
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
#            REFERENCE BASELINES (for skill columns; identical to SOTA)
# ==========================================================================
def baseline_forecast(name: str, X: np.ndarray, H: int, period: int) -> np.ndarray:
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


MODEL_LABEL = {"persistence": "Persistence", "seasonal_naive": "Seasonal naive",
               "historical_mean": "Historical mean", "mars": "MARS", "frat": "FRAT",
               "vgm": "VGM", "mars_dcs": "MARS-DCS", "frat_dcs": "FRAT-DCS", "vgm_dcs": "VGM-DCS"}


# ==========================================================================
#                       SHARED BUILDING BLOCKS
# ==========================================================================
_SCALES = (1, 2, 4, 8, 16, 32)


def window_stats(xb: torch.Tensor, xe: torch.Tensor) -> torch.Tensor:
    """Per sub-channel statistics of the window -> [B, C, F].
    xb binary occupancy [B,W,C], xe margin (e-eta)/10 [B,W,C]."""
    W = xb.shape[1]
    ks = [k for k in _SCALES if k <= W]
    f = [xb[:, -k:].mean(1) for k in ks] + [xe[:, -k:].mean(1) for k in ks]
    same = (xb == xb[:, -1:]).flip(1).float()
    f.append(torch.cumprod(same, 1).sum(1) / W)                      # current run length
    f.append((xb[:, 1:] - xb[:, :-1]).abs().mean(1))                  # flip rate
    f.append(xe.std(1, unbiased=False))                               # margin volatility
    return torch.stack(f, -1)


def n_stats(W: int) -> int:
    return 2 * len([k for k in _SCALES if k <= W]) + 3


def base_logit(xb: torch.Tensor, H: int) -> torch.Tensor:
    """Logit of the Krichevsky-Trofimov smoothed window mean -> [B, H, C]."""
    W = xb.shape[1]
    p = (xb.sum(1) + 0.5) / (W + 1.0)
    return torch.logit(p)[:, None, :].expand(-1, H, -1)


class ScenarioCalib(nn.Module):
    """Per-scenario temperature and per-(horizon, sub-channel) bias. Its
    parameters start with 'loc_' and are NEVER averaged in federated training:
    each client keeps its own calibration (personalisation)."""

    def __init__(self, S, H, C):
        super().__init__()
        self.loc_bias = nn.Parameter(torch.zeros(S, H, C))
        self.loc_logtemp = nn.Parameter(torch.zeros(S))

    def forward(self, z, sid):
        return z * torch.exp(self.loc_logtemp[sid])[:, None, None] + self.loc_bias[sid]


def _zero_last(seq: nn.Sequential):
    last = [m for m in seq.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


# ==========================================================================
#                              NOVEL MODELS
# ==========================================================================
class MARS(nn.Module):
    """Margin-Aware Residual-on-Statistics forecaster (novel)."""

    def __init__(self, W, H, C, S, d_model=64, dropout=0.1, **kw):
        super().__init__()
        self.H = H
        Fs = n_stats(W)
        self.fconv = nn.Conv1d(Fs, Fs, 9, padding=4, groups=Fs)       # across sub-channels
        self.mlp = nn.Sequential(nn.Linear(2 * Fs, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, H))
        _zero_last(self.mlp)
        self.calib = ScenarioCalib(S, H, C)

    def forward(self, xb, xe, sid):
        s = window_stats(xb, xe)                                      # [B,C,F]
        sf = self.fconv(s.transpose(1, 2)).transpose(1, 2)
        z = self.mlp(torch.cat([s, sf], -1)).permute(0, 2, 1)         # [B,H,C]
        return self.calib(base_logit(xb, self.H) + z, sid)


class _RelBlock(nn.Module):
    def __init__(self, d, nhead, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, z, bias):
        h = self.n1(z)
        z = z + self.att(h, h, h, attn_mask=bias, need_weights=False)[0]
        return z + self.ff(self.n2(z))


class FRAT(nn.Module):
    """Frequency-Relative Attention Transformer (novel). One token per
    sub-channel; attention gets a learned, per-head bias that depends only on
    the distance between sub-channels (captures 20 MHz Wi-Fi channel overlap
    and the 5 MHz channel raster)."""

    def __init__(self, W, H, C, S, d_model=64, dropout=0.1, nhead=4, layers=2, **kw):
        super().__init__()
        self.H, self.C, self.nhead = H, C, nhead
        self.embed = nn.Linear(2 * W + n_stats(W), d_model)
        self.chan = nn.Parameter(torch.zeros(1, C, d_model))
        self.rel = nn.Parameter(torch.zeros(nhead, 2 * C - 1))
        self.blocks = nn.ModuleList([_RelBlock(d_model, nhead, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, H))
        _zero_last(self.head)
        self.calib = ScenarioCalib(S, H, C)
        idx = torch.arange(C)
        self.register_buffer("rel_idx", (idx[:, None] - idx[None, :] + C - 1), persistent=False)

    def forward(self, xb, xe, sid):
        B = xb.shape[0]
        tok = torch.cat([xb.transpose(1, 2), xe.transpose(1, 2), window_stats(xb, xe)], -1)
        z = self.embed(tok) + self.chan
        bias = self.rel[:, self.rel_idx]                                # [nhead, C, C]
        bias = bias[None].expand(B, -1, -1, -1).reshape(B * self.nhead, self.C, self.C)
        for blk in self.blocks:
            z = blk(z, bias)
        out = self.head(self.norm(z)).permute(0, 2, 1)                 # [B,H,C]
        return self.calib(base_logit(xb, self.H) + out, sid)


class VGM(nn.Module):
    """Volatility-Gated Mixture (novel): fast expert on the last 4 steps,
    slow expert on multi-scale levels, gate from local volatility."""

    def __init__(self, W, H, C, S, d_model=64, dropout=0.1, **kw):
        super().__init__()
        self.H = H
        Fs = n_stats(W)
        self.fctx = nn.Conv1d(8, 8, 9, padding=4, groups=8)
        self.fast = nn.Sequential(nn.Linear(16, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, H))
        self.slow = nn.Sequential(nn.Linear(Fs, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, H))
        self.gate = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, H))
        _zero_last(self.fast); _zero_last(self.slow)
        self.calib = ScenarioCalib(S, H, C)

    def forward(self, xb, xe, sid):
        s = window_stats(xb, xe)                                       # [B,C,F]
        last = torch.cat([xb[:, -4:], xe[:, -4:]], 1).transpose(1, 2)  # [B,C,8]
        ctx = self.fctx(last.transpose(1, 2)).transpose(1, 2)
        zf = self.fast(torch.cat([last, ctx], -1))
        zs = self.slow(s)
        g = torch.sigmoid(self.gate(s[..., -3:]))
        z = (g * zf + (1 - g) * zs).permute(0, 2, 1)
        return self.calib(base_logit(xb, self.H) + z, sid)


MODEL_REGISTRY = {"mars": MARS, "frat": FRAT, "vgm": VGM}


def build_model(name: str, cfg: Dict, C: int, S: int) -> nn.Module:
    return MODEL_REGISTRY[name](W=cfg["lookback"], H=cfg["horizon"], C=C, S=S,
                                d_model=cfg["d_model"], dropout=cfg["dropout"])


# ==========================================================================
#                     DCS: DRIFT-COMPENSATED STREAMING
# ==========================================================================
def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def dcs_stream(Z: np.ndarray, Y: np.ndarray, eta: float, beta0: Optional[np.ndarray] = None):
    """Z logits [N,H,C] of CONSECUTIVE windows (stride 1), Y labels [N,H,C].
    The forecast of window j for horizon h becomes observable at window j+h;
    before forecasting window i, the offset of horizon h takes one logistic
    step on the residual of window i-h. Returns (probabilities, final offset)."""
    N, H, C = Z.shape
    beta = np.zeros((H, C)) if beta0 is None else beta0.copy()
    P = np.empty_like(Z)
    for i in range(N):
        if eta > 0:
            for h in range(H):
                j = i - (h + 1)
                if j >= 0:
                    beta[h] += eta * (Y[j, h] - _sig(Z[j, h] + beta[h]))
        P[i] = _sig(Z[i] + beta)
    return P.astype(np.float32), beta


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
#                     TRAINING (centralised / local / FL)
# ==========================================================================
def _groups(model, wd):
    dec = [p for n, p in model.named_parameters() if "loc_" not in n]
    nod = [p for n, p in model.named_parameters() if "loc_" in n]
    return [{"params": dec, "weight_decay": wd}, {"params": nod, "weight_decay": 0.0}]


def _loss(model, b):
    xb, xe, sid, y = b
    return F.binary_cross_entropy_with_logits(model(xb, xe, sid), y)


def _batches(T: Dict, idx):
    return (T["Xb"][idx], T["Xe"][idx], T["sid"][idx], T["Y"][idx])


def train_epochs(model, T, cfg, epochs, rng, prox_ref=None, mu=0.0):
    opt = torch.optim.AdamW(_groups(model, cfg["weight_decay"]), lr=cfg["lr"])
    n, bs = T["Xb"].shape[0], cfg["batch_size"]
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for a in range(0, n, bs):
            j = torch.as_tensor(perm[a:a + bs], device=T["Xb"].device)
            if len(j) < 2:
                continue
            loss = _loss(model, _batches(T, j))
            if prox_ref is not None and mu > 0:
                loss = loss + 0.5 * mu * sum(((p - q) ** 2).sum() for (nm, p), q in
                                             zip(model.named_parameters(), prox_ref) if "loc_" not in nm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()


@torch.no_grad()
def eval_loss(model, T, bs=2048) -> float:
    model.eval()
    n, tot = T["Xb"].shape[0], 0.0
    for a in range(0, n, bs):
        j = torch.arange(a, min(n, a + bs), device=T["Xb"].device)
        tot += float(_loss(model, _batches(T, j))) * len(j)
    return tot / max(n, 1)


def cat_sets(sets: List[Dict]) -> Dict:
    return {k: torch.cat([s[k] for s in sets]) for k in sets[0]}


def fit_central(model, Ttr, Tva, cfg, seed) -> Dict:
    rng = np.random.default_rng(seed)
    best, state, bad, n, t0 = float("inf"), None, 0, 0, time.time()
    for ep in range(cfg["epochs"]):
        train_epochs(model, Ttr, cfg, 1, rng)
        va = eval_loss(model, Tva)
        n += 1
        if va < best - 1e-5:
            best, bad, state = va, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if state is not None:
        model.load_state_dict(state)
    return {"epochs_run": n, "best_val_loss": best, "train_s": time.time() - t0}


def model_bytes(model, shared_only=True) -> int:
    return sum(v.numel() * v.element_size() for k, v in model.state_dict().items()
               if not (shared_only and "loc_" in k))


def _aggregate(g_old: Dict, states: List[Dict], weights) -> Dict:
    """Shared parameters: sample-weighted average (FedAvg). 'loc_' parameters:
    every client contributes only ITS OWN change (they stay personal)."""
    w = np.asarray(weights, np.float64); w = w / w.sum()
    out = {}
    for k, v in g_old.items():
        if "loc_" in k:
            out[k] = v + sum(s[k] - v for s in states)
        elif v.dtype.is_floating_point:
            out[k] = sum(float(wi) * s[k] for wi, s in zip(w, states))
        else:
            out[k] = states[0][k].clone()
    return out


def fit_federated(model, clients: List[Dict], vals: List[Dict], cfg, algo, seed) -> Dict:
    mu = cfg["fedprox_mu"] if algo == "fedprox" else 0.0
    rng = np.random.default_rng(seed)
    g = copy.deepcopy(model.state_dict())
    best, state, bad, hist, t0 = float("inf"), None, 0, 0, time.time()
    nval = sum(v["Xb"].shape[0] for v in vals)
    for r in range(cfg["fl_rounds"]):
        states, ws = [], []
        for T in clients:
            model.load_state_dict(g)
            ref = [p.detach().clone() for p in model.parameters()] if mu > 0 else None
            train_epochs(model, T, cfg, cfg["fl_local_epochs"], rng, ref, mu)
            states.append(copy.deepcopy(model.state_dict()))
            ws.append(T["Xb"].shape[0])
        g = _aggregate(g, states, ws)
        model.load_state_dict(g)
        va = sum(eval_loss(model, V) * V["Xb"].shape[0] for V in vals) / max(nval, 1)
        hist += 1
        if va < best - 1e-5:
            best, bad, state = va, 0, copy.deepcopy(g)
        else:
            bad += 1
            if bad >= cfg["fl_patience"]:
                break
    if state is not None:
        model.load_state_dict(state)
    nb = model_bytes(model)
    return {"rounds_run": hist, "best_val_loss": best, "train_s": time.time() - t0,
            "comm_bytes": 2 * nb * len(clients) * hist, "model_bytes": nb,
            "n_clients": len(clients)}


@torch.no_grad()
def predict_logits(model, T, bs=2048) -> np.ndarray:
    model.eval()
    n, out = T["Xb"].shape[0], []
    for a in range(0, n, bs):
        j = torch.arange(a, min(n, a + bs), device=T["Xb"].device)
        xb, xe, sid, _ = _batches(T, j)
        out.append(model(xb, xe, sid).float().cpu().numpy())
    return np.concatenate(out, 0)


def train_one(mname, mode, seed, D, names, cfg, dev, C, S):
    """-> {scenario: {'zte','zva','ztr1'}} logits, info."""
    out, info = {}, {"model": mname, "mode": mode, "seed": seed}

    def new(tag):
        set_seed(run_seed(cfg, mname, mode, seed, tag))
        return build_model(mname, cfg, C, S).to(dev)

    def pred(m, scen):
        for s in scen:
            out[s] = {"zte": predict_logits(m, D[s]["te"]), "zva": predict_logits(m, D[s]["va"]),
                      "ztr1": predict_logits(m, D[s]["tr"])[:, 0, :]}

    if mode == "local":
        tot = 0.0
        for s in names:
            m = new(s)
            tot += fit_central(m, D[s]["tr"], D[s]["va"], cfg, seed)["train_s"]
            pred(m, [s])
        info.update({"train_s": tot, "comm_bytes": 0})
    elif mode == "centralized":
        m = new("all")
        r = fit_central(m, cat_sets([D[s]["tr"] for s in names]),
                        cat_sets([D[s]["va"] for s in names]), cfg, seed)
        info.update({"train_s": r["train_s"], "epochs_run": r["epochs_run"], "comm_bytes": 0})
        pred(m, names)
    else:
        if len(names) >= 2:
            clients, vals, ctype = [D[s]["tr"] for s in names], [D[s]["va"] for s in names], "scenario"
        else:
            T = D[names[0]]["tr"]
            cut = np.linspace(0, T["Xb"].shape[0], 5).astype(int)
            clients = [{k: v[a:b] for k, v in T.items()} for a, b in zip(cut[:-1], cut[1:])]
            vals, ctype = [D[names[0]]["va"]], "temporal (single scenario)"
        m = new("fl")
        r = fit_federated(m, clients, vals, cfg, mode, seed)
        info.update({"train_s": r["train_s"], "rounds_run": r["rounds_run"],
                     "comm_bytes": r["comm_bytes"], "model_bytes": r["model_bytes"],
                     "n_clients": r["n_clients"], "client_type": ctype})
        pred(m, names)
    return out, info


# ==========================================================================
#                      COMPLEXITY (params, FLOPs, latency)
# ==========================================================================
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


def profile_novel(name, cfg, C, S) -> Dict:
    W = cfg["lookback"]
    m = build_model(name, cfg, C, S).cpu()
    n_par = sum(p.numel() for p in m.parameters())
    mk = lambda B: (torch.zeros(B, W, C), torch.zeros(B, W, C), torch.zeros(B, dtype=torch.long))
    m.train()
    fl = count_flops(lambda: m(*mk(2))) / 2.0
    m.eval()
    x1, xb = mk(1), mk(cfg["prof_batch"])
    row = {"model": MODEL_LABEL[name], "model_key": name, "params": n_par, "params_M": n_par / 1e6,
           "params_shared_in_FL": sum(p.numel() for nm, p in m.named_parameters() if "loc_" not in nm),
           "size_MB": sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6,
           "flops_per_sample": fl, "mflops_per_sample": fl / 1e6, "macs_per_sample": fl / 2.0}
    with torch.inference_mode():
        row["cpu_latency_ms_batch1"] = _time_call(lambda: m(*x1), cfg["prof_iters"], cfg["prof_warmup"]) * 1e3
        row["cpu_latency_ms_per_sample_b%d" % cfg["prof_batch"]] = _time_call(
            lambda: m(*xb), max(3, cfg["prof_iters"] // 5), cfg["prof_warmup"]) * 1e3 / cfg["prof_batch"]
        if torch.cuda.is_available():
            g = copy.deepcopy(m).cuda().eval()
            g1 = tuple(t.cuda() for t in x1); gb = tuple(t.cuda() for t in xb)
            row["gpu_latency_ms_batch1"] = _time_call(lambda: g(*g1), cfg["prof_iters"], cfg["prof_warmup"],
                                                      torch.cuda.synchronize) * 1e3
            row["gpu_latency_ms_per_sample_b%d" % cfg["prof_batch"]] = _time_call(
                lambda: g(*gb), cfg["prof_iters"], cfg["prof_warmup"], torch.cuda.synchronize) * 1e3 / cfg["prof_batch"]
    row["dcs_extra_cost"] = "H x C additions + H x C sigmoids per window (streaming offset)"
    return row


# ==========================================================================
#                             CSV / CACHE
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
                o[k] = float(np.mean(v)); o[k + "_std"] = float(np.std(v))
            elif k not in o:
                o[k] = rs[0][k]
        out.append(o)
    return out


def fingerprint(cfg: Dict, data_fp: str, *parts) -> str:
    keys = ["subchannel_hz", "noise_quantile", "eta_db", "lookback", "horizon", "stride",
            "val_frac", "test_frac", "epochs", "batch_size", "lr", "weight_decay", "patience",
            "d_model", "dropout", "fl_rounds", "fl_local_epochs", "fedprox_mu", "fl_patience", "seed"]
    s = json.dumps({k: cfg[k] for k in keys}, sort_keys=True, default=str) + data_fp + "|".join(map(str, parts))
    return "nv" + hashlib.sha1(s.encode()).hexdigest()[:14]


# ==========================================================================
#                              ORCHESTRATION
# ==========================================================================
def run_forecasting(cfg, dev, D, names, C, S, edges, out_dir, data_fp):
    cache = os.path.join(out_dir, "cache"); os.makedirs(cache, exist_ok=True)
    thr, H = cfg["decision_threshold"], cfg["horizon"]
    OV, HZ, CELL, UNIT, COMM, REF = [], [], [], [], [], []
    store: Dict[Tuple, Dict] = {}
    base = {s: {b: {"te": baseline_forecast(b, D[s]["Xbin_te"], H, D[s]["period"]),
                    "va": baseline_forecast(b, D[s]["Xbin_va"], H, D[s]["period"])}
                for b in cfg["baselines"]} for s in names}

    def emit(model, mode, seed, P, extra, sink_ov):
        for s in names + (["Combined"] if len(names) > 1 else []):
            sc = names if s == "Combined" else [s]
            y = np.concatenate([D[k]["y_te"] for k in sc]); E = np.concatenate([D[k]["E_te"] for k in sc])
            p = np.concatenate([P[k]["te"] for k in sc])
            yv = np.concatenate([D[k]["y_va"] for k in sc]); pv = np.concatenate([P[k]["va"] for k in sc])
            bp = {b: np.concatenate([base[k][b]["te"] for k in sc]) for b in cfg["baselines"]}
            ov, hz, ce, un = forecast_rows(model, mode, seed, s, y, p, E, bp, thr,
                                           f1_opt_threshold(yv, pv), edges, cfg, extra)
            ov["val_brier"] = float(np.mean((pv - yv) ** 2))
            sink_ov.append(ov)
            if sink_ov is OV:
                HZ.extend(hz); CELL.extend(ce); UNIT.extend(un)

    for b in cfg["baselines"]:
        emit(b, "non-learned", 0, {s: base[s][b] for s in names}, {}, REF)
        store[(b, "non-learned", 0)] = {s: {"te": base[s][b]["te"]} for s in names}

    for seed in cfg["seeds"]:
        for mode in cfg["modes"]:
            for mname in cfg["forecast_models"]:
                fp = fingerprint(cfg, data_fp, mname, mode, seed)
                cp = os.path.join(cache, f"{mname}__{mode}__s{seed}__{fp}.npz")
                t0 = time.time()
                if cfg["resume"] and os.path.exists(cp):
                    z = np.load(cp, allow_pickle=True)
                    Z = {s: {k: z[f"{s}__{k}"] for k in ("zte", "zva", "ztr1")} for s in names}
                    info = json.loads(str(z["info"]))
                    log(f"  cached  {MODEL_LABEL[mname]:<6s} {mode:<12s} seed {seed}")
                else:
                    Z, info = train_one(mname, mode, seed, D, names, cfg, dev, C, S)
                    np.savez_compressed(cp, info=json.dumps(info),
                                        **{f"{s}__{k}": v for s, d in Z.items() for k, v in d.items()})
                    log(f"  trained {MODEL_LABEL[mname]:<6s} {mode:<12s} seed {seed} ({time.time() - t0:.0f} s)")
                extra = {"train_s": info.get("train_s", float("nan")),
                         "comm_MB": info.get("comm_bytes", 0) / 1e6}
                P = {s: {"te": _sig(Z[s]["zte"]).astype(np.float32), "va": _sig(Z[s]["zva"]).astype(np.float32),
                         "tr1": _sig(Z[s]["ztr1"]).astype(np.float32)} for s in names}
                store[(mname, mode, seed)] = P
                emit(mname, mode, seed, P, extra, OV)
                if cfg["dcs"]:
                    Pd, etas = {}, {}
                    for s in names:                       # eta chosen on validation (causal stream)
                        best = min(cfg["dcs_etas"], key=lambda e: float(np.mean(
                            (dcs_stream(Z[s]["zva"], D[s]["y_va"], e)[0] - D[s]["y_va"]) ** 2)))
                        pva, beta = dcs_stream(Z[s]["zva"], D[s]["y_va"], best)
                        pte, _ = dcs_stream(Z[s]["zte"], D[s]["y_te"], best, beta)
                        Pd[s] = {"te": pte, "va": pva, "tr1": P[s]["tr1"]}
                        etas[s] = best
                    store[(mname + "_dcs", mode, seed)] = Pd
                    emit(mname + "_dcs", mode, seed, Pd, {**extra, **{f"dcs_eta_{s}": e for s, e in etas.items()}}, OV)
                COMM.append({"model": MODEL_LABEL[mname], "mode": mode, "seed": seed,
                             "comm_MB_shared_params": info.get("comm_bytes", 0) / 1e6,
                             "model_MB_shared": info.get("model_bytes", 0) / 1e6,
                             "rounds_run": info.get("rounds_run", ""), "n_clients": info.get("n_clients", ""),
                             "client_type": info.get("client_type", ""), "train_s": info.get("train_s", float("nan"))})
            write_outputs(out_dir, OV, HZ, CELL, UNIT, COMM, REF, None)
    log("\n[complexity]")
    CX = []
    for mname in cfg["forecast_models"]:
        try:
            CX.append(profile_novel(mname, cfg, C, S))
            log(f"  {MODEL_LABEL[mname]:<6s} params {CX[-1]['params']:>9,d}  "
                f"{CX[-1]['mflops_per_sample']:8.3f} MFLOPs  CPU b1 {CX[-1]['cpu_latency_ms_batch1']:.2f} ms")
        except Exception as e:
            log(f"  profiling {mname} failed: {e}")
    write_outputs(out_dir, OV, HZ, CELL, UNIT, COMM, REF, CX)
    return store, OV


def write_outputs(out_dir, OV, HZ, CELL, UNIT, COMM, REF, CX):
    write_csv(os.path.join(out_dir, "forecast_overall.csv"), OV)
    write_csv(os.path.join(out_dir, "forecast_overall_agg.csv"), aggregate(OV, ["model", "model_key", "mode", "scenario"]))
    write_csv(os.path.join(out_dir, "forecast_per_horizon.csv"),
              aggregate(HZ, ["model", "model_key", "mode", "scenario", "horizon_step"]))
    write_csv(os.path.join(out_dir, "forecast_per_snr_cell.csv"),
              aggregate(CELL, ["model", "model_key", "mode", "scenario", "snr_bin"]))
    write_csv(os.path.join(out_dir, "forecast_per_snr_window.csv"),
              aggregate(UNIT, ["model", "model_key", "mode", "scenario", "snr_bin"]))
    write_csv(os.path.join(out_dir, "reference_baselines.csv"), REF)
    write_csv(os.path.join(out_dir, "fl_communication.csv"), COMM)
    if CX:
        write_csv(os.path.join(out_dir, "forecast_complexity.csv"), CX)


def write_stage2(cfg, out_dir, D, names, C, edges_snr, store, OV) -> None:
    """Same format as the SOTA script's stage2/: labels + forecasts of the
    best NOVEL model of each mode (validation Brier, Combined)."""
    d = os.path.join(out_dir, "stage2")
    os.makedirs(d, exist_ok=True)
    lab = {"names": json.dumps(names), "C": C, "edges_snr": edges_snr, "eta_db": cfg["eta_db"],
           "horizon": cfg["horizon"]}
    for s in names:
        for k in ("y_te", "E_te", "y_tr1", "E_tr1", "inr_busy", "inr_idle"):
            lab[f"{s}__{k}"] = D[s][k]
    np.savez_compressed(os.path.join(d, "labels.npz"), **lab)
    scope = "Combined" if len(names) > 1 else names[0]
    keys = sorted({k for k, m, s in store if m != "non-learned"})
    best = {}
    for mode in cfg["modes"]:
        cand = [(np.mean([r["val_brier"] for r in OV if r["model_key"] == k and r["mode"] == mode
                          and r["scenario"] == scope]), k) for k in keys
                if any(r["model_key"] == k and r["mode"] == mode for r in OV)]
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
           "best_by_mode": best, "seeds": cfg["seeds"], "baselines": cfg["baselines"], "files": files,
           "labels": {k: cfg[k] for k in ("subchannel_hz", "noise_quantile", "eta_db", "lookback", "horizon")},
           "model_labels": MODEL_LABEL, "cache_dir": ""}
    with open(os.path.join(d, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    log(f"\n  stage-2 hand-off -> {d}")
    for mode, m in best.items():
        log(f"    best novel forecaster ({mode}, validation Brier): {MODEL_LABEL[m]}")


# ==========================================================================
#                                   MAIN
# ==========================================================================
def main() -> None:
    global _LOG_FH
    ap = argparse.ArgumentParser(description="Novel 2.4 GHz occupancy forecasters")
    ap.add_argument("--data", action="append", default=[], help="NAME=PATH (repeatable)")
    ap.add_argument("--root"); ap.add_argument("--out"); ap.add_argument("--seeds")
    ap.add_argument("--models"); ap.add_argument("--modes"); ap.add_argument("--device")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = CONFIG
    if a.root:
        cfg["data_root"] = a.root
    if a.out:
        cfg["out_dir"] = a.out
    for kv in a.data:
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
        sys.exit(f"unknown model(s) {bad}; choose from {list(MODEL_REGISTRY)}")
    out_dir = os.path.abspath(os.path.expanduser(cfg["out_dir"]))
    os.makedirs(out_dir, exist_ok=True)
    _LOG_FH = open(os.path.join(out_dir, "run_log.txt"), "a", encoding="utf-8")
    dev = get_device(cfg)
    log("=" * 78)
    log(f"novel_forecasting_2p4GHz  |  device {dev}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
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
        log("\nNO SCENARIO DATA FOUND. Nothing was trained and no data was invented.")
        sys.exit(1)
    spans = {n: freq_span(s, cfg["band_name"]) for n, s in srcs.items()}
    lo, hi = max(v[0] for v in spans.values()), min(v[1] for v in spans.values())
    sub = cfg["subchannel_hz"]
    C = int(math.floor((hi - lo) / sub + 1e-9))
    edges = lo + (hi - lo - C * sub) / 2 + np.arange(C + 1) * sub
    log(f"  common span {edges[0] / 1e6:.2f}-{edges[-1] / 1e6:.2f} MHz -> C = {C} sub-channels")

    names = [n for n in cfg["scenarios"] if n in srcs] + [n for n in srcs if n not in cfg["scenarios"]]
    S = len(names)
    D, cards, fp_parts = {}, [], []
    W, eta = cfg["lookback"], cfg["eta_db"]
    for si, n in enumerate(names):
        P, t, ni = to_subchannels(srcs[n], cfg["band_name"], edges)
        sc = build_scenario(n, P, t, edges, cfg, ni, srcs[n]["src"])
        w = make_windows(sc, cfg)
        cards.append(sc["card"])
        idx_in = w["start"][:, None] + np.arange(W)[None, :]
        Xe = np.clip((sc["E"][idx_in] - eta) / 10.0, -2.0, 4.0).astype(np.float32)

        def pack(ix):
            return {"Xb": _tt(w["Xbin"][ix], dev), "Xe": _tt(Xe[ix], dev),
                    "sid": torch.full((len(ix),), si, dtype=torch.long, device=dev),
                    "Y": _tt(w["Y"][ix], dev)}
        D[n] = {"tr": pack(w["tr"]), "va": pack(w["va"]), "te": pack(w["te"]),
                "Xbin_te": w["Xbin"][w["te"]], "Xbin_va": w["Xbin"][w["va"]],
                "y_te": w["Y"][w["te"]], "E_te": w["E"][w["te"]], "y_va": w["Y"][w["va"]],
                "y_tr1": w["Y"][w["tr"]][:, 0, :], "E_tr1": w["E"][w["tr"]][:, 0, :],
                "period": w["period"], "inr_busy": w["inr_busy"], "inr_idle": w["inr_idle"]}
        fp_parts.append(f"{n}:{sc['T']}:{float(sc['Y'].sum()):.0f}:{float(sc['E'].sum()):.3f}")
        del srcs[n]
    data_fp = hashlib.sha1("|".join(fp_parts).encode()).hexdigest()[:12]
    emax = max(float(D[n]["E_te"].max()) for n in names)
    edges_snr = snr_edges(emax, cfg)
    write_csv(os.path.join(out_dir, "dataset_card.csv"), cards)

    log("\n[novel forecasting]")
    store, OV = run_forecasting(cfg, dev, D, names, C, S, edges_snr, out_dir, data_fp)
    write_stage2(cfg, out_dir, D, names, C, edges_snr, store, OV)
    log(f"\nDone. Results in {out_dir}")


def _tt(a, dev):
    return torch.as_tensor(a, dtype=torch.float32, device=dev)


if __name__ == "__main__":
    main()
