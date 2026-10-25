# Instructions

This guide explains how to install the code, place the data, run every experiment and read the outputs.

**Contents**
1. [Requirements](#1-requirements)
2. [Get the data](#2-get-the-data)
3. [Paths and environment variables](#3-paths-and-environment-variables)
4. [Pipeline and run order](#4-pipeline-and-run-order)
5. [Running each script](#5-running-each-script)
6. [Running on a SLURM cluster](#6-running-on-a-slurm-cluster)
7. [Outputs](#7-outputs)
8. [Changing the experiments](#8-changing-the-experiments)
9. [Reproducibility notes](#9-reproducibility-notes)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Requirements

- **Python:** 3.9 or newer.
- **PyTorch:** 2.0 or newer. A CUDA GPU is recommended for the training scripts. The CPU works too, but it is slow.
- **Python packages:** listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To install PyTorch with CUDA, follow https://pytorch.org/get-started/locally/ first, then run `pip install -r requirements.txt`.

## 2. Get the data

Download the four session files from IEEE DataPort:
https://ieee-dataport.org/documents/real-world-24-ghz-wi-fiism-spectrum-occupancy-dataset-four-indoor-environments
(doi:10.21227/2zh7-zk86).

Put them in `data/`. Either name them as below, or keep the original names; the scripts find them automatically (see `data/README.md`).

```
data/
├── ieee_event.h5        # IEEE technical event   (3945 sweeps, 1.39 s revisit)
├── office.h5            # INRS-EMT office        (3517 sweeps, 1.55 s)
├── home.h5              # home                   (6593 sweeps, 0.90 s)
└── indoor_campus.h5     # Concordia campus       (6197 sweeps, 0.90 s)
```

**Automatic discovery.** If a configured name is not found, every `.h5` file or session folder under `data/` is searched. A file is picked when its path contains:
- a scenario keyword: `ieee` or `event`, `office`, `home`, `campus`;
- a 2.4 GHz marker: `wifi24g`, `wifi_2p4GHz` or `2p4`.

So the original DataPort names, such as `office_wifi24g_20261015T165932-015.h5`, work as they are.

**Explicit paths.** You can also give each file on the command line:

```bash
python src/spectrum_occupancy_forecasting_2p4GHz.py \
  --data IEEE=/path/ieee.h5 --data Office=/path/office.h5 \
  --data Home=/path/home.h5 --data Campus=/path/campus.h5
```

## 3. Paths and environment variables

Every script resolves its folders in this order:
1. **Command-line options:** `--root`, `--data`, `--out`, `--forecast-dir`, `--novel-forecast-dir`.
2. **Environment variables:**

   | Variable | Meaning | Default |
   |---|---|---|
   | `SPECTRUM_DATA` | dataset folder | `<repo>/data` |
   | `SPECTRUM_RESULTS` | results root | `<repo>/results` |
   | `SPECTRUM_PROJECT` | base for relative paths | `<repo>` |

3. **Defaults:** the `CONFIG` dictionary at the top of each script.

Example of keeping the data and results outside the repository:

```bash
export SPECTRUM_DATA=/scratch/me/spectrum_data
export SPECTRUM_RESULTS=/scratch/me/spectrum_results
```

## 4. Pipeline and run order

```
data/*.h5
   │
   ├─(1) spectrum_occupancy_forecasting_2p4GHz.py ──► results/Forecasting/ (+ stage2/)
   │                                                         │
   ├─(2) novel_forecasting_2p4GHz.py ───────────────► results/Novel_Forecasting/ (+ stage2/)
   │                                                         │
   │   (3) spectrum_allocation_2p4GHz.py  ◄── stage2 of (1) ─┤──► results/Allocation/
   │   (4) novel_allocation_2p4GHz.py     ◄── stage2 of (1) and (2) ──► results/Novel_Allocation/
   │
   └─(5) ablation_clients_forecasting_2p4GHz.py ───► results/Ablation_Clients/Forecasting/ (+ stage2/)
           (6) ablation_clients_allocation_2p4GHz.py ◄── stage2 of (5) ──► results/Ablation_Clients/Allocation/
```

- **Order:** steps 1 and 2 must run before 3 and 4. Step 5 depends only on the data, and step 6 needs step 5.
- **stage2/:** each forecasting script writes a `stage2/` hand-off folder. It holds the measured labels, the forecasts of every model, and the best model of each training mode, chosen by validation Brier score. The allocation scripts read only this folder; they never touch the raw data.
- **Run everything:** use `bash scripts/run_all.sh` (Linux/macOS) or `scripts\run_all.ps1` (Windows PowerShell).

## 5. Running each script

All scripts run with **no arguments** and use their `CONFIG` block. The options below override it.

| # | Script | Main options |
|---|---|---|
| 1 | `spectrum_occupancy_forecasting_2p4GHz.py` | `--data NAME=PATH` (repeatable), `--root`, `--out`, `--seeds 0,1,2`, `--models dlinear,patchtst,...`, `--modes local,centralized,fedavg,fedprox`, `--device auto\|cpu\|cuda`, `--quick` |
| 2 | `novel_forecasting_2p4GHz.py` | same as 1; `--models mars,frat,vgm` |
| 3 | `spectrum_allocation_2p4GHz.py` | `--forecast-dir` (out dir of 1), `--out`, `--seeds`, `--methods random,greedy_forecast,oracle,greedy_wmmse,greedy_uwmmse,regnn,wcgcn,agg_gnn,diffusion_gnn`, `--modes centralized,fedavg`, `--device`, `--quick` |
| 4 | `novel_allocation_2p4GHz.py` | `--forecast-dir` (out dir of 1), `--novel-forecast-dir` (out dir of 2), `--out`, `--seeds`, `--methods bega_ls,fsf_bega,grail`, `--modes`, `--device`, `--quick` |
| 5 | `ablation_clients_forecasting_2p4GHz.py` | `--data`, `--root`, `--out`, `--seeds`, `--models`, `--sota-models itransformer,patchtst,...` (`''` = none), `--clients 4,8,32`, `--partitions iid,noniid`, `--algos fedavg,fedprox`, `--task i`, `--merge`, `--device`, `--quick` |
| 6 | `ablation_clients_allocation_2p4GHz.py` | `--forecast-dir` (out dir of 5), `--out`, `--seeds`, `--methods`, `--sota-methods`, `--clients`, `--partitions`, `--task i`, `--merge`, `--device`, `--quick` |

Examples:

```bash
# SOTA forecasters, only FedAvg, one seed, on the GPU
python src/spectrum_occupancy_forecasting_2p4GHz.py --modes fedavg --seeds 0 --device cuda

# the proposed forecasters only
python src/novel_forecasting_2p4GHz.py --models frat,mars

# the proposed allocators with both forecaster hand-offs
python src/novel_allocation_2p4GHz.py \
  --forecast-dir results/Forecasting --novel-forecast-dir results/Novel_Forecasting

# one cell of the client ablation (task 5 = K 12, non IID)
python src/ablation_clients_forecasting_2p4GHz.py --task 5
```

**`--quick`** runs a tiny configuration (few epochs and rounds, one seed). It is for checking that the code runs end to end on your data; its numbers are not results.

**Resuming.** The forecasting scripts keep a `cache/` of trained predictions. The ablation scripts write one `parts_v2/<tag>.pkl` per task. Re-running the same command continues where it stopped.

## 6. Running on a SLURM cluster

Templates for an Alliance Canada style cluster are in `scripts/slurm/`. Before submitting:
1. Set `--account` in each `.sbatch` file.
2. Set the module or environment lines to match your cluster.
3. Set `SPECTRUM_DATA` and `SPECTRUM_RESULTS` in `scripts/slurm/env.sh`.

```bash
bash scripts/slurm/submit_all.sh
```

This submits the six steps with the right dependencies:
- **Steps 1 and 2:** submitted first.
- **Steps 3 and 4:** start after 1 and 2 finish.
- **Step 5:** an array job.
- **Step 6:** an array job that starts after 5 finishes.

The two ablation scripts accept `--task i` for the array index:

| `--task` | Configuration |
|---|---|
| 0 to 11 | (K, partition): (4, IID), (4, non IID), (8, IID), …, (32, non IID) |
| 12 | centralized and local references |

After all tasks finish, run each ablation script once with `--merge` to rebuild the merged CSVs. The array templates do this automatically in a final dependent job.

## 7. Outputs

Every output folder also contains `run_log.txt` and `config_used.json`.

**Forecasting** (`results/Forecasting`, `results/Novel_Forecasting`, `results/Ablation_Clients/Forecasting`)

| File | Content |
|---|---|
| `dataset_card.csv` | per environment: sweeps, revisit, noise floor, occupancy (train and test), lag-1 autocorrelation, max excess |
| `forecast_overall.csv` / `_agg.csv` | model × mode × scenario (IEEE, Office, Home, Campus, Combined): accuracy, precision, recall, false alarm, F1, balanced accuracy, MCC, Brier, log loss, AUROC, AUPRC, ECE, skill against baselines, train time, communication; `_agg` = mean and std over seeds |
| `forecast_per_horizon.csv` | the same metrics for each step h = 1 to 8 |
| `forecast_per_snr_cell.csv`, `forecast_per_snr_window.csv` | metrics in 2 dB SNR bins (6–8, 8–10, …) |
| `forecast_complexity.csv` | parameters, MFLOPs per window, CPU/GPU latency (batch 1 and per sample at batch 256) |
| `fl_communication.csv` | MB exchanged and rounds per federated run |
| `reference_baselines.csv` | novel and ablation runs: persistence, seasonal naive and historical mean recomputed; they must equal the baseline rows of the SOTA run, which proves the same data and labels were used |
| `client_partitions.csv` | ablation only: size, occupancy and environment mix of each client, and the TV distance |
| `stage2/` | hand-off for the allocation scripts |

**Allocation** (`results/Allocation`, `results/Novel_Allocation`, `results/Ablation_Clients/Allocation`)

| File | Content |
|---|---|
| `alloc_overall.csv` / `_agg.csv` | method × forecaster × mode × scenario × horizon: sum rate, rate per link, utility per slot, collisions, collision rate, PU harm rate, idle utilization, access ratio, outage, Jain fairness, energy efficiency, regret to the oracle, decision latency |
| `alloc_per_snr.csv` | PU hit rate per measured PU SNR bin, and idle-cell utilization |
| `alloc_complexity.csv` | parameters, MFLOPs per slot, CPU/GPU latency per slot |
| `fl_communication.csv` | MB exchanged per federated allocation policy |

The `results/` folder is created by the scripts on the first run; it is not part of the repository.

## 8. Changing the experiments

Everything is set in the `CONFIG` dictionary at the top of each script. The most useful keys:

| Key | Script | Meaning |
|---|---|---|
| `noise_quantile` (0.10), `eta_db` (6) | forecasting | noise floor percentile and occupancy threshold |
| `lookback` (32), `horizon` (8) | forecasting | window W and horizon H |
| `val_frac` (0.15), `test_frac` (0.25) | forecasting | chronological split (gap of W + H between parts) |
| `epochs` (40), `patience` (6), `lr`, `weight_decay`, `d_model` (64) | forecasting | training |
| `fl_rounds` (40), `fl_local_epochs` (2), `fedprox_mu` (0.01) | forecasting | federated learning |
| `dcs_etas` | novel forecasting | candidate DCS step sizes, chosen on validation |
| `su_links` (48), `su_area_m` (15), `su_link_dist_m`, `su_pathloss_exp`, `su_shadow_db`, `su_snr_ref_db` | allocation | simulated secondary network |
| `collision_penalty` (λ = 2) | allocation | utility penalty per transmission on a busy subchannel |
| `ls_passes` (2) | novel allocation | local search passes of BEGA-LS / FSF-BEGA |
| `ab_clients`, `ab_partitions` | ablation | client grid |

**Keep the allocation scripts consistent.** The secondary-network keys and `collision_penalty` must stay identical in all allocation scripts, so that their rows can be compared.

## 9. Reproducibility notes

- **Seeds:** every run uses seed `20261019` combined with the model, mode and seed index. The results do not depend on the order in which models are run.
- **Determinism:** CUDA convolutions are set to deterministic mode. Small differences between GPU models and library versions are still possible.
- **No leakage:** the noise floor is estimated on the training span only. The windows are split chronologically with a W + H gap. DCS and FSF use only labels and measurements observed before each forecast.
- **Baseline check:** the baseline rows (persistence, seasonal naive, historical mean) in `reference_baselines.csv` of the novel and ablation runs must equal those of the SOTA run, which shows that all scripts used the same data and labels.

## 10. Troubleshooting

| Message | Fix |
|---|---|
| `NO FORECAST HAND-OFF FOUND` | Run the forecasting script first, or point `--forecast-dir` to its output folder (the folder containing `stage2/`). |
| `... not found -> trying auto-discovery` / `scenario skipped` | Check `data/` and the file names (scenario keyword + 2.4 GHz marker), or pass `--data NAME=PATH`. |
| `h5py is required` | `pip install h5py` |
| CUDA out of memory | Use `--device cpu`, or reduce `batch_size` / `alloc_batch` in `CONFIG`. |
| Ablation results missing a K | Run the missing `--task i`, then `--merge`. |
