# Drift Compensated Spectrum Forecasting and Bayes Exact Allocation (2.4 GHz)

Code for

> **A. Mahabub and S. Vakili**, *Drift Compensated Spectrum Forecasting and Bayes Exact Allocation on a Real World Four Environment 2.4 GHz Dataset under Federated Learning*, submitted to IEEE ICC 2027.

The repository covers the full pipeline on the real 2.4 GHz occupancy dataset
([IEEE DataPort, doi:10.21227/2zh7-zk86](https://ieee-dataport.org/documents/real-world-24-ghz-wi-fiism-spectrum-occupancy-dataset-four-indoor-environments)):

1. **Occupancy forecasting.** 10 state of the art forecasters plus the proposed **FRAT-DCS** and **MARS-DCS** (and VGM-DCS).
2. **Forecast driven spectrum allocation.** 8 state of the art allocators plus the proposed **BEGA-LS** and **FSF-BEGA** (and GRAIL).
3. **Federated learning study.** Local, centralized, FedAvg and FedProx training, with 4 to 32 clients under IID and non IID partitions.

Every script is a single file that runs with no arguments, uses deterministic seeds and writes CSV files.
**No synthetic spectrum data is ever generated.** A missing scenario is reported and skipped. Only the secondary network used for allocation (link positions and path loss) is simulated, with fixed seeds.

---

## Repository layout

```
spectrum-forecast-alloc-2p4ghz/
├── src/                                      # experiment code (run in this order)
│   ├── spectrum_occupancy_forecasting_2p4GHz.py   # 1  SOTA forecasters + baselines
│   ├── novel_forecasting_2p4GHz.py                # 2  FRAT / MARS / VGM (+DCS)
│   ├── spectrum_allocation_2p4GHz.py              # 3  SOTA allocators
│   ├── novel_allocation_2p4GHz.py                 # 4  BEGA-LS / FSF-BEGA / GRAIL
│   ├── ablation_clients_forecasting_2p4GHz.py     # 5  K = 4..32 clients, IID / non IID
│   ├── ablation_clients_allocation_2p4GHz.py      # 6  same grid for allocation
│   └── legacy/                                     # first ablation versions (novel models only)
├── scripts/                                  # run_all.sh, run_all.ps1, SLURM templates
├── data/                                     # put the four .h5 sessions here (see data/README.md)
├── docs/
│   ├── INSTRUCTIONS.md                       # step by step guide: install, data, run, outputs
│   ├── MODELS.md                             # every model: architecture, parameters, reference
│   └── RESULTS.md                            # headline numbers and where they come from
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Quick start

```bash
git clone https://github.com/<user>/spectrum-forecast-alloc-2p4ghz.git
cd spectrum-forecast-alloc-2p4ghz
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) download the dataset from IEEE DataPort into data/ (see data/README.md)
# 2) run the whole pipeline (hours on a GPU; see docs/INSTRUCTIONS.md for single steps)
bash scripts/run_all.sh                                    # Windows: scripts\run_all.ps1
# results are written as CSV files to results/ (created on the first run)
```

To check that the code paths work before a full run, use the smoke test:

```bash
python src/spectrum_occupancy_forecasting_2p4GHz.py --quick
```

`--quick` still needs the real data.

The complete guide covers paths, environment variables, the command-line options of each script, SLURM arrays, outputs and troubleshooting: **[docs/INSTRUCTIONS.md](docs/INSTRUCTIONS.md)**.

## Models

| Task | Proposed | State of the art (re-implemented) |
|---|---|---|
| Forecasting | **FRAT-DCS**, **MARS-DCS**, VGM-DCS | DLinear, PatchTST, iTransformer, Crossformer, N-HiTS, NN-ResNet, Bi-ConvLSTM-S2S, TSB, DeepSense-F, SpectrumFM-F; persistence, seasonal naive, historical mean |
| Allocation | **BEGA-LS**, **FSF-BEGA**, GRAIL | Greedy-forecast, Greedy+WMMSE, Greedy+UWMMSE, REGNN, WCGCN, Agg-GNN, Diffusion-GNN; random and oracle references |
| Training | local, centralized, FedAvg, FedProx, K ∈ {4, 8, 12, 16, 24, 32}, IID and non IID | |

Architectures, parameter counts and references are in **[docs/MODELS.md](docs/MODELS.md)**.

## Headline results (Combined test set, mean of three seeds)

| | Proposed | Best baseline |
|---|---|---|
| Forecast AUROC / Brier, centralized | FRAT-DCS **0.815 / 0.166** | iTransformer 0.807 / 0.169 |
| Forecast AUROC / Brier, FedAvg | FRAT-DCS **0.814 / 0.166** | PatchTST 0.803 / 0.171 |
| Forecast AUROC, 32 non IID clients | FRAT-DCS **0.807** | PatchTST 0.802 |
| Allocation utility per slot, centralized | BEGA-LS **154.4** (regret 8.6 %) | Greedy+WMMSE 138.2 (regret 18.2 %) |
| Allocation utility per slot, FedAvg | BEGA-LS **154.1** | Greedy+WMMSE 132.7 |

BEGA-LS achieves its higher utility by reusing busy subchannels more. As a result, it hits the primary user more often than Greedy+WMMSE (PU harm 0.73 vs 0.29), and FSF-BEGA partly reduces this.
Details and per-environment results are in **[docs/RESULTS.md](docs/RESULTS.md)**.

## Citation

Please cite the dataset (and the paper once published) (see [CITATION.cff](CITATION.cff)):

```bibtex
@misc{mahabub2026dataset,
  author       = {A. Mahabub and S. Vakili},
  title        = {A Real-World 2.4 {GHz} {Wi-Fi/ISM} Spectrum Occupancy Dataset from Four Indoor Environments for Forecasting, Allocation and Sharing},
  howpublished = {IEEE Dataport},
  year         = {2026},
  doi          = {10.21227/2zh7-zk86}
}
```

## Acknowledgment

Supported by The Natural Sciences and Engineering Research Council of Canada (NSERC) grant RGPIN-2024-06358.

## License

MIT, see [LICENSE](LICENSE). The dataset is distributed by IEEE DataPort under its own terms.
