# Models

This page describes every model in `src/`: what it does, its architecture, its parameter and FLOP counts, and the paper it comes from.

**How to read the numbers**
- All counts come from `results/*/forecast_complexity.csv` and `results/*/alloc_complexity.csv`.
- MFLOPs are counted by PyTorch's `FlopCounterMode`:
  - forecasters: for one window of W = 32 sweeps × C = 32 subchannels;
  - allocators: for one slot with 48 links.

**Contents**
1. [Problem and notation](#1-problem-and-notation)
2. [Forecasting baselines](#2-forecasting-baselines)
3. [State of the art forecasters](#3-state-of-the-art-forecasters)
4. [Proposed forecasters: FRAT, MARS, VGM and DCS](#4-proposed-forecasters)
5. [Forecaster training: local, centralized, FedAvg, FedProx](#5-forecaster-training)
6. [Allocation problem](#6-allocation-problem)
7. [State of the art allocators](#7-state-of-the-art-allocators)
8. [Proposed allocators: BEGA-LS, FSF-BEGA, GRAIL](#8-proposed-allocators)
9. [Client ablation](#9-client-ablation)

---

## 1. Problem and notation

| Symbol | Meaning |
|---|---|
| C = 32 | subchannels of 2.5 MHz between 2402 and 2482 MHz |
| P[t, c] | power of subchannel c at sweep t, the linear mean of its 100 kHz bins |
| ν_c | noise floor: 10th percentile of P[·, c] over the **training** span only |
| s[t, c] = P − ν_c | cell SNR (excess over the noise floor, dB) |
| y[t, c] = 1[s ≥ η] | occupancy label, η = 6 dB |
| m[t, c] = (s − η)/10 | margin input of the proposed forecasters |
| W = 32, H = 8 | look-back window and horizon (sweeps) |

- **Split:** 60/15/25 % train/validation/test per environment, in time order, with a gap of W + H sweeps between parts.
- **Inputs:** the SOTA forecasters see the binary window only. The proposed forecasters also see the margin m.
- **Output:** every forecaster outputs logits z[h, c], and the busy probability is q = σ(z).

## 2. Forecasting baselines

File: `spectrum_occupancy_forecasting_2p4GHz.py` (`baseline_forecast`).

| Name | Forecast |
|---|---|
| Persistence | last observed state, repeated for all H steps |
| Seasonal naive | value one period earlier; the period comes from the autocorrelation (1 if none is found) |
| Historical mean | mean occupancy of the 32-sweep window |

## 3. State of the art forecasters

File: `spectrum_occupancy_forecasting_2p4GHz.py` (`MODEL_REGISTRY`). All use d_model = 64 and dropout 0.1, and are trained with binary cross-entropy on the logits.

| Model | Class | Architecture as implemented | Params | MFLOPs | Reference |
|---|---|---|---|---|---|
| DLinear | `DLinear` | moving-average decomposition (kernel 25) + one linear map W→H each for trend and seasonal parts | 528 | 0.03 | Zeng et al., AAAI 2023 |
| N-HiTS | `NHiTS` | 3 blocks with pooling 4/2/1; MLP 256-256 per block; backcast/forecast with linear interpolation | 1.31 M | 2.62 | Challu et al., AAAI 2023 |
| PatchTST | `PatchTST` | channel-independent patches (length 8, stride 4) → 2-layer pre-norm Transformer (4 heads, FFN 128) → linear head | 71.6 k | 30.6 | Nie et al., ICLR 2023 |
| iTransformer | `ITransformer` | one token per subchannel (W→64) → 2-layer Transformer over subchannels → linear head | 69.6 k | 4.88 | Liu et al., ICLR 2024 |
| Crossformer | `Crossformer` | segments of 4 sweeps → attention over time, then over subchannels (two-stage attention) → FFN → head | 55.2 k | 28.2 | Zhang and Yan, ICLR 2023 |
| NN-ResNet | `NNResNet` | 2-D CNN stem (32 ch) + 3 residual blocks over time × frequency → pooled → linear head | 1.11 M | 115.9 | Ren et al., IEEE TCCN 2022 |
| Bi-ConvLSTM-S2S | `BiConvLSTMSeq2Seq` | bidirectional ConvLSTM encoder (conv along frequency, 16 ch) + ConvLSTM decoder that feeds back its own prediction for H steps | 19.5 k | 19.9 | Wang et al., Sensors 24(5):1498, 2024 |
| TSB | `TSB` | 2-layer Transformer encoder → stacked Bi-LSTM encoder → learned queries → stacked Bi-LSTM decoder | 174.0 k | 4.88 | Pan et al., arXiv:2405.19138 (China Commun.) |
| DeepSense-F | `DeepSenseF` | the DeepSense CNN (3 conv blocks 16-32-32, BN, pooling) on the time × subchannel window + dense forecasting head *(adapted: the original reads raw IQ for sensing)* | 1.13 M | 21.4 | Uvaydov et al., IEEE INFOCOM 2021 |
| SpectrumFM-F | `SpectrumFMF` | conv stem + 3 conv-attention blocks; self-supervised pre-training (masked reconstruction, mask 0.3, + next-slot prediction) on the training split, then fine-tuned *(adapted: tokens are time steps, smaller model, no external corpus)* | 170.3 k | 9.17 | Liu et al., IEEE GLOBECOM 2025 |

## 4. Proposed forecasters

File: `novel_forecasting_2p4GHz.py`.

### Shared design

- **Residual on the strongest baseline.** Every model predicts a correction to the logit of the smoothed window mean, ȳ = (Σ_τ y_τ + ½)/(W + 1) (Krichevsky–Trofimov). The last layer is initialized to zero, so training starts exactly at the historical-mean baseline. (`base_logit`, `_zero_last`)
- **Per-environment calibration.** `ScenarioCalib` holds a bias per (horizon, subchannel) and a temperature for each environment: z ← z · e^{log T_e} + b_e[h, c]. These parameters are prefixed `loc_` and are **never averaged in federated learning**, so each client keeps its own calibration.
- **15 window statistics per subchannel** (`window_stats`):
  - the mean label and the mean margin over the last 1, 2, 4, 8, 16 and 32 sweeps (12 values);
  - the current run length;
  - the flip rate;
  - the margin volatility (standard deviation).

### FRAT: Frequency Relative Attention Transformer (`FRAT`)

- **Token:** one per subchannel, 79 values: 32 past labels, 32 past margins and the 15 statistics. It is embedded as `Linear(79 → 64)` plus a learned per-channel embedding.
- **Encoder:** 2 pre-norm attention blocks with 4 heads and FFN 128 (GELU).
- **Frequency bias:** each head adds a learned bias that depends only on the distance between subchannels, B[c, c'] = β_head[|c − c'|]. This learns how adjacent subchannels leak into each other and the 20 MHz Wi-Fi channel width (about 8 subchannels).
- **Head:** LayerNorm, then `Linear(64 → H)` (zero init), then the residual on the base logit, then calibration.
- **Size:** 76.0 k parameters, 5.07 MFLOPs.

### MARS: Margin Aware Residual on Statistics (`MARS`)

- **Frequency mixing:** depthwise `Conv1d` across subchannels on the 15 statistics, with kernel 9 (about one Wi-Fi channel).
- **Network:** the raw and convolved statistics (30 values) go through an MLP 30 → 128 → 64 → H (GELU, dropout, last layer zero init).
- **Output:** residual on the base logit, then calibration.
- **Size:** 13.9 k parameters, 0.81 MFLOPs. It is built for constrained sensing nodes.

### VGM: Volatility Gated Mixture (`VGM`, reported in the CSVs, not in the paper)

- **Fast expert:** reads the last 4 sweeps (labels and margins) plus a frequency convolution.
- **Slow expert:** reads the multiscale statistics.
- **Gate:** an MLP (3 → 16 → H) driven by run length, flip rate and volatility mixes the two experts per subchannel and horizon.
- **Size:** 4.5 k parameters, 0.21 MFLOPs.

### DCS: Drift Compensated Streaming inference (`dcs_stream`)

DCS is not a network. It is an online correction applied at inference to any of the models above, which gives FRAT-DCS, MARS-DCS and VGM-DCS.

- **Offset:** a per-environment offset b[h, c] is added to every logit.
- **Update:** the forecast made at window j for step h is verified at window j + h. The offset then takes one logistic-regression step:

  ```
  q[i, h, c] = σ(z[i, h, c] + b[h, c])
  b[h, c]   ← b[h, c] + γ · (y[j, h, c] − σ(z[j, h, c] + b[h, c])),   j = i − h
  ```

- **Step size:** γ is chosen per environment on the validation Brier score from {0, 0.01, 0.03, 0.1, 0.3}.
- **No leakage:** the offset state carries from the validation part into the test part, so only labels observed **before** each forecast are used.
- **Cost:** H × C additions and sigmoids per window. Nothing is communicated in federated learning.

## 5. Forecaster training

Settings are the same for the SOTA and proposed forecasters (`CONFIG` of each script):

- **Optimizer:** AdamW, learning rate 1e-3, weight decay 1e-4.
- **Batches and epochs:** batch 256, gradient clipping at 1, up to 40 epochs.
- **Early stopping:** on the validation Brier score, with patience 6.
- **Seeds:** 0, 1, 2.

| Mode | Description |
|---|---|
| `local` | one model per environment, trained and tested on that environment |
| `centralized` | one model on the pooled training windows of all four environments |
| `fedavg` | FedAvg (McMahan et al., AISTATS 2017): 4 clients, one per environment, 40 rounds × 2 local epochs, sample-weighted averaging, server early stopping (patience 8) |
| `fedprox` | FedProx (Li et al., MLSys 2020): same clients, proximal term μ = 0.01 |

**Selecting the forecaster for allocation.** The best forecaster of each mode is chosen by validation Brier score on the Combined set and written to `stage2/manifest.json`. Allocation uses it: N-HiTS (centralized) and PatchTST (FedAvg) among the SOTA models, and FRAT-DCS among the proposed ones.

## 6. Allocation problem

Files: `spectrum_allocation_2p4GHz.py`, `novel_allocation_2p4GHz.py`.

**Secondary network (simulated, seeded, identical for all methods)**
- 48 secondary (SU) links placed in a 15 × 15 m area, with link lengths 2 to 8 m.
- Path loss exponent 3, shadowing 4 dB, 40 dB SNR at 1 m.
- The spectrum sensor sits at the centre and stands for the primary-user (PU) receiver.
- 64 test topologies are used, one per block of 50 consecutive test slots.

**Primary users (measured)**
- The true occupancy y[c] of each test slot.
- The PU interference-to-noise ratio, INR[c] = 10^{s/10} − 1.

**Decision.** Each link picks a subchannel in {0, …, C − 1} or stays idle, and picks a power in [10⁻³, 1]. The decision may only use the forecast q[c] and the training statistics of the PU power.

**Utility of a slot (measured outcome)**

```
U = Σ_k log2(1 + p_k G_kk / (1 + INR[c_k] + Σ_{j≠k, c_j=c_k} p_j G_jk))  −  λ · #(SU transmissions on busy subchannels),   λ = 2
```

**Metrics.** Utility per slot, sum rate, regret against the oracle, collisions, PU harm rate, idle-cell utilization, outage, Jain fairness, energy efficiency, PU hit rate per SNR bin, and decision latency.

## 7. State of the art allocators

File: `spectrum_allocation_2p4GHz.py` (`METHODS`).

**Common setup for the learned policies**
- **Node features:** 2C + 1 values per link: the forecast, the expected INR and the direct gain.
- **Edges:** the cross gains.
- **Training:** on training slots, maximizing a differentiable mean-field utility, or the sum rate for UWMMSE.
- **Federated variant:** FedAvg with one client per environment, 30 rounds × 50 local steps.
- **Decoding (GNN policies):** the argmax plus 16 samples; the candidate with the best forecast-predicted utility is kept.

| Method | Class | Architecture | Params | MFLOPs | CPU ms/slot | Reference |
|---|---|---|---|---|---|---|
| Random | `RandomAlloc` | random subchannel, full power (lower reference) | 0 | 0 | 0.01 | – |
| Greedy-forecast | `GreedyForecast` | links in decreasing direct gain; each takes the subchannel with the largest positive marginal expected utility (own rate − harm to co-channel links − λq), or stays idle | 0 | 0.29 | 12.2 | – |
| Oracle | `OracleAlloc` | greedy + WMMSE with the **true** future (upper reference, not causal) | 0 | 0.72 | 13.1 | – |
| Greedy+WMMSE | `GreedyWMMSE` | greedy assignment + 30 WMMSE power-control iterations on the co-channel graph | 0 | 0.72 | 12.7 | Shi et al., IEEE TSP 2011, as used by Chowdhury et al. 2021 |
| Greedy+UWMMSE | `GreedyUWMMSE` | greedy + 4 unrolled WMMSE iterations; in each, a 2-layer GCN learns (a, b) in w ← a·w + b | 392 | 0.86 | 14.9 | Chowdhury et al., IEEE TWC 2021 |
| REGNN | `REGNNAlloc` | 3 graph-filter layers (3 taps, hidden 64), with the channel matrix as the graph shift; heads for subchannel (33 choices) and power | 39.5 k | 8.05 | 4.1 | Eisen and Ribeiro, IEEE TSP 2020 |
| WCGCN (MPGNN) | `WCGCNAlloc` | 3 message-passing layers with edge features in both directions and max aggregation | 69.0 k | 121.7 | 5.1 | Shen et al., IEEE JSAC 2021 |
| Agg-GNN | `AggGNNAlloc` | aggregation sequence [x, Sx, S²x, S³x] from neighbour exchanges only (decentralized) → MLP shared by all links | 43.9 k | 7.60 | 3.9 | Wang, Eisen, Ribeiro, arXiv:2107.01489 |
| Diffusion-GNN | `DiffusionGNN` | DDPM training (T = 100) to imitate oracle allocations; DDIM sampling (10 steps, 8 samples, best kept); noise predictor = WCGCN with a time embedding | 82.9 k | 9551.7 | 54.7 | Uslu et al., arXiv:2504.20277 |

## 8. Proposed allocators

File: `novel_allocation_2p4GHz.py` (`Engine`, `BEGALS`, `FSFBEGA`, `GRAIL`).

### Measured PU statistics (`scenario_stats`, training span only)

For each environment and subchannel, the following are measured on the training span:
- `atoms`: Q = 8 quantiles of the busy-cell INR, at levels (k + ½)/8;
- `inr_idle`: the mean INR on idle cells;
- `persist`: the share of consecutive busy sweeps whose power changes by at most 3 dB.

### BEGA-LS: Bayes Exact Greedy Allocation with Local Search

**Expected rate.** For link k on subchannel c, with power p and co-channel interference I, the expected rate is computed **exactly** over the busy/idle outcome and over the measured spread of PU power. It does not use one rate at the mean interference, which is biased by Jensen's inequality.

```
E[r] = (1 − q_c) log2(1 + p S_k / (1 + INR_idle,c + I))
     +  q_c Σ_a w_ca log2(1 + p S_k / (1 + X_ca + I))
```

Here X_ca are the atoms and w_ca their weights, each 1/8.

**Objective.** Σ E[r] − λ Σ q_c over the transmitting links.

**Solver (`Engine.solve`)**
1. **Greedy pass:** links in decreasing direct gain. Each link takes the subchannel with the largest positive marginal objective (its own expected rate − the expected loss it causes to links already on that subchannel − λq_c), or stays idle.
2. **Local search:** 2 best-response passes. Each link is removed and re-placed at its best subchannel, or left idle. Every move strictly increases the objective, so the search terminates.
3. **Power:** links transmit at full power or stay idle (power level set `[1.0]`).

**Cost.** No training, 0 parameters, 21.7 MFLOPs, 82 ms per slot on CPU.

### FSF-BEGA: Forecast Sensing Fusion + BEGA-LS (`FSF`, `build_ctx`)

**Fusion.** At decision time the sensor has just measured the current sweep. FSF combines that measurement with the forecast through a logistic model:

```
q̃_c = σ( w1 logit(q_c) + w2 (2y_c − 1)·ok + w3 clip((s_c − η)/10, −2, 4)·ok + w4·ok + b_c )
```

- y_c and s_c are the state and SNR sensed at forecast time.
- `ok` = 1 when that measurement exists.
- Parameters: 4 weights + 32 offsets b_c = **36 parameters per environment**.
- Fitting: on training slots with BCE and Adam (lr 0.05, 300 steps).
- They are **never shared in federated learning**.

**PU memory.** When the subchannel was busy in the last sweep, its measured INR is added as an extra atom with weight `persist`. The remaining atoms are scaled by 1 − `persist`.

**Then:** BEGA-LS runs on q̃ and the extended atoms.

**Cost.** 21.7 MFLOPs, 113 ms per slot on CPU.

### GRAIL: Graph Residual Allocation with Interference-aware Learning (reported in the CSVs, not in the paper)

- **Inputs:** an attention network over the links reads the analytic value of every action, taken from the FSF-BEGA engine, plus 3 gain features.
- **Network:** linear 64 → 2 attention blocks (4 heads) with an edge bias from the cross gains → FFN 128.
- **Output:** analytic values / τ plus a zero-initialized learned residual.
- **Training:** REINFORCE on the measured utility of training slots, with the FSF-BEGA solution as the baseline.
- **Test time:** from the FSF-BEGA solution, the argmax and 8 samples, the candidate with the best exact expected utility is kept; then one local-search pass.
- **Size:** 71.5 k parameters, 44.4 MFLOPs.

## 9. Client ablation

Files: `ablation_clients_forecasting_2p4GHz.py`, `ablation_clients_allocation_2p4GHz.py`.

**Grid**
- Clients: K ∈ {4, 8, 12, 16, 24, 32}.
- Partitions:
  - **IID:** each client gets a random share of the pooled training windows of all environments.
  - **Non IID:** each client holds one contiguous time block of one environment, with K/4 clients per environment.
- **Heterogeneity:** the total variation distance between a client's environment mix and the pooled mix is 0.013 to 0.035 for IID and 0.75 for non IID (`client_partitions.csv`).

**Federated training**
- **Forecasting:** every client runs 2 local epochs per round, so the computation per round does not depend on K. The `loc_` calibration of an environment is averaged only over the clients that hold that environment.
- **Allocation:** learned policies split 200 local steps per round across the K clients (at least 4 steps per client). FSF stays per environment.

**Models**
- **Forecasting:** MARS, FRAT and VGM (each with and without DCS); iTransformer, PatchTST, Crossformer, Bi-ConvLSTM-S2S, DLinear and N-HiTS.
- **Allocation:** BEGA-LS, FSF-BEGA and GRAIL; Greedy+WMMSE, Greedy+UWMMSE, Greedy-forecast, REGNN and Agg-GNN. Each is evaluated with the best SOTA forecaster (`best`) and the best proposed forecaster (`best_novel`) of the same configuration.

**Legacy versions.** `src/legacy/` holds the first versions of the two ablation scripts, which covered the proposed models only. The current allocation ablation can reuse their `parts/` results (`reuse_previous = True`); on a fresh run it computes everything itself.
