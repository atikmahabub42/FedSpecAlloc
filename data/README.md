# Data

The 2.4 GHz occupancy dataset is **not** stored in this repository. Download it from IEEE DataPort:

**A Real-World 2.4 GHz Wi-Fi/ISM Spectrum Occupancy Dataset from Four Indoor Environments for Forecasting, Allocation and Sharing**
A. Mahabub and S. Vakili, IEEE DataPort, 2026, doi:[10.21227/2zh7-zk86](https://doi.org/10.21227/2zh7-zk86)

Place the four session files here, either renamed:

```
data/
├── ieee_event.h5
├── office.h5
├── home.h5
└── indoor_campus.h5
```

or with their original names. Original names are found automatically when the file path contains:
- a scenario keyword: `ieee` or `event`, `office`, `home`, `campus`;
- a 2.4 GHz marker: `wifi24g`, `wifi_2p4GHz` or `2p4`.

Examples: `ieee_wifi24g_20261014T203014-013.h5`, `office_wifi24g_20261015T165932-015.h5`.

## Recording setup

| Item | Value |
|---|---|
| Receiver | Ettus USRP B205mini-i with a dual-band 2.4/5 GHz omnidirectional antenna |
| Sweep | 7 tiles at 15 MS/s; Welch PSD (2048-point FFT, 50 % overlap) per dwell, plus periodic raw IQ recordings |
| Band used here | 2402 to 2482 MHz, mapped onto 32 subchannels of 2.5 MHz |
| Sessions | 90 minutes each, Montréal, 14 to 18 October 2026 |

## Sessions

| Environment | Sweeps | Median revisit | Occupancy* |
|---|---|---|---|
| IEEE event | 3945 | 1.39 s | 51.7 % |
| Office (INRS-EMT) | 3517 | 1.55 s | 70.4 % |
| Home | 6593 | 0.90 s | 44.0 % |
| Indoor campus (Concordia) | 6197 | 0.90 s | 53.7 % |

\* 10th-percentile noise floor and η = 6 dB, computed by the forecasting scripts.

## Reading the files

The scripts read the gap-handled grid `/grid/ISM_2400` of each `.h5` file. If that grid is missing, they fall back to `/sweeps` + `/psd/psd_db`. You can also point a scenario to a raw session folder (`sweeps_*.csv` + `psd_*.f16`) with `--data NAME=PATH`.

`h5py` is required.
