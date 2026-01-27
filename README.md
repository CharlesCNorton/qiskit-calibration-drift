# qiskit-calibration-drift

Automated collection of IBM Quantum hardware calibration data with environmental measurements.

## Dataset

[huggingface.co/datasets/phanerozoic/qiskit-calibration-drift](https://huggingface.co/datasets/phanerozoic/qiskit-calibration-drift)

## Schema

| Field | Type | Description |
|-------|------|-------------|
| `backend` | string | Backend name (ibm_torino, ibm_fez, ibm_marrakesh) |
| `qubit` | int | Qubit index (-1 for two-qubit gate data) |
| `property` | string | T1, T2, readout_error, prob_meas0_prep1, prob_meas1_prep0, cz_error_i_j |
| `value` | float | Measured value |
| `calibrated_time` | string | IBM calibration timestamp (UTC) |
| `observed_time` | string | Collection timestamp (UTC) |
| `location` | string | Data center location |
| `latitude` | float | Data center latitude |
| `longitude` | float | Data center longitude |
| `temperature_c` | float | Local temperature (°C) |
| `pressure_hpa` | float | Barometric pressure (hPa) |
| `humidity_pct` | float | Relative humidity (%) |
| `kp_index` | float | Planetary K-index (0-9) |
| `solar_flux_sfu` | float | 10.7cm solar radio flux (SFU) |
| `dst_nt` | float | Dst index (nT) |
| `bz_gsm_nt` | float | IMF Bz component (nT) |
| `neutron_flux` | float | Cosmic ray flux (Newark, DE monitor) |

## Data Sources

| Data | Source | Update Frequency |
|------|--------|------------------|
| Calibration | IBM Quantum Runtime API | Per IBM calibration cycle |
| Weather | NWS API (NOAA) | Hourly |
| Kp, Solar flux | SWPC (NOAA) | 3-hourly / daily |
| Dst index | Kyoto/NOAA | Hourly |
| Bz (IMF) | SWPC (NOAA) | Real-time |
| Neutron flux | NMDB (Newark, DE) | Hourly |

## Collection

Runs every 30 minutes via GitHub Actions. Only new calibration data (by `calibrated_time`) is appended.

## Setup

1. Fork this repository
2. Add secrets in Settings → Secrets → Actions:
   - `IBM_QUANTUM_TOKEN`: Your IBM Quantum API token
   - `HF_TOKEN`: Your HuggingFace token
3. Enable Actions

## Acknowledgments

We acknowledge the NMDB database (www.nmdb.eu), founded under the European Union's FP7 programme (contract no. 213007) for providing neutron monitor data.

## License

CC-BY-4.0
