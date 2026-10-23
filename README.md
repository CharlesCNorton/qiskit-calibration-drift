# qiskit-calibration-drift

Automated collection of IBM Quantum hardware calibration data with environmental measurements.

## Dataset

[huggingface.co/datasets/phanerozoic/qiskit-calibration-drift](https://huggingface.co/datasets/phanerozoic/qiskit-calibration-drift)

The schema, one row per `(backend, property, qubit_a, qubit_b, calibrated_time)` with 31 columns, is documented on the dataset card.

## Data Sources

| Data | Source | Update Frequency |
|------|--------|------------------|
| Calibration | IBM Quantum Runtime API (`backend.properties()`) | Per IBM calibration cycle |
| Weather | NWS API, station KHPN | Hourly |
| Kp, ap, Ap, SN, F10.7 | GFZ Potsdam | 3-hourly / daily |
| Kp (fallback), 10.7 cm flux, Dst | NOAA SWPC | 3-hourly / daily / hourly |
| Bz (IMF, GSM) | NOAA SWPC real-time solar wind, operational L1 spacecraft | 1 minute |
| Neutron flux | NMDB, Newark station | Hourly |

## Collection

`poll.py` runs every 30 minutes via GitHub Actions. For each backend it reads the historical `BackendProperties` at every half hour since that backend's last poll in the dataset (up to 14 days back) and then the live snapshot, and appends the rows whose `(backend, property, qubit_a, qubit_b, calibrated_time)` is not yet in the dataset. Each covariate is the sample nearest to `calibrated_time` among those its source had published by `observed_time`, within a tolerance set per source in `COVARIATES`; empty covariate cells of rows observed in the last 35 days are filled on later runs once the source publishes them.

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
