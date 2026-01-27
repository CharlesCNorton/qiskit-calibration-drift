# qiskit-calibration-drift

Automated collection of IBM Quantum hardware calibration data.

## Dataset

[phanerozoic/qiskit-calibration-drift](https://huggingface.co/datasets/phanerozoic/qiskit-calibration-drift)

## Schema

| Field | Type | Description |
|-------|------|-------------|
| backend | string | Backend name (ibm_torino, ibm_fez, ibm_marrakesh) |
| qubit | int | Qubit index (-1 for two-qubit gate data) |
| property | string | T1, T2, readout_error, prob_meas0_prep1, prob_meas1_prep0, or cz_error_i_j |
| value | float | Measured value |
| calibrated_time | string | IBM calibration timestamp |
| observed_time | string | Collection timestamp |

## Collection

Runs every 30 minutes via GitHub Actions. Only new calibration data (determined by `calibrated_time`) is appended.

## License

CC-BY-4.0
