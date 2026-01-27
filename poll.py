"""
IBM Quantum Calibration Data Collector

Extracts calibration parameters from IBM Quantum backends and appends
new measurements to a HuggingFace dataset. Designed for scheduled execution.
"""

import os
from datetime import datetime, timezone
from qiskit_ibm_runtime import QiskitRuntimeService
from datasets import load_dataset, Dataset, concatenate_datasets


REPO_ID = "phanerozoic/qiskit-calibration-drift"


def get_existing_keys():
    """Return set of (backend, qubit, property, calibrated_time) already in dataset."""
    try:
        ds = load_dataset(REPO_ID, split="train")
        return set(
            (r["backend"], r["qubit"], r["property"], r["calibrated_time"])
            for r in ds
        )
    except Exception:
        return set()


def extract_calibration():
    """Extract current calibration data from all available backends."""
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if token:
        service = QiskitRuntimeService(channel="ibm_cloud", token=token,
            instance="crn:v1:bluemix:public:quantum-computing:us-east:a/a9114248c6c44fe88a40cda24e7073c3:a72852a9-5e25-429b-b8fc-8ac73fb30240::")
    else:
        service = QiskitRuntimeService()

    backends = service.backends()
    observed_time = datetime.now(timezone.utc).isoformat()
    records = []

    for backend in backends:
        name = backend.name
        props = backend.properties()
        config = backend.configuration()

        for q in range(config.n_qubits):
            try:
                qprops = props.qubit_property(q)
                for prop_name, (value, cal_time) in qprops.items():
                    if prop_name == "readout_length":
                        continue
                    records.append({
                        "backend": name,
                        "qubit": q,
                        "property": prop_name,
                        "value": float(value) if value is not None else None,
                        "calibrated_time": str(cal_time),
                        "observed_time": observed_time,
                    })
            except Exception:
                pass

        for edge in config.coupling_map:
            try:
                err = props.gate_error("cz", edge)
                records.append({
                    "backend": name,
                    "qubit": -1,
                    "property": f"cz_error_{edge[0]}_{edge[1]}",
                    "value": float(err) if err is not None else None,
                    "calibrated_time": str(props.last_update_date),
                    "observed_time": observed_time,
                })
            except Exception:
                pass

    return records


def main():
    existing_keys = get_existing_keys()
    print(f"Existing records: {len(existing_keys)}")

    records = extract_calibration()
    print(f"Extracted: {len(records)}")

    new_records = [
        r for r in records
        if (r["backend"], r["qubit"], r["property"], r["calibrated_time"])
        not in existing_keys
    ]
    print(f"New records: {len(new_records)}")

    if not new_records:
        print("No new calibration data.")
        return

    new_ds = Dataset.from_list(new_records)

    try:
        existing_ds = load_dataset(REPO_ID, split="train")
        combined = concatenate_datasets([existing_ds, new_ds])
    except Exception:
        combined = new_ds

    combined.push_to_hub(REPO_ID, private=False)
    print(f"Uploaded {len(new_records)} new records.")


if __name__ == "__main__":
    main()
