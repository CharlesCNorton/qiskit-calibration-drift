"""
IBM Quantum Calibration Data Collector

Extracts calibration parameters from IBM Quantum backends and appends
new measurements to a HuggingFace dataset. Includes environmental data
(weather, space weather) for correlation analysis.
"""

import os
import sys
import json
import urllib.request
from datetime import datetime, timezone
from dateutil import parser as dateparser


REPO_ID = "phanerozoic/qiskit-calibration-drift"

# IBM Quantum Data Center locations
LOCATIONS = {
    "yorktown_heights_ny": {
        "lat": 41.27,
        "lon": -73.78,
        "backends": ["ibm_torino", "ibm_fez", "ibm_marrakesh"]
    }
}


def log(msg):
    """Print timestamped log message."""
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] {msg}")


def normalize_timestamp(ts_str):
    """Convert any timestamp string to UTC ISO format for consistent comparison."""
    try:
        dt = dateparser.parse(str(ts_str))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception as e:
        log(f"Warning: Could not parse timestamp '{ts_str}': {e}")
        return str(ts_str)


def get_space_weather():
    """Fetch current space weather data from NOAA SWPC."""
    log("Fetching space weather data...")
    data = {"kp_index": None, "solar_flux": None}

    # Kp index
    try:
        url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        with urllib.request.urlopen(url, timeout=15) as resp:
            kp_data = json.loads(resp.read())
            if len(kp_data) > 1:
                latest = kp_data[-1]
                data["kp_index"] = float(latest[1])
                log(f"  Kp index: {data['kp_index']}")
    except Exception as e:
        log(f"  Warning: Could not fetch Kp index: {e}")

    # Solar flux (10.7cm)
    try:
        url = "https://services.swpc.noaa.gov/products/summary/10cm-flux.json"
        with urllib.request.urlopen(url, timeout=15) as resp:
            flux_data = json.loads(resp.read())
            data["solar_flux"] = float(flux_data.get("Flux", 0))
            log(f"  Solar flux: {data['solar_flux']} SFU")
    except Exception as e:
        log(f"  Warning: Could not fetch solar flux: {e}")

    return data


def get_weather(lat, lon):
    """Fetch current weather from NWS API."""
    log(f"Fetching weather for ({lat}, {lon})...")
    data = {"temperature_c": None, "pressure_hpa": None, "humidity_pct": None}

    try:
        # Get grid point
        url = f"https://api.weather.gov/points/{lat},{lon}"
        req = urllib.request.Request(url, headers={"User-Agent": "qiskit-calibration-drift"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            point_data = json.loads(resp.read())
            stations_url = point_data["properties"]["observationStations"]

        # Get nearest station
        req = urllib.request.Request(stations_url, headers={"User-Agent": "qiskit-calibration-drift"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            stations_data = json.loads(resp.read())
            station_id = stations_data["features"][0]["properties"]["stationIdentifier"]

        # Get latest observation
        obs_url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
        req = urllib.request.Request(obs_url, headers={"User-Agent": "qiskit-calibration-drift"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            obs_data = json.loads(resp.read())
            props = obs_data["properties"]

            if props.get("temperature", {}).get("value") is not None:
                data["temperature_c"] = round(props["temperature"]["value"], 2)

            if props.get("barometricPressure", {}).get("value") is not None:
                data["pressure_hpa"] = round(props["barometricPressure"]["value"] / 100, 2)

            if props.get("relativeHumidity", {}).get("value") is not None:
                data["humidity_pct"] = round(props["relativeHumidity"]["value"], 2)

        log(f"  Temperature: {data['temperature_c']}°C")
        log(f"  Pressure: {data['pressure_hpa']} hPa")
        log(f"  Humidity: {data['humidity_pct']}%")

    except Exception as e:
        log(f"  Warning: Could not fetch weather: {e}")

    return data


def get_existing_keys():
    """Return set of (backend, qubit, property, calibrated_time) already in dataset."""
    log("Loading existing dataset from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset(REPO_ID, split="train")
        keys = set(
            (r["backend"], r["qubit"], r["property"], normalize_timestamp(r["calibrated_time"]))
            for r in ds
        )
        log(f"Loaded {len(keys)} existing records.")
        return keys
    except Exception as e:
        log(f"Warning: Could not load existing dataset: {e}")
        log("Assuming empty dataset.")
        return set()


def connect_ibm():
    """Connect to IBM Quantum service."""
    log("Connecting to IBM Quantum...")
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        token = os.environ.get("IBM_QUANTUM_TOKEN")
        if token:
            log("Using token from environment variable.")
            service = QiskitRuntimeService(
                channel="ibm_cloud",
                token=token,
                instance="crn:v1:bluemix:public:quantum-computing:us-east:a/a9114248c6c44fe88a40cda24e7073c3:a72852a9-5e25-429b-b8fc-8ac73fb30240::"
            )
        else:
            log("Using saved credentials.")
            service = QiskitRuntimeService()
        log("Connected successfully.")
        return service
    except Exception as e:
        log(f"Error: Failed to connect to IBM Quantum: {e}")
        raise


def get_location_for_backend(backend_name):
    """Return location info for a backend."""
    for loc_name, loc_info in LOCATIONS.items():
        if backend_name in loc_info["backends"]:
            return loc_name, loc_info["lat"], loc_info["lon"]
    return "unknown", None, None


def extract_calibration(service, env_data):
    """Extract current calibration data from all available backends."""
    log("Fetching backend list...")
    try:
        backends = service.backends()
        log(f"Found {len(backends)} backends: {[b.name for b in backends]}")
    except Exception as e:
        log(f"Error: Failed to fetch backends: {e}")
        raise

    observed_time = datetime.now(timezone.utc).isoformat()
    records = []
    errors = []

    for backend in backends:
        name = backend.name
        log(f"Processing {name}...")

        location, lat, lon = get_location_for_backend(name)

        try:
            props = backend.properties()
            config = backend.configuration()
        except Exception as e:
            log(f"  Error: Failed to get properties for {name}: {e}")
            errors.append((name, "properties", str(e)))
            continue

        # Get weather for this location if we haven't already
        weather_key = f"{lat},{lon}"
        if weather_key not in env_data["weather_cache"] and lat is not None:
            env_data["weather_cache"][weather_key] = get_weather(lat, lon)

        weather = env_data["weather_cache"].get(weather_key, {})

        qubit_count = 0
        qubit_errors = 0
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
                        "calibrated_time": normalize_timestamp(cal_time),
                        "observed_time": observed_time,
                        "location": location,
                        "latitude": lat,
                        "longitude": lon,
                        "temperature_c": weather.get("temperature_c"),
                        "pressure_hpa": weather.get("pressure_hpa"),
                        "humidity_pct": weather.get("humidity_pct"),
                        "kp_index": env_data["space"].get("kp_index"),
                        "solar_flux_sfu": env_data["space"].get("solar_flux"),
                    })
                qubit_count += 1
            except Exception as e:
                qubit_errors += 1
                if qubit_errors <= 3:
                    log(f"  Warning: Qubit {q} error: {e}")

        if qubit_errors > 3:
            log(f"  ... and {qubit_errors - 3} more qubit errors")

        log(f"  Qubits processed: {qubit_count}/{config.n_qubits}")

        edge_count = 0
        edge_errors = 0
        for edge in config.coupling_map:
            try:
                err = props.gate_error("cz", edge)
                records.append({
                    "backend": name,
                    "qubit": -1,
                    "property": f"cz_error_{edge[0]}_{edge[1]}",
                    "value": float(err) if err is not None else None,
                    "calibrated_time": normalize_timestamp(props.last_update_date),
                    "observed_time": observed_time,
                    "location": location,
                    "latitude": lat,
                    "longitude": lon,
                    "temperature_c": weather.get("temperature_c"),
                    "pressure_hpa": weather.get("pressure_hpa"),
                    "humidity_pct": weather.get("humidity_pct"),
                    "kp_index": env_data["space"].get("kp_index"),
                    "solar_flux_sfu": env_data["space"].get("solar_flux"),
                })
                edge_count += 1
            except Exception:
                edge_errors += 1

        log(f"  CZ edges processed: {edge_count}/{len(config.coupling_map)}")

    log(f"Extraction complete. Total records: {len(records)}")
    if errors:
        log(f"Backend errors: {errors}")

    return records


def upload_records(new_records):
    """Upload new records to HuggingFace dataset."""
    log(f"Preparing to upload {len(new_records)} new records...")

    try:
        from datasets import load_dataset, Dataset, concatenate_datasets

        new_ds = Dataset.from_list(new_records)
        log("Created new dataset from records.")

        try:
            log("Loading existing dataset for concatenation...")
            existing_ds = load_dataset(REPO_ID, split="train")
            log(f"Existing dataset has {len(existing_ds)} records.")
            combined = concatenate_datasets([existing_ds, new_ds])
            log(f"Combined dataset has {len(combined)} records.")
        except Exception as e:
            log(f"Warning: Could not load existing dataset: {e}")
            log("Uploading as new dataset.")
            combined = new_ds

        log("Pushing to HuggingFace Hub...")
        combined.push_to_hub(REPO_ID, private=False)
        log("Upload complete.")

    except Exception as e:
        log(f"Error: Upload failed: {e}")
        raise


def main():
    log("=" * 60)
    log("IBM Quantum Calibration Poller (with Environmental Data)")
    log(f"Target dataset: {REPO_ID}")
    log("=" * 60)

    # Collect environmental data first
    env_data = {
        "space": get_space_weather(),
        "weather_cache": {}
    }

    try:
        existing_keys = get_existing_keys()
    except Exception as e:
        log(f"Fatal: Could not get existing keys: {e}")
        sys.exit(1)

    try:
        service = connect_ibm()
    except Exception as e:
        log(f"Fatal: Could not connect to IBM: {e}")
        sys.exit(1)

    try:
        records = extract_calibration(service, env_data)
    except Exception as e:
        log(f"Fatal: Extraction failed: {e}")
        sys.exit(1)

    log("Filtering for new records...")
    new_records = [
        r for r in records
        if (r["backend"], r["qubit"], r["property"], r["calibrated_time"])
        not in existing_keys
    ]
    log(f"New records: {len(new_records)} / {len(records)} extracted")

    if not new_records:
        log("No new calibration data. Done.")
        return

    try:
        upload_records(new_records)
    except Exception as e:
        log(f"Fatal: Upload failed: {e}")
        sys.exit(1)

    log("=" * 60)
    log(f"Successfully uploaded {len(new_records)} new records.")
    log("=" * 60)


if __name__ == "__main__":
    main()
