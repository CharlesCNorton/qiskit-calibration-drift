"""
IBM Quantum Calibration Data Collector

Extracts calibration parameters from IBM Quantum backends and appends
new measurements to a HuggingFace dataset. Includes environmental data
(weather, space weather) matched to calibration timestamps.
"""

import os
import sys
import json
import math
import re
import time
import urllib.request
import traceback
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser


REPO_ID = "phanerozoic/qiskit-calibration-drift"

LOCATIONS = {
    "yorktown_heights_ny": {
        "lat": 41.27,
        "lon": -73.78,
        "backends": ["ibm_torino", "ibm_fez", "ibm_marrakesh"],
        "weather_station": "KHPN"
    }
}

FLOAT_COLUMNS = [
    "value", "latitude", "longitude", "solar_zenith_deg",
    "temperature_c", "pressure_hpa", "humidity_pct",
    "kp_index", "solar_flux_sfu", "dst_nt", "bz_gsm_nt", "neutron_flux"
]

EXPECTED_QUBIT_PROPERTIES = ["T1", "T2", "readout_error", "prob_meas0_prep1", "prob_meas1_prep0"]

MAX_RETRIES = 3
RETRY_DELAY = 5


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] {msg}")


def parse_timestamp(ts_str):
    """Parse timestamp string to datetime object."""
    if ts_str is None:
        return None
    try:
        dt = dateparser.parse(str(ts_str))
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def normalize_timestamp(ts_str):
    """Convert timestamp to standard string format."""
    dt = parse_timestamp(ts_str)
    if dt:
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return str(ts_str) if ts_str else ""


def solar_zenith(lat, lon, dt):
    """Calculate solar zenith angle in degrees. >90 = night."""
    if lat is None or lon is None or dt is None:
        return None
    try:
        doy = dt.timetuple().tm_yday
        hour = dt.hour + dt.minute / 60 + dt.second / 3600
        decl = -23.45 * math.cos(math.radians(360 / 365 * (doy + 10)))
        solar_noon = 12 - lon / 15
        hour_angle = 15 * (hour - solar_noon)
        lat_rad = math.radians(lat)
        decl_rad = math.radians(decl)
        hour_rad = math.radians(hour_angle)
        cos_zenith = (math.sin(lat_rad) * math.sin(decl_rad) +
                      math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_rad))
        zenith = math.degrees(math.acos(max(-1, min(1, cos_zenith))))
        return round(zenith, 2)
    except Exception:
        return None


def find_closest(history, target_dt, max_hours=48):
    """Find closest historical value to target datetime."""
    if not history or not target_dt:
        return None

    best_match = None
    best_diff = timedelta(hours=max_hours)

    for ts, value in history:
        diff = abs(ts - target_dt)
        if diff < best_diff:
            best_diff = diff
            best_match = value

    return best_match


def fetch_with_retry(url, headers=None, timeout=30, description="data"):
    """Fetch URL with retry logic."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                log(f"    Retry {attempt + 1}/{MAX_RETRIES} for {description}: {e}")
                time.sleep(RETRY_DELAY)
    log(f"    Failed to fetch {description} after {MAX_RETRIES} attempts: {last_error}")
    return None


def fetch_kp_history():
    """Fetch Kp index history (30 days available)."""
    log("  Fetching Kp history...")
    history = []
    try:
        url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        data_bytes = fetch_with_retry(url, description="Kp index")
        if data_bytes:
            data = json.loads(data_bytes)
            for row in data[1:]:
                ts = parse_timestamp(row[0])
                if ts and len(row) > 1 and row[1] is not None:
                    try:
                        history.append((ts, float(row[1])))
                    except (ValueError, TypeError):
                        pass
            log(f"    Got {len(history)} Kp records")
    except Exception as e:
        log(f"    Warning: Kp fetch failed: {e}")
    return history


def fetch_dst_history():
    """Fetch Dst index history."""
    log("  Fetching Dst history...")
    history = []
    try:
        url = "https://services.swpc.noaa.gov/products/kyoto-dst.json"
        data_bytes = fetch_with_retry(url, description="Dst index")
        if data_bytes:
            data = json.loads(data_bytes)
            for row in data[1:]:
                ts = parse_timestamp(row[0])
                if ts and len(row) > 1 and row[1] is not None:
                    try:
                        history.append((ts, float(row[1])))
                    except (ValueError, TypeError):
                        pass
            log(f"    Got {len(history)} Dst records")
    except Exception as e:
        log(f"    Warning: Dst fetch failed: {e}")
    return history


def fetch_bz_history():
    """Fetch Bz IMF history (7 days)."""
    log("  Fetching Bz history...")
    history = []
    try:
        url = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"
        data_bytes = fetch_with_retry(url, description="Bz IMF")
        if data_bytes:
            data = json.loads(data_bytes)
            for row in data[1:]:
                ts = parse_timestamp(row[0])
                if ts and len(row) > 3 and row[3] is not None:
                    try:
                        history.append((ts, float(row[3])))
                    except (ValueError, TypeError):
                        pass
            log(f"    Got {len(history)} Bz records")
    except Exception as e:
        log(f"    Warning: Bz fetch failed: {e}")
    return history


def fetch_solar_flux_history():
    """Fetch solar flux (current value only, updates daily)."""
    log("  Fetching solar flux...")
    try:
        url = "https://services.swpc.noaa.gov/products/summary/10cm-flux.json"
        data_bytes = fetch_with_retry(url, timeout=15, description="solar flux")
        if data_bytes:
            data = json.loads(data_bytes)
            flux_val = data.get("Flux")
            if flux_val is not None:
                flux = float(flux_val)
                log(f"    Solar flux: {flux} SFU")
                return flux
    except Exception as e:
        log(f"    Warning: Solar flux fetch failed: {e}")
    return None


def fetch_neutron_history():
    """Fetch neutron monitor history (last 48 hours)."""
    log("  Fetching neutron history...")
    history = []
    try:
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=48)
        url = (f"https://www.nmdb.eu/nest/draw_graph.php?formchk=1&stations[]=NEWK"
               f"&tabchoice=revori&dtype=corr_for_pressure&tresolution=60"
               f"&date_choice=bydate"
               f"&start_year={start.year}&start_month={start.month:02d}"
               f"&start_day={start.day:02d}&start_hour=0&start_min=0"
               f"&end_year={now.year}&end_month={now.month:02d}"
               f"&end_day={now.day:02d}&end_hour=23&end_min=59"
               f"&output=ascii")
        headers = {"User-Agent": "qiskit-calibration-drift"}
        data_bytes = fetch_with_retry(url, headers=headers, description="neutron flux")
        if data_bytes:
            data = data_bytes.decode("utf-8")
            pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2});\s*([\d.]+)"
            matches = re.findall(pattern, data)
            for ts_str, val in matches:
                ts = parse_timestamp(ts_str)
                if ts:
                    try:
                        history.append((ts, float(val)))
                    except (ValueError, TypeError):
                        pass
            log(f"    Got {len(history)} neutron records")
    except Exception as e:
        log(f"    Warning: Neutron fetch failed: {e}")
    return history


def fetch_weather_history(station):
    """Fetch weather observation history."""
    log(f"  Fetching weather history ({station})...")
    history = []
    try:
        url = f"https://api.weather.gov/stations/{station}/observations"
        headers = {"User-Agent": "qiskit-calibration-drift"}
        data_bytes = fetch_with_retry(url, headers=headers, description=f"weather ({station})")
        if data_bytes:
            data = json.loads(data_bytes)
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                ts = parse_timestamp(props.get("timestamp"))
                if ts:
                    temp = props.get("temperature", {}).get("value")
                    pres = props.get("barometricPressure", {}).get("value")
                    hum = props.get("relativeHumidity", {}).get("value")
                    history.append((ts, {
                        "temperature_c": round(float(temp), 2) if temp is not None else None,
                        "pressure_hpa": round(float(pres) / 100, 2) if pres is not None else None,
                        "humidity_pct": round(float(hum), 2) if hum is not None else None
                    }))
            log(f"    Got {len(history)} weather records")
    except Exception as e:
        log(f"    Warning: Weather fetch failed: {e}")
    return history


def fetch_environmental_history():
    """Fetch all environmental data history."""
    log("Fetching environmental history (48h)...")
    return {
        "kp": fetch_kp_history(),
        "dst": fetch_dst_history(),
        "bz": fetch_bz_history(),
        "solar_flux": fetch_solar_flux_history(),
        "neutron": fetch_neutron_history(),
        "weather": {}
    }


def get_env_for_time(env_history, cal_dt, lat, lon, weather_station):
    """Get environmental data for a specific calibration time."""
    if weather_station and weather_station not in env_history["weather"]:
        env_history["weather"][weather_station] = fetch_weather_history(weather_station)

    weather_match = None
    if weather_station and weather_station in env_history["weather"]:
        weather_match = find_closest(env_history["weather"][weather_station], cal_dt)
    weather = weather_match if weather_match else {}

    return {
        "solar_zenith_deg": solar_zenith(lat, lon, cal_dt),
        "temperature_c": weather.get("temperature_c"),
        "pressure_hpa": weather.get("pressure_hpa"),
        "humidity_pct": weather.get("humidity_pct"),
        "kp_index": find_closest(env_history["kp"], cal_dt),
        "solar_flux_sfu": env_history["solar_flux"],
        "dst_nt": find_closest(env_history["dst"], cal_dt),
        "bz_gsm_nt": find_closest(env_history["bz"], cal_dt),
        "neutron_flux": find_closest(env_history["neutron"], cal_dt),
    }


def normalize_record_schema(record):
    """Ensure all float columns have consistent float64 type (or None)."""
    for col in FLOAT_COLUMNS:
        val = record.get(col)
        if val is not None:
            try:
                record[col] = float(val)
            except (ValueError, TypeError):
                record[col] = None
    return record


def get_existing_keys():
    """Return set of existing record keys and the existing dataset."""
    log("Loading existing dataset from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset(REPO_ID, split="train")
        keys = set(
            (r["backend"], r["qubit"], r["property"], normalize_timestamp(r["calibrated_time"]))
            for r in ds
        )
        log(f"Loaded {len(keys)} existing records.")
        return keys, ds
    except Exception as e:
        log(f"Warning: Could not load existing dataset: {e}")
        log("Starting with empty dataset.")
        return set(), None


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
            return loc_name, loc_info["lat"], loc_info["lon"], loc_info["weather_station"]
    return "unknown", None, None, None


def extract_calibration(service, env_history):
    """Extract calibration data with matched environmental data."""
    log("Fetching backend list...")
    try:
        backends = service.backends()
        log(f"Found {len(backends)} backends: {[b.name for b in backends]}")
    except Exception as e:
        log(f"Error: Failed to fetch backends: {e}")
        raise

    if not backends:
        raise RuntimeError("No backends available")

    observed_time = datetime.now(timezone.utc).isoformat()
    records = []
    errors = []
    property_counts = {}

    for backend in backends:
        name = backend.name
        log(f"Processing {name}...")
        property_counts[name] = {p: 0 for p in EXPECTED_QUBIT_PROPERTIES}
        property_counts[name]["sx_error"] = 0
        property_counts[name]["cz_error"] = 0

        location, lat, lon, weather_station = get_location_for_backend(name)

        try:
            props = backend.properties()
            config = backend.configuration()
        except Exception as e:
            log(f"  Error: Failed to get properties for {name}: {e}")
            errors.append((name, "properties", str(e)))
            continue

        if props is None:
            log(f"  Warning: No properties available for {name}")
            errors.append((name, "properties", "None returned"))
            continue

        qubit_count = 0
        qubit_errors = 0
        for q in range(config.n_qubits):
            try:
                qprops = props.qubit_property(q)
                if qprops is None:
                    qubit_errors += 1
                    continue

                for prop_name, prop_data in qprops.items():
                    if prop_name == "readout_length":
                        continue

                    if prop_data is None or len(prop_data) < 2:
                        continue

                    value, cal_time = prop_data[0], prop_data[1]
                    cal_dt = parse_timestamp(cal_time)
                    env = get_env_for_time(env_history, cal_dt, lat, lon, weather_station)

                    record = {
                        "backend": name,
                        "qubit": q,
                        "property": prop_name,
                        "value": float(value) if value is not None else None,
                        "calibrated_time": normalize_timestamp(cal_time),
                        "observed_time": observed_time,
                        "location": location,
                        "latitude": float(lat) if lat is not None else None,
                        "longitude": float(lon) if lon is not None else None,
                        **env
                    }
                    records.append(normalize_record_schema(record))

                    if prop_name in property_counts[name]:
                        property_counts[name][prop_name] += 1

                qubit_count += 1
            except Exception as e:
                qubit_errors += 1
                if qubit_errors <= 3:
                    log(f"  Warning: Qubit {q} error: {e}")

        if qubit_errors > 3:
            log(f"  ... and {qubit_errors - 3} more qubit errors")

        log(f"  Qubits processed: {qubit_count}/{config.n_qubits}")

        for prop_name, count in property_counts[name].items():
            if prop_name in EXPECTED_QUBIT_PROPERTIES:
                expected = config.n_qubits
                if count < expected:
                    log(f"  Warning: {prop_name} only got {count}/{expected} qubits")

        sx_count = 0
        sx_errors = 0
        for q in range(config.n_qubits):
            try:
                sx_err = props.gate_error("sx", q)
                if sx_err is None:
                    continue
                cal_dt = parse_timestamp(props.last_update_date)
                env = get_env_for_time(env_history, cal_dt, lat, lon, weather_station)

                record = {
                    "backend": name,
                    "qubit": q,
                    "property": "sx_error",
                    "value": float(sx_err),
                    "calibrated_time": normalize_timestamp(props.last_update_date),
                    "observed_time": observed_time,
                    "location": location,
                    "latitude": float(lat) if lat is not None else None,
                    "longitude": float(lon) if lon is not None else None,
                    **env
                }
                records.append(normalize_record_schema(record))
                sx_count += 1
            except Exception as e:
                sx_errors += 1
                if sx_errors <= 3:
                    log(f"  Warning: SX gate error for qubit {q}: {e}")

        property_counts[name]["sx_error"] = sx_count
        log(f"  SX gates processed: {sx_count}/{config.n_qubits}")

        edge_count = 0
        edge_errors = 0
        for edge in config.coupling_map:
            try:
                err = props.gate_error("cz", edge)
                if err is None:
                    continue
                cal_dt = parse_timestamp(props.last_update_date)
                env = get_env_for_time(env_history, cal_dt, lat, lon, weather_station)

                record = {
                    "backend": name,
                    "qubit": -1,
                    "property": f"cz_error_{edge[0]}_{edge[1]}",
                    "value": float(err),
                    "calibrated_time": normalize_timestamp(props.last_update_date),
                    "observed_time": observed_time,
                    "location": location,
                    "latitude": float(lat) if lat is not None else None,
                    "longitude": float(lon) if lon is not None else None,
                    **env
                }
                records.append(normalize_record_schema(record))
                edge_count += 1
            except Exception as e:
                edge_errors += 1

        property_counts[name]["cz_error"] = edge_count
        log(f"  CZ edges processed: {edge_count}/{len(config.coupling_map)}")

    log(f"Extraction complete. Total records: {len(records)}")

    log("Property extraction summary:")
    for backend_name, counts in property_counts.items():
        log(f"  {backend_name}:")
        for prop, count in counts.items():
            log(f"    {prop}: {count}")

    if errors:
        log(f"Backend errors: {errors}")

    if not records:
        raise RuntimeError("No records extracted from any backend")

    return records


def cast_dataset_to_schema(ds):
    """Cast dataset columns to consistent schema to avoid type mismatches."""
    from datasets import Features, Value

    features = Features({
        "backend": Value("string"),
        "qubit": Value("int64"),
        "property": Value("string"),
        "value": Value("float64"),
        "calibrated_time": Value("string"),
        "observed_time": Value("string"),
        "location": Value("string"),
        "latitude": Value("float64"),
        "longitude": Value("float64"),
        "solar_zenith_deg": Value("float64"),
        "temperature_c": Value("float64"),
        "pressure_hpa": Value("float64"),
        "humidity_pct": Value("float64"),
        "kp_index": Value("float64"),
        "solar_flux_sfu": Value("float64"),
        "dst_nt": Value("float64"),
        "bz_gsm_nt": Value("float64"),
        "neutron_flux": Value("float64"),
    })

    return ds.cast(features)


def upload_records(new_records, existing_ds):
    """Upload new records to HuggingFace dataset."""
    log(f"Preparing to upload {len(new_records)} new records...")

    from datasets import Dataset, concatenate_datasets, Features, Value

    features = Features({
        "backend": Value("string"),
        "qubit": Value("int64"),
        "property": Value("string"),
        "value": Value("float64"),
        "calibrated_time": Value("string"),
        "observed_time": Value("string"),
        "location": Value("string"),
        "latitude": Value("float64"),
        "longitude": Value("float64"),
        "solar_zenith_deg": Value("float64"),
        "temperature_c": Value("float64"),
        "pressure_hpa": Value("float64"),
        "humidity_pct": Value("float64"),
        "kp_index": Value("float64"),
        "solar_flux_sfu": Value("float64"),
        "dst_nt": Value("float64"),
        "bz_gsm_nt": Value("float64"),
        "neutron_flux": Value("float64"),
    })

    for record in new_records:
        normalize_record_schema(record)

    new_ds = Dataset.from_list(new_records, features=features)
    log(f"Created new dataset with {len(new_ds)} records.")

    if existing_ds is not None and len(existing_ds) > 0:
        log(f"Existing dataset has {len(existing_ds)} records.")
        try:
            existing_casted = cast_dataset_to_schema(existing_ds)
            log("Cast existing dataset to consistent schema.")
        except Exception as e:
            log(f"Error: Failed to cast existing dataset: {e}")
            log("ABORTING to prevent data loss. Please fix schema manually.")
            raise RuntimeError(f"Schema cast failed: {e}")

        try:
            combined = concatenate_datasets([existing_casted, new_ds])
            log(f"Combined dataset has {len(combined)} records.")
        except Exception as e:
            log(f"Error: Concatenation failed: {e}")
            log("ABORTING to prevent data loss. Do NOT overwrite existing data.")
            raise RuntimeError(f"Concatenation failed: {e}")
    else:
        log("No existing dataset. Creating new.")
        combined = new_ds

    log("Pushing to HuggingFace Hub...")
    for attempt in range(MAX_RETRIES):
        try:
            combined.push_to_hub(REPO_ID, private=False)
            log("Upload complete.")
            return
        except Exception as e:
            log(f"Upload attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * 2)
            else:
                raise RuntimeError(f"Upload failed after {MAX_RETRIES} attempts: {e}")


def validate_records(records):
    """Validate extracted records for data quality."""
    log("Validating extracted records...")

    issues = []

    if not records:
        issues.append("No records extracted")
        return issues

    backends = set(r["backend"] for r in records)
    log(f"  Backends in records: {backends}")

    for backend in backends:
        backend_records = [r for r in records if r["backend"] == backend]

        t1_count = len([r for r in backend_records if r["property"] == "T1"])
        t2_count = len([r for r in backend_records if r["property"] == "T2"])
        readout_count = len([r for r in backend_records if r["property"] == "readout_error"])
        p01_count = len([r for r in backend_records if r["property"] == "prob_meas0_prep1"])
        p10_count = len([r for r in backend_records if r["property"] == "prob_meas1_prep0"])
        sx_count = len([r for r in backend_records if r["property"] == "sx_error"])

        log(f"  {backend}: T1={t1_count}, T2={t2_count}, readout_error={readout_count}, "
            f"prob_meas0_prep1={p01_count}, prob_meas1_prep0={p10_count}, sx_error={sx_count}")

        if t1_count == 0:
            issues.append(f"{backend}: No T1 data extracted")
        if t2_count == 0:
            issues.append(f"{backend}: No T2 data extracted")
        if readout_count == 0:
            issues.append(f"{backend}: No readout_error data extracted")

    null_counts = {col: 0 for col in FLOAT_COLUMNS}
    for r in records:
        for col in FLOAT_COLUMNS:
            if r.get(col) is None:
                null_counts[col] += 1

    total = len(records)
    for col, count in null_counts.items():
        pct = 100 * count / total
        if pct > 50:
            issues.append(f"Column {col} is {pct:.1f}% null")

    if issues:
        log("Validation issues found:")
        for issue in issues:
            log(f"  - {issue}")
    else:
        log("  Validation passed.")

    return issues


def main():
    log("=" * 60)
    log("IBM Quantum Calibration Poller (with Historical Env Data)")
    log(f"Target dataset: {REPO_ID}")
    log("=" * 60)

    exit_code = 0

    try:
        env_history = fetch_environmental_history()
    except Exception as e:
        log(f"Warning: Environmental fetch had errors: {e}")
        env_history = {"kp": [], "dst": [], "bz": [], "solar_flux": None, "neutron": [], "weather": {}}

    try:
        existing_keys, existing_ds = get_existing_keys()
    except Exception as e:
        log(f"Warning: Could not get existing keys: {e}")
        log(traceback.format_exc())
        existing_keys = set()
        existing_ds = None

    try:
        service = connect_ibm()
    except Exception as e:
        log(f"Fatal: Could not connect to IBM: {e}")
        log(traceback.format_exc())
        sys.exit(1)

    try:
        records = extract_calibration(service, env_history)
    except Exception as e:
        log(f"Fatal: Extraction failed: {e}")
        log(traceback.format_exc())
        sys.exit(1)

    validation_issues = validate_records(records)

    log("Filtering for new records...")
    new_records = [
        r for r in records
        if (r["backend"], r["qubit"], r["property"], r["calibrated_time"])
        not in existing_keys
    ]
    log(f"New records: {len(new_records)} / {len(records)} extracted")

    if not new_records:
        log("No new calibration data. Done.")
        if validation_issues:
            log("WARNING: Validation issues were found (see above).")
            exit_code = 0
        sys.exit(exit_code)

    try:
        upload_records(new_records, existing_ds)
    except Exception as e:
        log(f"Fatal: Upload failed: {e}")
        log(traceback.format_exc())
        sys.exit(1)

    log("=" * 60)
    log(f"Successfully uploaded {len(new_records)} new records.")
    if validation_issues:
        log(f"WARNING: {len(validation_issues)} validation issues found.")
    log("=" * 60)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
