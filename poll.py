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
import urllib.request
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


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] {msg}")


def parse_timestamp(ts_str):
    """Parse timestamp string to datetime object."""
    try:
        dt = dateparser.parse(str(ts_str))
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
    return str(ts_str)


def solar_zenith(lat, lon, dt):
    """Calculate solar zenith angle in degrees. >90 = night."""
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


def fetch_kp_history():
    """Fetch Kp index history (30 days available)."""
    log("  Fetching Kp history...")
    history = []
    try:
        url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
            for row in data[1:]:
                ts = parse_timestamp(row[0])
                if ts:
                    history.append((ts, float(row[1])))
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
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
            for row in data[1:]:
                ts = parse_timestamp(row[0])
                if ts:
                    history.append((ts, float(row[1])))
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
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
            for row in data[1:]:
                ts = parse_timestamp(row[0])
                if ts and row[3] is not None:
                    history.append((ts, float(row[3])))
        log(f"    Got {len(history)} Bz records")
    except Exception as e:
        log(f"    Warning: Bz fetch failed: {e}")
    return history


def fetch_solar_flux_history():
    """Fetch solar flux (current value only, updates daily)."""
    log("  Fetching solar flux...")
    try:
        url = "https://services.swpc.noaa.gov/products/summary/10cm-flux.json"
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
            flux = float(data.get("Flux", 0))
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
        req = urllib.request.Request(url, headers={"User-Agent": "qiskit-calibration-drift"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode("utf-8")
            pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2});\s*([\d.]+)"
            matches = re.findall(pattern, data)
            for ts_str, val in matches:
                ts = parse_timestamp(ts_str)
                if ts:
                    history.append((ts, float(val)))
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
        req = urllib.request.Request(url, headers={"User-Agent": "qiskit-calibration-drift"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                ts = parse_timestamp(props.get("timestamp"))
                if ts:
                    temp = props.get("temperature", {}).get("value")
                    pres = props.get("barometricPressure", {}).get("value")
                    hum = props.get("relativeHumidity", {}).get("value")
                    history.append((ts, {
                        "temperature_c": round(temp, 2) if temp is not None else None,
                        "pressure_hpa": round(pres / 100, 2) if pres is not None else None,
                        "humidity_pct": round(hum, 2) if hum is not None else None
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
    if weather_station not in env_history["weather"]:
        env_history["weather"][weather_station] = fetch_weather_history(weather_station)

    weather_match = find_closest(env_history["weather"][weather_station], cal_dt)
    weather = weather_match if weather_match else {}

    return {
        "solar_zenith_deg": solar_zenith(lat, lon, cal_dt) if cal_dt and lat else None,
        "temperature_c": weather.get("temperature_c"),
        "pressure_hpa": weather.get("pressure_hpa"),
        "humidity_pct": weather.get("humidity_pct"),
        "kp_index": find_closest(env_history["kp"], cal_dt),
        "solar_flux_sfu": env_history["solar_flux"],
        "dst_nt": find_closest(env_history["dst"], cal_dt),
        "bz_gsm_nt": find_closest(env_history["bz"], cal_dt),
        "neutron_flux": find_closest(env_history["neutron"], cal_dt),
    }


def get_existing_keys():
    """Return set of existing record keys."""
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

    observed_time = datetime.now(timezone.utc).isoformat()
    records = []
    errors = []

    for backend in backends:
        name = backend.name
        log(f"Processing {name}...")

        location, lat, lon, weather_station = get_location_for_backend(name)

        try:
            props = backend.properties()
            config = backend.configuration()
        except Exception as e:
            log(f"  Error: Failed to get properties for {name}: {e}")
            errors.append((name, "properties", str(e)))
            continue

        qubit_count = 0
        qubit_errors = 0
        for q in range(config.n_qubits):
            try:
                qprops = props.qubit_property(q)
                for prop_name, (value, cal_time) in qprops.items():
                    if prop_name == "readout_length":
                        continue

                    cal_dt = parse_timestamp(cal_time)
                    env = get_env_for_time(env_history, cal_dt, lat, lon, weather_station)

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
                        **env
                    })
                qubit_count += 1
            except Exception as e:
                qubit_errors += 1
                if qubit_errors <= 3:
                    log(f"  Warning: Qubit {q} error: {e}")

        if qubit_errors > 3:
            log(f"  ... and {qubit_errors - 3} more qubit errors")

        log(f"  Qubits processed: {qubit_count}/{config.n_qubits}")

        sx_count = 0
        for q in range(config.n_qubits):
            try:
                sx_err = props.gate_error("sx", q)
                cal_dt = parse_timestamp(props.last_update_date)
                env = get_env_for_time(env_history, cal_dt, lat, lon, weather_station)

                records.append({
                    "backend": name,
                    "qubit": q,
                    "property": "sx_error",
                    "value": float(sx_err) if sx_err is not None else None,
                    "calibrated_time": normalize_timestamp(props.last_update_date),
                    "observed_time": observed_time,
                    "location": location,
                    "latitude": lat,
                    "longitude": lon,
                    **env
                })
                sx_count += 1
            except Exception:
                pass

        log(f"  SX gates processed: {sx_count}/{config.n_qubits}")

        edge_count = 0
        edge_errors = 0
        for edge in config.coupling_map:
            try:
                err = props.gate_error("cz", edge)
                cal_dt = parse_timestamp(props.last_update_date)
                env = get_env_for_time(env_history, cal_dt, lat, lon, weather_station)

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
                    **env
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
    log("IBM Quantum Calibration Poller (with Historical Env Data)")
    log(f"Target dataset: {REPO_ID}")
    log("=" * 60)

    env_history = fetch_environmental_history()

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
        records = extract_calibration(service, env_history)
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
