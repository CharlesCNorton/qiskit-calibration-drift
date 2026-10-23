"""
IBM Quantum Calibration Drift Poller

Pulls the full BackendProperties from every available IBM Heron backend,
parses every qubit-level and gate-level property, joins environmental and
space-weather covariates, and appends new calibration events to the
HuggingFace dataset.

Schema (one row per (backend, property, qubit_a, qubit_b, calibrated_time)
observation):
  - backend, property, property_family, qubit_a, qubit_b, value, unit, scope
  - is_failure_ceiling, is_new_measurement
  - observed_time, calibrated_time, snapshot_update_time, calibration_age_seconds
  - chipwide_recal_event_id
  - latitude, longitude
  - solar_zenith_deg, temperature_c, pressure_hpa, humidity_pct
  - bz_gsm_nt, neutron_flux, kp_index, ap_index, Ap_daily, SN
  - f107_observed_sfu, f107_adjusted_sfu, solar_flux_sfu, dst_nt

Behaviour:
  - Loads existing dataset; if the schema is the legacy 17-column version,
    treats it as empty and starts the new schema fresh on the next push.
    Existing legacy rows are not auto-migrated by the poller itself; that
    is done as a separate one-shot job.
  - Pulls each backend's `properties().to_dict()` and walks the qubits[] /
    gates[] structures to emit every parameter. Before the live snapshot it
    pulls the historical snapshot at every half hour since the backend's last
    poll in the dataset (at most GAP_DAYS back), so a late or failed run loses
    no calibration.
  - Joins environmental data (NOAA SWPC, NMDB, weather.gov, GFZ Potsdam): each
    covariate is the sample nearest to calibrated_time among those the source
    had published by observed_time, within the tolerance in COVARIATES.
  - Fills empty covariate cells of rows observed in the last REFILL_DAYS once
    the source has published them.
  - Computes is_new_measurement against the latest earlier value for the same
    (backend, property, qubit_a, qubit_b) key.
  - is_failure_ceiling=True iff property ends with `_error` or starts with
    `prob_meas` and value >= 0.999.
  - Detects chipwide_recal events: if >=20% of (qubit, property) units on the
    same backend share a calibrated_time within the same poll, all rows from
    that group get a shared event_id.
"""
import os
import sys
import json
import math
import re
import time
import urllib.request
import urllib.parse
import io
import traceback
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser

REPO_ID = "phanerozoic/qiskit-calibration-drift"

LOCATIONS = {
    "yorktown_heights_ny": {
        "lat": 41.27,
        "lon": -73.78,
        "backends": ["ibm_torino", "ibm_fez", "ibm_marrakesh", "ibm_kingston"],
        "weather_station": "KHPN",
    }
}

MAX_RETRIES = 3
RETRY_DELAY = 5
GAP_DAYS = 14
REFILL_DAYS = 35

# Property -> family mapping (epoch4 canonical)
FAMILY_MAP = {
    "T1": "T1", "T2": "T2",
    "readout_error": "readout_error", "readout_length": "readout_length",
    "prob_meas0_prep1": "prob_meas0_prep1", "prob_meas1_prep0": "prob_meas1_prep0",
    "reset_gate_length": "reset_length",
    "sx_gate_error": "sx_error", "sx_gate_length": "sx_length",
    "x_gate_error": "x_error", "x_gate_length": "x_length",
    "id_gate_error": "id_error", "id_gate_length": "id_length",
    "rx_gate_error": "rx_error", "rx_gate_length": "rx_length",
    "xslow_gate_error": "xslow_error", "xslow_gate_length": "xslow_length",
    "cz_gate_error": "cz_error", "cz_gate_length": "cz_length",
    "rzz_gate_error": "rzz_error", "rzz_gate_length": "rzz_length",
    "measure_2_gate_error": "measure_2_error", "measure_2_gate_length": "measure_2_length",
}


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] {msg}", flush=True)


def parse_timestamp(ts):
    if ts is None or ts == "":
        return None
    try:
        if isinstance(ts, datetime):
            dt = ts
        else:
            dt = dateparser.parse(str(ts))
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def solar_zenith(lat, lon, dt):
    if lat is None or lon is None or dt is None:
        return None
    try:
        doy = dt.timetuple().tm_yday
        hour = dt.hour + dt.minute / 60 + dt.second / 3600
        decl = -23.45 * math.cos(math.radians(360 / 365 * (doy + 10)))
        solar_noon = 12 - lon / 15
        hour_angle = 15 * (hour - solar_noon)
        lat_rad = math.radians(lat); decl_rad = math.radians(decl); hour_rad = math.radians(hour_angle)
        cos_zenith = (math.sin(lat_rad) * math.sin(decl_rad) +
                      math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_rad))
        return round(math.degrees(math.acos(max(-1, min(1, cos_zenith)))), 2)
    except Exception:
        return None


def fetch_with_retry(url, headers=None, timeout=30, description="data"):
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            last = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    log(f"    {description} fetch failed after {MAX_RETRIES} tries: {last}")
    return None


# ---------- Environmental data sources ----------

def fetch_kp_noaa():
    log("  Fetching NOAA Kp...")
    url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
    data_bytes = fetch_with_retry(url, description="Kp")
    history = []
    if data_bytes:
        try:
            data = json.loads(data_bytes)
            for row in data:
                ts = parse_timestamp(row.get("time_tag"))
                if ts and row.get("Kp") is not None:
                    try: history.append((ts, float(row["Kp"])))
                    except: pass
        except Exception as e:
            log(f"    Kp parse failed: {e}")
    log(f"    Kp records: {len(history)}")
    return history


def fetch_dst_noaa():
    log("  Fetching NOAA Dst...")
    url = "https://services.swpc.noaa.gov/products/kyoto-dst.json"
    data_bytes = fetch_with_retry(url, description="Dst")
    history = []
    if data_bytes:
        try:
            data = json.loads(data_bytes)
            for row in data:
                ts = parse_timestamp(row.get("time_tag"))
                if ts and row.get("dst") is not None:
                    try: history.append((ts, float(row["dst"])))
                    except: pass
        except Exception as e:
            log(f"    Dst parse failed: {e}")
    log(f"    Dst records: {len(history)}")
    return history


def fetch_bz_noaa():
    log("  Fetching NOAA Bz...")
    # Last 24 h of 1-minute L1 data, one row per spacecraft per minute; `active`
    # marks the spacecraft SWPC uses operationally.
    url = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
    data_bytes = fetch_with_retry(url, description="Bz")
    history = []
    if data_bytes:
        try:
            data = json.loads(data_bytes)
            for row in [r for r in data if r.get("active")] or data:
                ts = parse_timestamp(row.get("time_tag"))
                if ts and row.get("bz_gsm") is not None:
                    try: history.append((ts, float(row["bz_gsm"])))
                    except: pass
        except Exception as e:
            log(f"    Bz parse failed: {e}")
    log(f"    Bz records: {len(history)}")
    return history


def fetch_solar_flux_noaa():
    """Daily 20 UT (noon) 10.7 cm flux for the last 30 days plus the current summary value."""
    log("  Fetching NOAA 10cm flux...")
    history = {}
    for url in ("https://services.swpc.noaa.gov/products/10cm-flux-30-day.json",
                "https://services.swpc.noaa.gov/products/summary/10cm-flux.json"):
        data_bytes = fetch_with_retry(url, timeout=15, description="solar flux")
        if data_bytes:
            try:
                for row in json.loads(data_bytes):
                    ts = parse_timestamp(row.get("time_tag"))
                    if ts and row.get("flux") is not None:
                        history[ts] = float(row["flux"])
            except Exception as e:
                log(f"    solar flux parse failed: {e}")
    log(f"    solar flux records: {len(history)}")
    return sorted(history.items())


def fetch_neutron_nmdb():
    log("  Fetching NMDB neutron flux...")
    history = []
    try:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=7)
        url = (f"https://www.nmdb.eu/nest/draw_graph.php?formchk=1&stations[]=NEWK"
               f"&tabchoice=revori&dtype=corr_for_pressure&tresolution=60"
               f"&date_choice=bydate&start_year={start.year}&start_month={start.month:02d}"
               f"&start_day={start.day:02d}&start_hour=0&start_min=0"
               f"&end_year={now.year}&end_month={now.month:02d}"
               f"&end_day={now.day:02d}&end_hour=23&end_min=59&output=ascii")
        data_bytes = fetch_with_retry(url, headers={"User-Agent": "qiskit-calibration-drift"},
                                      description="neutron")
        if data_bytes:
            data = data_bytes.decode("utf-8", errors="replace")
            for ts_str, val in re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2});\s*([\d.]+)", data):
                ts = parse_timestamp(ts_str)
                if ts:
                    try: history.append((ts, float(val)))
                    except: pass
    except Exception as e:
        log(f"    neutron fetch error: {e}")
    log(f"    neutron records: {len(history)}")
    return history


def fetch_weather(station):
    log(f"  Fetching weather ({station})...")
    history = []
    try:
        url = f"https://api.weather.gov/stations/{station}/observations"
        data_bytes = fetch_with_retry(url, headers={"User-Agent": "qiskit-calibration-drift"},
                                       description=f"weather ({station})")
        if data_bytes:
            data = json.loads(data_bytes)
            for feat in data.get("features", []):
                props = feat.get("properties", {})
                ts = parse_timestamp(props.get("timestamp"))
                if not ts: continue
                temp = props.get("temperature", {}).get("value")
                pres = props.get("barometricPressure", {}).get("value")
                hum = props.get("relativeHumidity", {}).get("value")
                history.append((ts, {
                    "temperature_c": round(float(temp), 2) if temp is not None else None,
                    "pressure_hpa": round(float(pres) / 100, 2) if pres is not None else None,
                    "humidity_pct": round(float(hum), 2) if hum is not None else None,
                }))
    except Exception as e:
        log(f"    weather fetch error: {e}")
    log(f"    weather records: {len(history)}")
    return history


def fetch_gfz_indices():
    """GFZ Kp_ap_Ap_SN_F107 since 1932. Returns 3-hourly history."""
    log("  Fetching GFZ Kp/ap/Ap/SN/F10.7...")
    history = []
    urls = [
        "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_nowcast.txt",
        "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt",
    ]
    raw = None
    for u in urls:
        b = fetch_with_retry(u, description=f"GFZ ({u.split('/')[-1]})", timeout=60)
        if b:
            raw = b.decode("utf-8", errors="replace"); break
    if not raw:
        log("    GFZ fetch failed")
        return history
    cutoff_year = datetime.now(timezone.utc).year - 1
    for line in raw.split("\n"):
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        nums = []
        for p in parts:
            try: nums.append(float(p))
            except: pass
        if len(nums) < 27:
            continue
        yyyy, mm, dd = int(nums[0]), int(nums[1]), int(nums[2])
        if yyyy < cutoff_year:
            continue
        kp8 = nums[7:15]; ap8 = nums[15:23]
        Ap = nums[23]; SN = nums[24]; f107_obs = nums[25]; f107_adj = nums[26]
        for i, (kp, ap) in enumerate(zip(kp8, ap8)):
            try:
                ts = datetime(year=yyyy, month=mm, day=dd, hour=i*3, tzinfo=timezone.utc)
            except Exception:
                continue
            history.append((ts, {
                "kp_index_def": float(kp) if kp >= 0 else None,
                "ap_index": float(ap) if ap >= 0 else None,
                "Ap_daily": float(Ap) if Ap >= 0 else None,
                "SN": float(SN) if SN >= 0 else None,
                "f107_observed_sfu": float(f107_obs) if f107_obs >= 0 else None,
                "f107_adjusted_sfu": float(f107_adj) if f107_adj >= 0 else None,
            }))
    log(f"    GFZ records: {len(history)}")
    return history


def fetch_environmental():
    log("Fetching environmental sources...")
    return {
        "kp": fetch_kp_noaa(),
        "dst": fetch_dst_noaa(),
        "bz": fetch_bz_noaa(),
        "solar_flux": fetch_solar_flux_noaa(),
        "neutron": fetch_neutron_nmdb(),
        "weather": {loc["weather_station"]: fetch_weather(loc["weather_station"])
                    for loc in LOCATIONS.values()},
        "gfz": fetch_gfz_indices(),
    }


# (column, env history, field of a dict-valued sample, tolerance, publication lag). A sample
# tagged at the start of its interval is published one interval later; GFZ's file carries
# every 3-hour bin of the current day, so its limit is the day's last bin (lag None).
# kp_index takes GFZ's value and falls back to NOAA's.
COVARIATES = [
    ("temperature_c", "weather", "temperature_c", timedelta(hours=3), timedelta(0)),
    ("pressure_hpa", "weather", "pressure_hpa", timedelta(hours=3), timedelta(0)),
    ("humidity_pct", "weather", "humidity_pct", timedelta(hours=3), timedelta(0)),
    ("bz_gsm_nt", "bz", None, timedelta(minutes=10), timedelta(minutes=1)),
    ("neutron_flux", "neutron", None, timedelta(hours=3), timedelta(hours=1)),
    ("dst_nt", "dst", None, timedelta(hours=3), timedelta(hours=1)),
    ("kp_index", "gfz", "kp_index_def", timedelta(hours=3), None),
    ("ap_index", "gfz", "ap_index", timedelta(hours=3), None),
    ("Ap_daily", "gfz", "Ap_daily", timedelta(hours=3), None),
    ("SN", "gfz", "SN", timedelta(hours=3), None),
    ("f107_observed_sfu", "gfz", "f107_observed_sfu", timedelta(hours=3), None),
    ("f107_adjusted_sfu", "gfz", "f107_adjusted_sfu", timedelta(hours=3), None),
    ("kp_index", "kp", None, timedelta(hours=6), timedelta(hours=3)),
]
COVARIATE_COLUMNS = list(dict.fromkeys([c[0] for c in COVARIATES] + ["solar_flux_sfu"]))
BACKEND_STATION = {b: loc["weather_station"] for loc in LOCATIONS.values() for b in loc["backends"]}


def _us(series):
    import pandas as pd
    return ((series - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(microseconds=1)).to_numpy("int64")


def _series(history, field=None):
    """Sorted sample times (us since epoch) and values from a [(ts, value)] history."""
    import numpy as np
    pts = sorted(history, key=lambda p: p[0])
    t = np.array([int(ts.timestamp() * 1_000_000) for ts, _ in pts], dtype="int64")
    v = np.array([np.nan if (x if field is None else x.get(field)) is None
                  else float(x if field is None else x.get(field)) for _, x in pts], dtype="float64")
    return t, v


def _nearest(src_t, src_v, target, limit, tol):
    """Value of the sample nearest to each target among samples at or before its limit,
    strictly within `tol` us; the earlier sample on a tie; NaN when there is none."""
    import numpy as np
    out = np.full(len(target), np.nan)
    if len(src_t) == 0 or len(target) == 0:
        return out
    last = np.searchsorted(src_t, limit, side="right") - 1
    i = np.searchsorted(src_t, target, side="left")
    a = np.minimum(i, last); b = np.minimum(i - 1, last)
    big = np.iinfo(np.int64).max
    da = np.where(a >= 0, np.abs(src_t[np.clip(a, 0, None)] - target), big)
    db = np.where(b >= 0, np.abs(src_t[np.clip(b, 0, None)] - target), big)
    j = np.clip(np.where(da < db, a, b), 0, None)
    ok = np.minimum(da, db) < tol
    out[ok] = src_v[j[ok]]
    return out


def add_covariates(df, env):
    """Fill the empty covariate cells of `df` (backend, calibrated_time, observed_time and any
    covariate columns) from `env`; cells that already hold a value are kept."""
    import numpy as np
    import pandas as pd
    obs = pd.to_datetime(df["observed_time"], utc=True)
    target = _us(pd.to_datetime(df["calibrated_time"], utc=True).fillna(obs))
    obs_us = _us(obs)
    day_limit = _us(obs.dt.floor("D") + pd.Timedelta(hours=21))
    station = df["backend"].astype(object).map(BACKEND_STATION).to_numpy()
    us = lambda td: int(td / timedelta(microseconds=1))
    for col, key, field, tol, lag in COVARIATES:
        have = (df[col].to_numpy(dtype="float64", na_value=np.nan, copy=True) if col in df
                else np.full(len(df), np.nan))
        need = np.isnan(have)
        if not need.any():
            df[col] = have
            continue
        limit = day_limit if lag is None else obs_us - us(lag)
        if key == "weather":
            groups = [(ws, need & (station == ws)) for ws in env["weather"]]
            hists = {ws: env["weather"][ws] for ws in env["weather"]}
        else:
            groups = [(key, need)]
            hists = {key: env[key]}
        for name, m in groups:
            if m.any():
                st, sv = _series(hists[name], field)
                have[m] = _nearest(st, sv, target[m], limit[m], us(tol))
        df[col] = have
    # SWPC's summary flux at observed_time: the day's 20 UT flux, published from 21 UT
    have = (df["solar_flux_sfu"].to_numpy(dtype="float64", na_value=np.nan, copy=True) if "solar_flux_sfu" in df
            else np.full(len(df), np.nan))
    need = np.isnan(have)
    st, sv = _series(env["solar_flux"])
    if need.any() and len(st):
        pub = st + us(timedelta(hours=1))
        k = np.searchsorted(pub, obs_us, side="right") - 1
        ok = need & (k >= 0) & (obs_us - pub[np.clip(k, 0, None)] < us(timedelta(hours=72)))
        have[ok] = sv[np.clip(k, 0, None)[ok]]
    df["solar_flux_sfu"] = have
    return df


# ---------- IBM properties parsing ----------

def parse_properties_dict(raw, backend_name, observed_dt):
    """Walk the raw BackendProperties dict and emit canonical rows (covariates other than
    solar_zenith_deg are attached later by add_covariates)."""
    lat = None; lon = None
    for loc in LOCATIONS.values():
        if backend_name in loc["backends"]:
            lat = loc["lat"]; lon = loc["lon"]; break

    snapshot_update = parse_timestamp(raw.get("last_update_date"))
    records = []

    # qubit-level entries
    for q_idx, q_entries in enumerate(raw.get("qubits", [])):
        for nd in q_entries:
            name = nd.get("name")
            value = nd.get("value")
            unit = nd.get("unit", "")
            cal_dt = parse_timestamp(nd.get("date"))
            records.append({
                "backend": backend_name,
                "property": name,
                "property_family": FAMILY_MAP.get(name, name),
                "qubit_a": q_idx, "qubit_b": None,
                "value": float(value) if value is not None else None,
                "unit": unit, "scope": "qubit",
                "is_failure_ceiling": _is_ceiling(name, value),
                "observed_time": observed_dt,
                "calibrated_time": cal_dt,
                "snapshot_update_time": snapshot_update,
                "calibration_age_seconds": ((observed_dt - cal_dt).total_seconds()
                                            if cal_dt else None),
                "latitude": lat, "longitude": lon,
                "solar_zenith_deg": solar_zenith(lat, lon, cal_dt or observed_dt),
            })

    # gate-level entries
    for g in raw.get("gates", []):
        gname = g.get("gate")
        qubits = g.get("qubits", [])
        qa = int(qubits[0]) if len(qubits) >= 1 else None
        qb = int(qubits[1]) if len(qubits) >= 2 else None
        scope = "gate2q" if qb is not None else "gate1q"
        for param in g.get("parameters", []):
            pname = param.get("name")
            value = param.get("value")
            unit = param.get("unit", "")
            cal_dt = parse_timestamp(param.get("date"))
            prop = f"{gname}_{pname}"
            records.append({
                "backend": backend_name,
                "property": prop,
                "property_family": FAMILY_MAP.get(prop, prop),
                "qubit_a": qa, "qubit_b": qb,
                "value": float(value) if value is not None else None,
                "unit": unit, "scope": scope,
                "is_failure_ceiling": _is_ceiling(prop, value),
                "observed_time": observed_dt,
                "calibrated_time": cal_dt,
                "snapshot_update_time": snapshot_update,
                "calibration_age_seconds": ((observed_dt - cal_dt).total_seconds()
                                            if cal_dt else None),
                "latitude": lat, "longitude": lon,
                "solar_zenith_deg": solar_zenith(lat, lon, cal_dt or observed_dt),
            })

    return records


def _is_ceiling(prop, value):
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if prop.endswith("_error") or prop.startswith("prob_meas"):
        return v >= 0.999
    return False


# ---------- Existing dataset handling ----------

DATA_FILE = "data/train-00000-of-00001.parquet"
KEY_COLS = ["backend", "property", "qubit_a", "qubit_b"]
ROW_GROUP_ROWS = 1024 * 1024


def load_existing_dataset():
    """Download the existing parquet and return its local path, or None to start fresh.
    Only the file's schema is read here; rows are streamed later."""
    log("Loading existing HF dataset...")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    import pyarrow.parquet as pq
    try:
        path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=DATA_FILE)
    except EntryNotFoundError:
        log("  no existing dataset file; starting fresh.")
        return None
    except Exception as e:
        log(f"FATAL: could not download existing dataset: {e}")
        sys.exit(1)
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    log(f"  found {pf.metadata.num_rows:,} rows, {len(names)} cols")
    if "qubit_a" not in names and "qubit" in names:
        log("  legacy 17-col schema detected; will be replaced by new-schema rows.")
        return None
    if names != CANONICAL_COLUMNS:
        log(f"FATAL: existing columns differ from the canonical schema: {names}")
        sys.exit(1)
    return path


def _qubit(q):
    import pandas as pd
    return None if q is None or pd.isna(q) else int(q)


def _unit_key(backend, prop, qubit_a, qubit_b):
    """(backend, property, qubit_a, qubit_b) with qubit indices as int and missing ones as None,
    so keys read from the parquet (Int64/Float64, NA) match keys built from parsed records."""
    return (backend, prop, _qubit(qubit_a), _qubit(qubit_b))


def read_last_observed(path):
    """Latest observed_time per backend in the existing parquet."""
    import pyarrow.parquet as pq
    if path is None:
        return {}
    df = pq.read_table(path, columns=["backend", "observed_time"], read_dictionary=["backend"]).to_pandas()
    return {str(b): t.to_pydatetime() for b, t in df.groupby("backend", observed=True)["observed_time"].max().items()}


def read_existing_state(path, records):
    """Read the key columns of the existing parquet and return (prior, keys): the latest
    (calibrated_time, value) per unit key, and every (unit key, calibrated_time) whose
    calibrated_time occurs in `records`."""
    import pandas as pd
    import pyarrow.parquet as pq
    prior, keys = {}, set()
    if path is None:
        return prior, keys
    df = pq.read_table(path, columns=KEY_COLS + ["calibrated_time", "value"],
                       read_dictionary=["backend", "property"]).to_pandas()
    times = {r["calibrated_time"] for r in records}
    hit = df["calibrated_time"].isin(pd.to_datetime([t for t in times if t is not None], utc=True))
    if None in times:
        hit |= df["calibrated_time"].isna()
    for b, p, qa, qb, t in df.loc[hit, KEY_COLS + ["calibrated_time"]].itertuples(index=False):
        keys.add(_unit_key(b, p, qa, qb) + (None if pd.isna(t) else t.to_pydatetime(),))
    df = df[df["calibrated_time"].notna()]
    last = df.groupby(KEY_COLS, dropna=False, observed=True, sort=False)["calibrated_time"].idxmax()
    for b, p, qa, qb, t, v in df.loc[last, KEY_COLS + ["calibrated_time", "value"]].itertuples(index=False):
        prior[_unit_key(b, p, qa, qb)] = (t.to_pydatetime(), v)
    return prior, keys


CANONICAL_COLUMNS = [
    "backend", "property_family", "property", "qubit_a", "qubit_b", "value", "unit", "scope",
    "is_failure_ceiling", "observed_time", "calibrated_time", "snapshot_update_time",
    "calibration_age_seconds", "is_new_measurement", "chipwide_recal_event_id",
    "latitude", "longitude",
    "solar_zenith_deg", "temperature_c", "pressure_hpa", "humidity_pct",
    "bz_gsm_nt", "neutron_flux", "kp_index", "ap_index", "Ap_daily", "SN",
    "f107_observed_sfu", "f107_adjusted_sfu", "solar_flux_sfu", "dst_nt",
]


def to_canonical(df):
    import pandas as pd
    for c in CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[CANONICAL_COLUMNS]


def detect_chipwide(records):
    """Group rows whose (backend, calibrated_time) match and that span >= 20% of
    that backend's units; assign a shared chipwide_recal_event_id."""
    if not records:
        return
    by_bt = {}
    for i, r in enumerate(records):
        key = (r["backend"], r["calibrated_time"])
        by_bt.setdefault(key, []).append(i)
    backend_counts = {}
    for r in records:
        backend_counts[r["backend"]] = backend_counts.get(r["backend"], 0) + 1
    for (be, t), idxs in by_bt.items():
        if t is None or backend_counts.get(be, 0) == 0:
            continue
        if len(idxs) / backend_counts[be] >= 0.20:
            eid = f"{be}__chipwide__{t.strftime('%Y%m%dT%H%M%SZ')}"
            for i in idxs:
                records[i]["chipwide_recal_event_id"] = eid


def new_rows(records, existing_keys):
    """Records in poll order whose (unit key, calibrated_time) is neither in the dataset nor
    earlier in `records`."""
    seen, out = set(existing_keys), []
    for r in records:
        k = _unit_key(r["backend"], r["property"], r["qubit_a"], r["qubit_b"]) + (r["calibrated_time"],)
        if k not in seen:
            seen.add(k); out.append(r)
    return out


def compute_is_new(new_records, prior):
    """Set is_new_measurement on deduplicated records in poll order: true when the unit has no
    earlier row, or `value` is set and differs from the value of the unit's latest calibration
    so far. `prior` maps unit key -> (calibrated_time, value) and is updated in place."""
    for r in new_records:
        k = _unit_key(r["backend"], r["property"], r["qubit_a"], r["qubit_b"])
        pct, pv = prior.get(k, (None, None))
        r["is_new_measurement"] = k not in prior or (r["value"] is not None and pv != r["value"])
        ct = r["calibrated_time"]
        if ct is not None and (pct is None or ct > pct):
            prior[k] = (ct, r["value"])


# ---------- IBM connection + main ----------

def connect_ibm():
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if token:
        return QiskitRuntimeService(
            channel="ibm_cloud", token=token,
            instance="crn:v1:bluemix:public:quantum-computing:us-east:"
                     "a/a9114248c6c44fe88a40cda24e7073c3:"
                     "a72852a9-5e25-429b-b8fc-8ac73fb30240::"
        )
    return QiskitRuntimeService()


def refill(tbl, env):
    """Fill the empty covariate cells of an Arrow table from `env`; returns (table, cells filled)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    df = add_covariates(tbl.select(["backend", "calibrated_time", "observed_time"] + COVARIATE_COLUMNS)
                        .to_pandas(), env)
    n = 0
    for c in COVARIATE_COLUMNS:
        col = tbl.column(c).combine_chunks()
        v = df[c].to_numpy(dtype="float64", na_value=np.nan)
        out = pc.coalesce(col, pa.array(v, mask=np.isnan(v), type=col.type))
        n += col.null_count - out.null_count
        i = tbl.schema.get_field_index(c)
        tbl = tbl.set_column(i, tbl.schema.field(i), out)
    return tbl, n


def write_combined(existing_path, new_df, out_path, env):
    """Write the existing rows followed by `new_df` to `out_path`, holding one row group in
    memory at a time, and refill the row groups holding rows observed in the last REFILL_DAYS;
    returns (combined row count, cells refilled). The last existing row group is merged with
    the new rows, so row groups stay at ROW_GROUP_ROWS as with a whole-frame write."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if existing_path is None:
        new_df.to_parquet(out_path, index=False)
        return len(new_df), 0
    pf = pq.ParquetFile(existing_path)
    schema = pf.schema_arrow
    new_tbl = pa.Table.from_pandas(new_df, schema=schema, preserve_index=False)
    new_tbl = new_tbl.replace_schema_metadata(schema.metadata)
    since = datetime.now(timezone.utc) - timedelta(days=REFILL_DAYS)
    obs_col = schema.get_field_index("observed_time")

    def recent(i):
        st = pf.metadata.row_group(i).column(obs_col).statistics
        if st is None or not st.has_min_max:
            return True
        mx = st.max if st.max.tzinfo else st.max.replace(tzinfo=timezone.utc)
        return mx >= since

    last = pf.num_row_groups - 1
    filled = 0
    with pq.ParquetWriter(out_path, schema) as w:
        for i in range(last):
            rg = pf.read_row_group(i)
            if recent(i):
                rg, n = refill(rg, env); filled += n
            w.write_table(rg)
        tail = pa.concat_tables([pf.read_row_group(last), new_tbl]) if last >= 0 else new_tbl
        tail, n = refill(tail, env); filled += n
        w.write_table(tail, row_group_size=ROW_GROUP_ROWS)
    return pf.metadata.num_rows + len(new_df), filled


def gap_times(last_seen, now):
    """Half-hour marks after `last_seen` (at most GAP_DAYS back) up to ten minutes before `now`."""
    if last_seen is None:
        return []
    start = max(last_seen, now - timedelta(days=GAP_DAYS))
    t = start.replace(minute=0, second=0, microsecond=0)
    out = []
    while t < now - timedelta(minutes=10):
        if t > last_seen + timedelta(minutes=10):
            out.append(t)
        t += timedelta(minutes=30)
    return out


def snapshot_records(be, observed_dt, at=None):
    """Parsed records of one BackendProperties snapshot (the live one, or the historical one
    at `at`) with chip-wide events marked; [] when it cannot be fetched."""
    try:
        props_obj = be.properties(datetime=at) if at else be.properties()
        if props_obj is None:
            return []
        recs = parse_properties_dict(props_obj.to_dict(), be.name, observed_dt)
    except Exception as e:
        log(f"  property fetch failed ({be.name} {at or 'live'}): {e}")
        return []
    detect_chipwide(recs)
    return recs


def push(path, n_new):
    from huggingface_hub import HfApi
    # Upload by file (avoids dataset.push_to_hub schema issues)
    log(f"  wrote {path} ({os.path.getsize(path):,} bytes)")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=DATA_FILE,
        repo_id=REPO_ID, repo_type="dataset",
        commit_message=f"poll {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}: "
                       f"+{n_new} rows",
    )
    log("  uploaded to HF")


def main():
    log("=" * 60)
    log("IBM Quantum Calibration Drift Poller")
    log(f"Target: {REPO_ID}")
    log("=" * 60)

    env = fetch_environmental()

    existing = load_existing_dataset()

    try:
        service = connect_ibm()
    except Exception as e:
        log(f"FATAL: IBM connect failed: {e}")
        log(traceback.format_exc())
        sys.exit(1)

    log("Fetching backend list...")
    backends = list(service.backends())
    log(f"  {len(backends)} backends: {[b.name for b in backends]}")

    observed_dt = datetime.now(timezone.utc).replace(microsecond=0)
    last_seen = read_last_observed(existing)
    all_records = []
    for be in backends:
        times = gap_times(last_seen.get(be.name), observed_dt)
        log(f"Processing {be.name}: {len(times)} historical snapshot(s) since "
            f"{last_seen.get(be.name)}, then live...")
        prev, n = set(), 0
        for at in times + [None]:
            recs = snapshot_records(be, at or observed_dt, at)
            keys = [_unit_key(r["backend"], r["property"], r["qubit_a"], r["qubit_b"])
                    + (r["calibrated_time"],) for r in recs]
            # rows unchanged since the previous snapshot cannot be new
            all_records.extend(r for r, k in zip(recs, keys) if k not in prev)
            prev = set(keys) or prev
            n += len(recs)
        log(f"  {n} records")

    if not all_records:
        log("No records extracted; exiting.")
        sys.exit(0)

    log(f"Total extracted: {len(all_records):,}")
    prior, existing_keys = read_existing_state(existing, all_records)
    kept = new_rows(all_records, existing_keys)
    compute_is_new(kept, prior)
    log(f"After dedup vs existing: {len(kept):,} truly new rows")

    import pandas as pd
    new_df = to_canonical(pd.DataFrame(kept, columns=CANONICAL_COLUMNS))
    for c in ("observed_time", "calibrated_time", "snapshot_update_time"):
        new_df[c] = pd.to_datetime(new_df[c], utc=True)
    new_df = add_covariates(new_df, env)

    tmp = "/tmp/qiskit_train.parquet" if os.name != "nt" else os.path.join(
        os.environ.get("TEMP", "."), "qiskit_train.parquet")
    total, filled = write_combined(existing, new_df, tmp, env)
    log(f"Combined dataset: {total:,} rows; {filled:,} empty covariate cells of recent rows filled")

    push(tmp, len(new_df))
    log("Done.")
    sys.exit(0)


if __name__ == "__main__":
    main()
