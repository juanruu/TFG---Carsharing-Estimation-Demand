"""
trip_cleaning.py
================

Shared trip-record cleaning utilities used by both the XGBoost (Milan) and
the multi-city LSTM pipelines. Centralising the logic guarantees that both
predictive frameworks operate on a dataset cleaned with identical criteria,
which is essential for a fair cross-framework comparison.

The cleaning pipeline addresses the four classes of degenerate or anomalous
records that appear in free-floating carsharing operator logs:

    1. Trips with effectively zero displacement  (distance < 100 m)
    2. Trips with anomalous durations            (< 1 min or > 8 h)
    3. Trips outside the operational area        (per-city geographic outliers)
    4. Exact duplicate rows                      (re-ingestion artefacts)

Each filter records the number of trips removed so that the pipeline can
report a per-city audit trail.
"""

from __future__ import annotations
import re
from typing import Dict, Tuple
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Default thresholds (rationale documented inline)
# ---------------------------------------------------------------------------
MIN_DIST_KM   = 0.10    # < 100 m: vehicle did not actually move
MIN_DUR_MIN   = 1       # < 1 min: telemetry glitches / cancelled reservations
MAX_DUR_MIN   = 480     # > 8 h : free-floating providers cap rentals around this value;
                        # longer rentals usually indicate stranded vehicles or
                        # operator maintenance windows, not user demand.
BBOX_QUANTILE = 0.999   # drop points outside the per-city 0.1%-99.9% lat/lon range


# ---------------------------------------------------------------------------
# Low-level parsers (shared between both notebooks)
# ---------------------------------------------------------------------------
_DATE_TZ_NAME_RE = re.compile(r'\s*\(.*?\)')


def parse_distance_km(value) -> float:
    """Parse a string such as '1.4 km' or '4 m' into a float in km.

    Returns NaN if the value cannot be parsed (e.g. truly missing).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).lower().replace(',', '.').strip()
    try:
        if 'km' in s:
            return float(s.replace('km', '').strip())
        if 'm' in s:
            return float(s.replace('m', '').strip()) / 1000.0
        return float(s)
    except ValueError:
        return np.nan


def parse_duration_min(value) -> float:
    """Parse a string such as '29 mins' or '211 mins' into integer minutes."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    try:
        return float(str(value).split(' ')[0])
    except ValueError:
        return np.nan


def parse_coord(value) -> Tuple[float, float]:
    """Parse the 'lon,lat,elevation' string into a (lat, lon) tuple."""
    parts = str(value).strip().split(',')
    return float(parts[1]), float(parts[0])


def parse_date_local(value) -> pd.Timestamp:
    """Parse the trip-log timestamp keeping local wall-clock time.

    Strips both the 'GMT+ZZZZ' suffix and the trailing time-zone name
    (e.g. '(W. Europe Daylight Time)'). Produces a naive Timestamp in the
    operator's local time, which is what the XGBoost pipeline assumes.
    """
    if value is None:
        return pd.NaT
    s = _DATE_TZ_NAME_RE.sub('', str(value))                # drop "(name)"
    s = re.sub(r'\s*GMT[+-]\d{4}\s*$', '', s).strip()        # drop "GMT+0200"
    return pd.to_datetime(s, format='%a %b %d %Y %H:%M:%S', errors='coerce')


def parse_date_utc(value) -> pd.Timestamp:
    """Parse the trip-log timestamp converting to UTC.

    Used by the multi-city LSTM pipeline so that all cities share a common
    absolute time reference (hour-of-day is recovered later via local
    re-conversion if needed).
    """
    if value is None:
        return pd.NaT
    s = _DATE_TZ_NAME_RE.sub('', str(value)).strip()
    ts = pd.to_datetime(s, format='%a %b %d %Y %H:%M:%S GMT%z', errors='coerce')
    if ts is pd.NaT:
        return pd.NaT
    return ts.tz_convert('UTC')


# ---------------------------------------------------------------------------
# Main cleaning routine
# ---------------------------------------------------------------------------
def clean_trips(
    df: pd.DataFrame,
    city: str,
    *,
    min_dist_km:  float = MIN_DIST_KM,
    min_dur_min:  float = MIN_DUR_MIN,
    max_dur_min:  float = MAX_DUR_MIN,
    bbox_quantile: float = BBOX_QUANTILE,
    verbose:      bool   = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Apply the standard cleaning pipeline to a city's raw trip dataframe.

    The input dataframe is expected to contain at least these columns
    (the original schema of the *_trips.txt files):
        s_date, s_coord, traveltime, google_distance, vin

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned dataframe with two derived columns added (`dist_km`, `dur_min`)
        and the geographic outliers / degenerate trips removed.
    stats : dict
        Per-step record counts for reporting.
    """
    n0 = len(df)
    stats = {'city': city, 'initial': n0}

    df = df.copy()

    # --- 1. Derive numeric quantities ---
    df['dist_km'] = df['google_distance'].apply(parse_distance_km)
    df['dur_min'] = df['traveltime'].apply(parse_duration_min)

    # --- 2. Drop trips with effectively zero displacement ---
    mask = df['dist_km'] >= min_dist_km
    stats['removed_short_distance'] = int((~mask).sum())
    df = df[mask]

    # --- 3. Drop duration outliers (telemetry glitches / stranded vehicles) ---
    mask = (df['dur_min'] >= min_dur_min) & (df['dur_min'] <= max_dur_min)
    stats['removed_duration_anomaly'] = int((~mask).sum())
    df = df[mask]

    # --- 4. Drop geographic outliers via per-city percentile bounding box ---
    coords = df['s_coord'].apply(parse_coord)
    df['lat'] = coords.apply(lambda x: x[0])
    df['lon'] = coords.apply(lambda x: x[1])

    lo, hi = (1.0 - bbox_quantile), bbox_quantile
    lat_lo, lat_hi = df['lat'].quantile([lo, hi])
    lon_lo, lon_hi = df['lon'].quantile([lo, hi])
    mask = (
        df['lat'].between(lat_lo, lat_hi) & df['lon'].between(lon_lo, lon_hi)
    )
    stats['removed_geo_outliers'] = int((~mask).sum())
    df = df[mask]

    # --- 5. Drop exact duplicates (re-ingestion artefacts) ---
    n_before = len(df)
    df = df.drop_duplicates(subset=['vin', 's_date', 's_coord'])
    stats['removed_duplicates'] = n_before - len(df)

    # --- Final accounting ---
    stats['final']        = len(df)
    stats['removed_total'] = n0 - len(df)
    stats['removed_pct']   = round(100 * (n0 - len(df)) / n0, 2) if n0 else 0.0

    if verbose:
        print(
            f"[{city:>10s}] "
            f"raw={stats['initial']:>7,d}  "
            f"-dist={stats['removed_short_distance']:>5,d}  "
            f"-dur={stats['removed_duration_anomaly']:>5,d}  "
            f"-geo={stats['removed_geo_outliers']:>4,d}  "
            f"-dup={stats['removed_duplicates']:>4,d}  "
            f"-> clean={stats['final']:>7,d} "
            f"({stats['removed_pct']:.2f}% removed)"
        )

    return df.reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# Optional: outlier diagnosis (used in the EDA section of section 3.1.3)
# ---------------------------------------------------------------------------
def trip_quality_summary(df_raw: pd.DataFrame, city: str) -> Dict[str, float]:
    """Compute a non-destructive summary of trip-record quality.

    Useful for the exploratory analysis section: it reports how many trips
    *would* be removed by each filter without actually applying them.
    """
    dist = df_raw['google_distance'].apply(parse_distance_km)
    dur  = df_raw['traveltime'].apply(parse_duration_min)

    return {
        'city': city,
        'n_total': len(df_raw),
        'pct_short_distance': round(100 * (dist < MIN_DIST_KM).mean(), 2),
        'pct_too_short':      round(100 * (dur  < MIN_DUR_MIN).mean(), 2),
        'pct_too_long':       round(100 * (dur  > MAX_DUR_MIN).mean(), 2),
        'dist_median_km':     round(float(dist.median()), 2),
        'dist_p99_km':        round(float(dist.quantile(0.99)), 2),
        'dur_median_min':     round(float(dur.median()), 1),
        'dur_p99_min':        round(float(dur.quantile(0.99)), 1),
    }
