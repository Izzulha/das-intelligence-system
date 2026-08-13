#!/usr/bin/env python3

from __future__ import annotations

import base64
import math
import re
import traceback
import json
import os
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import dash
from dash import (
    Dash, Input, Output, State, callback_context,
    dcc, html, dash_table, ALL, no_update,
)
from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap

try:
    import joblib
    _JOBLIB_OK = True
except Exception:
    _JOBLIB_OK = False

try:
    from PIL import Image as _PILImage  # noqa: F401
    _PILLOW_OK = True
except Exception:
    _PILLOW_OK = False

try:
    import selenium  # noqa: F401
    _SELENIUM_OK = True
except Exception:
    _SELENIUM_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
LOG_FILE = Path(os.getenv("DAS_LOG_FILE", "das_intelligence.log"))
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")],
)
logger = logging.getLogger(__name__)


def log_error(context: str, error: Exception, severity: str = "ERROR") -> None:
    """Unified error logging for callbacks and data/model operations."""
    logger.log(
        level=getattr(logging, severity.upper(), logging.ERROR),
        msg=f"[{context}] {type(error).__name__}: {error}",
    )
    traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# App bootstrap
# ──────────────────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="DAS Intelligence",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
PARENT_DIR  = PROJECT_DIR.parent
DATA_DIR    = PROJECT_DIR / "alarm_data"

# ──────────────────────────────────────────────────────────────────────────────
# PostgreSQL connection
# ──────────────────────────────────────────────────────────────────────────────
DB_URL = os.getenv("DAS_DB_URL")
DB_TABLE = os.getenv("DAS_DB_TABLE", "alarm_data")

# Do not crash the whole Dash app when the database URL is missing.
# The app will open with an empty dataset and print the exact fix in the terminal.
DB_CONNECTION_ERROR = None
if not DB_URL:
    DB_CONNECTION_ERROR = (
        "DAS_DB_URL is not set. Set it before running app.py. "
        "PowerShell example: $env:DAS_DB_URL = 'postgresql://USER:PASSWORD@HOST:5432/DBNAME'"
    )
    print(f"[db] {DB_CONNECTION_ERROR}")

_db_engine = None


def get_engine():
    global _db_engine
    if not DB_URL:
        raise RuntimeError(DB_CONNECTION_ERROR or "DAS_DB_URL is not set.")
    if _db_engine is None:
        _db_engine = create_engine(DB_URL, pool_pre_ping=True)
    return _db_engine


def load_from_db() -> pd.DataFrame:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(text(f'SELECT * FROM "{DB_TABLE}"'), conn)
        print(f"[db] Loaded {len(df):,} rows from {DB_TABLE!r}")
        return df
    except Exception as e:
        print(f"[db] Failed to load from PostgreSQL: {e}")
        return pd.DataFrame()


MODEL_CANDIDATES = [
    Path(__file__).parent / "repeated_alarm_model.joblib",
    Path(__file__).parent / "data" / "repeated_alarm_model.joblib",
]
METADATA_CANDIDATES = [
    Path(__file__).parent / "repeated_alarm_metadata.json",
    Path(__file__).parent / "data" / "repeated_alarm_metadata.json",
]

POSITION_BIN_SIZE  = 5000
INCIDENT_MAP_LIMIT = 500

PAGES = [
    {"key": "overview",   "label": "Overview",    "icon": "▦"},
    {"key": "trends",     "label": "Trends",      "icon": "⌁"},
    {"key": "hotspots",   "label": "Hotspots",    "icon": "◎"},
    {"key": "map",        "label": "Map",         "icon": "⌖"},
    {"key": "prediction", "label": "Prediction",  "icon": "▣"},
    {"key": "explorer",   "label": "Report",      "icon": "◉"},
]

COLORS = {
    # Professional, user-friendly palette: calm neutrals + accessible accents.
    # Red/amber are reserved for warning states so normal category charts do not
    # accidentally look like errors.
    "bg":          "#F6F8FB",
    "surface":     "#FFFFFF",
    "surface_2":   "#F8FAFC",
    "surface_3":   "#EEF2F7",
    "border":      "#E2E8F0",
    "border_2":    "#CBD5E1",
    "text":        "#0F172A",
    "text_2":      "#334155",
    "text_3":      "#64748B",
    "text_4":      "#94A3B8",
    "blue":        "#2563EB",
    "blue_light":  "#DBEAFE",
    "blue_soft":   "#EFF6FF",
    "indigo":      "#4F46E5",
    "violet":      "#7C3AED",
    "cyan":        "#0891B2",
    "cyan_light":  "#CFFAFE",
    "teal":        "#0F766E",
    "teal_light":  "#CCFBF1",
    "green":       "#15803D",
    "green_light": "#DCFCE7",
    "amber":       "#F59E0B",
    "amber_dark":  "#D97706",
    "amber_light": "#FEF3C7",
    "red":         "#DC2626",
    "red_light":   "#FEE2E2",
    "pink":        "#BE185D",
    "slate":       "#475569",
    "slate_light": "#E2E8F0",
}

THREAT_MAP = {
    "red":      COLORS["red"],
    "high":     COLORS["red"],
    "critical": COLORS["red"],
    "amber":    COLORS["amber_dark"],
    "medium":   COLORS["amber_dark"],
    "yellow":   COLORS["amber_dark"],
    "green":    COLORS["green"],
    "low":      COLORS["green"],
    "unknown":  COLORS["text_4"],
}

# ── Professional chart palettes ─────────────────────────────────────────────
# Categorical colors are color-blind friendly and calm. Warning colors only appear
# in threat/status charts or when highlighting risk.
PALETTE = [
    "#2563EB",  # blue
    "#0F766E",  # teal
    "#4F46E5",  # indigo
    "#0891B2",  # cyan
    "#7C3AED",  # violet
    "#15803D",  # green
    "#475569",  # slate
    "#BE185D",  # rose
    "#334155",  # dark slate
    "#64748B",  # grey slate
]

BAR_SEQ_BLUE   = ["#EFF6FF", "#DBEAFE", "#BFDBFE", "#93C5FD", "#60A5FA", "#2563EB"]
BAR_SEQ_TEAL   = ["#F0FDFA", "#CCFBF1", "#99F6E4", "#5EEAD4", "#2DD4BF", "#0F766E"]
BAR_SEQ_INDIGO = ["#EEF2FF", "#E0E7FF", "#C7D2FE", "#A5B4FC", "#818CF8", "#4F46E5"]
BAR_SEQ_ROUTE  = ["#F8FAFC", "#E2E8F0", "#CBD5E1", "#94A3B8", "#64748B", "#334155"]
BAR_SEQ_WARM   = ["#FEF3C7", "#FDE68A", "#F59E0B", "#D97706", "#B45309"]
HEATMAP_SCALE  = [[0.00, "#F8FAFC"], [0.25, "#DBEAFE"], [0.55, "#93C5FD"], [0.80, "#2563EB"], [1.00, "#1E3A8A"]]

DAYS_OF_WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ALL_MONTHS_VALUE = "__all__"

# WITH THIS:
MODEL_FEATURES = [
    "route", "type", "position_bin",
    "dominant_fibre",
    "month_num", "month_sin", "month_cos",
    "alarm_count", "accident_count", "alarm_share",
    "avg_hour", "std_hour", "most_common_hour",
    "peak_hour_share", "weekend_share",
    "avg_threat_score", "max_threat_score",
    "avg_resolution_hrs", "max_resolution_hrs", "prev_avg_resolution_hrs",
    "unresolved_count", "unresolved_share", "roll3_unresolved",
    "dominant_status",
    "avg_lat", "avg_lon", "avg_route_dist", "distinct_markers",
    "active_days",
    "prev_month_count", "prev2_month_count", "prev3_month_count",
    "roll3_mean", "roll3_std", "roll6_mean", "roll6_std",
    "ewma3_count", "momentum_diff",
    "past_total_count", "past_months_active", "past_avg_count", "active_ratio",
    "had_prev_month", "trend_diff",
    "months_since_last_seen", "months_since_first_seen",
    "route_type_popularity", "neighbor_bin_count",
]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def first_existing(paths: list[Path]) -> Optional[Path]:
    return next((p for p in paths if p.exists()), None)


_NA_STRINGS = frozenset({"nan", "none", "nat", "", "n/a", "null", "unknown"})

COLUMN_ALIASES = {
    "id": "ID", "alarm id": "ID", "event id": "ID",
    "time": "Time", "date": "Time", "datetime": "Time",
    "date time": "Time", "timestamp": "Time",
    "type": "Type", "event type": "Type", "alarm type": "Type",
    "status": "Status",
    "threat level": "Threat level", "threat": "Threat level",
    "resolved on": "Resolved on", "resolved at": "Resolved on",
    "resolved time": "Resolved on",
    "fibre line": "Fibre Line", "fiber line": "Fibre Line", "fibre": "Fibre Line",
    "position (m)": "Position (m)", "position m": "Position (m)", "position": "Position (m)", "pos": "Position (m)",
    "latitude": "Latitude", "lat": "Latitude",
    "longitude": "Longitude", "lon": "Longitude", "lng": "Longitude",
    "route": "Route",
    "route distance (m)": "Route distance (m)",
    "route distance m": "Route distance (m)",
    "route distance": "Route distance (m)",
    "marker name": "Marker name", "marker": "Marker name",
    "occurrence type": "Occurrence type", "occurrence": "Occurrence type",
    "event class": "Occurrence type",
    "region": "Region",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned_cols: list[str] = []
    seen: dict[str, int] = {}
    for col in df.columns:
        raw = str(col).replace("\ufeff", "").strip()
        key = re.sub(r"[\s_]+", " ", raw.lower()).strip()
        new_col = COLUMN_ALIASES.get(key, raw)
        if new_col in seen:
            seen[new_col] += 1
            new_col = f"{new_col}.{seen[new_col]}"
        else:
            seen[new_col] = 0
        cleaned_cols.append(new_col)
    df.columns = cleaned_cols
    return df


def _series_or_na(df: pd.DataFrame, col: str, default=pd.NA) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _clean_col(s) -> pd.Series:
    if not isinstance(s, pd.Series):
        s = pd.Series([s] if s is not None else [pd.NA])
    return s.astype(str).str.strip().where(
        lambda x: ~x.str.lower().isin(_NA_STRINGS), pd.NA
    )


def _to_datetime_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Parse datetimes safely and normalize timezone-aware values to naive UTC.

    This prevents errors when PostgreSQL / CSV values mix timezone-aware and
    timezone-naive timestamps.
    """
    parsed = pd.to_datetime(_series_or_na(df, col), errors="coerce", dayfirst=True, utc=True)
    return parsed.dt.tz_convert(None)


def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(_series_or_na(df, col), errors="coerce")


def _pos_range_label(bin_idx: int, bin_size: int = POSITION_BIN_SIZE) -> str:
    """Generate a human-readable position range label like 0–4,999 m."""
    start = int(bin_idx) * bin_size
    end   = start + bin_size - 1
    return f"{start:,}\u2013{end:,} m"


def _positive_int(value, default: int, minimum: int = 1) -> int:
    """Safely parse Dash input values that may arrive as strings, numbers, or None."""
    try:
        val = int(float(value))
        return max(minimum, val) if val > 0 else default
    except (TypeError, ValueError):
        return default


def _positive_float(value, default: Optional[float] = None) -> Optional[float]:
    """Safely parse optional positive numeric inputs from Dash controls."""
    try:
        val = float(value)
        return val if val > 0 else default
    except (TypeError, ValueError):
        return default


def _validate_numeric_input(
    value,
    default: int | float | None,
    minimum: int | float = 1,
    maximum: int | float | None = None,
    step: int | float | None = None,
) -> int | float | None:
    """Safely validate and constrain numeric Dash inputs."""
    if value is None or value == "":
        return default
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return default
        val = max(minimum, val)
        if maximum is not None:
            val = min(maximum, val)
        if step is not None and step > 0:
            val = round(val / step) * step
        return int(val) if isinstance(default, int) else val
    except (TypeError, ValueError):
        return default


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)

    df["_dt"]     = _to_datetime_series(df, "Time")
    df["_month"]  = df["_dt"].dt.month
    df["_year"]   = df["_dt"].dt.year
    df["_hour"]   = df["_dt"].dt.hour
    df["_dow"]    = df["_dt"].dt.dayofweek
    df["_ym"]     = df["_dt"].dt.strftime("%Y-%m")
    df["_ym_ts"]  = df["_dt"].dt.to_period("M").dt.to_timestamp()

    df["_resolved_dt"] = _to_datetime_series(df, "Resolved on")
    try:
        df["_res_minutes"] = (df["_resolved_dt"] - df["_dt"]).dt.total_seconds() / 60.0
    except Exception as e:
        log_error("prepare_data resolution-time calculation", e, "WARNING")
        df["_res_minutes"] = np.nan
    df.loc[df["_res_minutes"] < 0, "_res_minutes"] = np.nan

    for old, new in [
        ("Type",            "_type"),
        ("Status",          "_status"),
        ("Region",          "_region"),
        ("Occurrence type", "_occ"),
        ("Fibre Line",      "_fibre"),
        ("Route",           "_route"),
        ("Marker name",     "_marker"),
    ]:
        df[new] = _clean_col(_series_or_na(df, old))

    # Normalize event type at the source so values like
    # "animal", "Animal", and "ANIMAL" are treated as one category
    # across every chart, table, report, and prediction feature.
    if "_type" in df.columns:
        df["_type"] = df["_type"].astype("string").str.strip().str.title()

    df["_threat"]     = _clean_col(_series_or_na(df, "Threat level"))
    df["_threat_key"] = (
        df["_threat"].fillna("unknown").astype(str).str.strip().str.lower()
    )

    df["_pos"]  = _to_numeric_series(df, "Position (m)")
    df["_dist"] = _to_numeric_series(df, "Route distance (m)")
    df["_lat"]  = _to_numeric_series(df, "Latitude")
    df["_lon"]  = _to_numeric_series(df, "Longitude")

    df["_event_class"] = df["_occ"].astype(str).str.strip().str.title()
    df["_event_class"] = df["_event_class"].where(
        df["_event_class"].isin(["Alarm", "Incident"]), "Alarm",
    )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Server-side data store
# ──────────────────────────────────────────────────────────────────────────────
_APP_DATA: pd.DataFrame = pd.DataFrame()
_APP_DATA_VERSION: int = 0


def _set_app_data(df: pd.DataFrame) -> str:
    global _APP_DATA, _APP_DATA_VERSION
    _APP_DATA = df
    _APP_DATA_VERSION += 1
    return f"v{_APP_DATA_VERSION}"


def _df_from_store(_version_token) -> pd.DataFrame:
    if _APP_DATA.empty:
        return _APP_DATA
    return _APP_DATA.copy()


def to_store(df: pd.DataFrame) -> str:
    return _set_app_data(df)


def from_store(records) -> pd.DataFrame:
    return _df_from_store(records)


DATA_SOURCE_FILENAME = "—"


def load_initial() -> str:
    global DATA_SOURCE_FILENAME
    try:
        raw = load_from_db()
        if raw.empty:
            DATA_SOURCE_FILENAME = "—"
            return _set_app_data(pd.DataFrame())
        DATA_SOURCE_FILENAME = f"postgresql://{DB_TABLE}"
        return _set_app_data(prepare_data(raw))
    except Exception as e:
        print(f"[startup] Failed to load from PostgreSQL: {e}")
        DATA_SOURCE_FILENAME = "—"
        return _set_app_data(pd.DataFrame())


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
def _load_model_and_metadata() -> tuple[object | None, dict | None, str | None]:
    if not _JOBLIB_OK:
        return None, None, "joblib / scikit-learn not installed."
    model_path = first_existing(MODEL_CANDIDATES)
    meta_path  = first_existing(METADATA_CANDIDATES)
    if model_path is None:
        return None, None, (
            "Model file not found. Place 'repeated_alarm_model.joblib' "
            "next to this script."
        )
    try:
        model = joblib.load(model_path)
    except Exception as e:
        return None, None, f"Failed to load model: {e}"

    meta: dict = {}
    if meta_path is not None:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    return model, meta, None


MODEL, MODEL_META, MODEL_ERROR = _load_model_and_metadata()
MODEL_THRESHOLD = (
    float(MODEL_META.get("chosen_threshold", 0.5)) if MODEL_META else 0.5
)

if MODEL_META and MODEL_META.get("features"):
    _meta_features = MODEL_META["features"]
    if _meta_features != MODEL_FEATURES:
        _missing = [f for f in _meta_features if f not in MODEL_FEATURES]
        _extra   = [f for f in MODEL_FEATURES  if f not in _meta_features]
        print("[WARN] Feature mismatch between app and loaded model metadata!")
        if _missing:
            print(f"  In model but not in app: {_missing}")
        if _extra:
            print(f"  In app but not in model: {_extra}")
    else:
        print("[OK] App feature list matches model metadata exactly.")


# ──────────────────────────────────────────────────────────────────────────────
# Monthly feature table
# ──────────────────────────────────────────────────────────────────────────────
def build_monthly_feature_table(df_clean: pd.DataFrame) -> pd.DataFrame:
    if df_clean.empty:
        return pd.DataFrame()

    # Map app internal columns to trainer column names
    col_map = {
        "_dt":     "time",
        "_route":  "route",
        "_type":   "type",
        "_pos":    "position_m",
        "_lat":    "latitude",
        "_lon":    "longitude",
        "_dist":   "route_distance_m",
        "_hour":   "hour",
        "_dow":    "dayofweek",
        "_threat": "threat_level",
        "_status": "status",
        "_fibre":  "fibre_line",
        "_occ":    "occurrence_type",
        "_resolved_dt": "resolved_on",
        "_marker": "marker_name",
    }

    sub = pd.DataFrame()
    for app_col, trainer_col in col_map.items():
        if app_col in df_clean.columns:
            sub[trainer_col] = df_clean[app_col]
        else:
            sub[trainer_col] = np.nan

    sub["time"]       = pd.to_datetime(sub["time"], errors="coerce")
    sub["position_m"] = pd.to_numeric(sub["position_m"], errors="coerce")
    sub["latitude"]   = pd.to_numeric(sub["latitude"],   errors="coerce")
    sub["longitude"]  = pd.to_numeric(sub["longitude"],  errors="coerce")
    sub["route_distance_m"] = pd.to_numeric(sub["route_distance_m"], errors="coerce")

    sub = sub.dropna(subset=["time", "route", "type", "position_m"])
    sub["route"] = sub["route"].astype(str).str.strip()
    sub["type"]  = sub["type"].astype(str).str.strip().str.title()
    sub = sub[
        (sub["route"] != "") & (sub["type"] != "")
        & (sub["route"].str.lower() != "nan")
        & (sub["type"].str.lower()  != "nan")
        & (sub["position_m"] >= 0)
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    sub["month"]      = sub["time"].dt.to_period("M").dt.to_timestamp()
    sub["month_num"]  = sub["time"].dt.month
    sub["hour"]       = pd.to_numeric(sub["hour"], errors="coerce").fillna(sub["time"].dt.hour)
    sub["dayofweek"]  = pd.to_numeric(sub["dayofweek"], errors="coerce").fillna(sub["time"].dt.dayofweek)
    sub["is_weekend"] = (sub["dayofweek"] >= 5).astype(int)
    sub["is_peak"]    = sub["hour"].between(6, 17).astype(int)

    sub["position_bin"] = (sub["position_m"] // POSITION_BIN_SIZE).astype(int)

    # Threat score
    THREAT_ORDER = {"low": 1, "medium": 2, "high": 3, "red": 3, "orange": 2, "critical": 3}
    sub["threat_score"] = sub["threat_level"].apply(
        lambda v: float(THREAT_ORDER.get(str(v).strip().lower(), 2.0)) if pd.notna(v) else 2.0
    )

    # Occurrence type
    # Keep the old feature name `accident_count` for model compatibility,
    # but count current database Incident rows. Also supports old backups that still say Accident.
    sub["is_accident"] = (
        sub["occurrence_type"].astype(str).str.strip().str.lower().isin(["incident", "accident"])
    ).astype(int)

    # Resolution time
    sub["resolved_on"] = pd.to_datetime(sub["resolved_on"], errors="coerce")
    sub["resolution_hours"] = np.where(
        sub["resolved_on"].notna() & sub["time"].notna(),
        (sub["resolved_on"] - sub["time"]).dt.total_seconds() / 3600,
        np.nan,
    )
    sub["resolution_hours"] = sub["resolution_hours"].clip(lower=0, upper=8760)
    sub["is_unresolved"] = sub["resolved_on"].isna().astype(int)

    sub["status_clean"] = sub["status"].astype(str).str.strip().str.lower().fillna("unknown")
    sub["fibre_line"]   = sub["fibre_line"].astype(str).str.strip().fillna("unknown")

    group_cols = ["route", "type", "position_bin", "month"]

    def _most_common_hour(x):
        clean = x.dropna()
        return int(clean.value_counts().index[0]) if len(clean) else 0

    def _most_common_status(x):
        clean = x.dropna()
        return str(clean.value_counts().index[0]) if len(clean) else "unknown"

    monthly = (
        sub.groupby(group_cols)
        .agg(
            alarm_count          = ("type",             "size"),
            accident_count       = ("is_accident",      "sum"),
            unresolved_count     = ("is_unresolved",    "sum"),
            avg_resolution_hrs   = ("resolution_hours", "mean"),
            max_resolution_hrs   = ("resolution_hours", "max"),
            avg_hour             = ("hour",             "mean"),
            std_hour             = ("hour",             "std"),
            most_common_hour     = ("hour",             _most_common_hour),
            peak_hour_share      = ("is_peak",          "mean"),
            weekend_share        = ("is_weekend",       "mean"),
            avg_threat_score     = ("threat_score",     "mean"),
            max_threat_score     = ("threat_score",     "max"),
            active_days          = ("time",             lambda x: x.dt.date.nunique()),
            distinct_markers     = ("marker_name",      lambda x: x.dropna().nunique()),
            dominant_status      = ("status_clean",     _most_common_status),
            dominant_fibre       = ("fibre_line",       lambda x: x.mode().iloc[0] if len(x) else "unknown"),
            avg_lat              = ("latitude",         "mean"),
            avg_lon              = ("longitude",        "mean"),
            avg_route_dist       = ("route_distance_m", "mean"),
        )
        .reset_index()
    )

    monthly["std_hour"]           = monthly["std_hour"].fillna(0)
    monthly["avg_lat"]            = monthly["avg_lat"].fillna(0)
    monthly["avg_lon"]            = monthly["avg_lon"].fillna(0)
    monthly["avg_route_dist"]     = monthly["avg_route_dist"].fillna(0)
    monthly["avg_resolution_hrs"] = monthly["avg_resolution_hrs"].fillna(
        monthly["avg_resolution_hrs"].median()
    )
    monthly["max_resolution_hrs"] = monthly["max_resolution_hrs"].fillna(0)
    monthly["alarm_share"]        = (
        monthly["alarm_count"] /
        (monthly["alarm_count"] + monthly["accident_count"]).replace(0, 1)
    )
    monthly["unresolved_share"] = (
        monthly["unresolved_count"] / monthly["alarm_count"].replace(0, 1)
    )

    monthly["month_num"] = monthly["month"].dt.month
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["month_num"] / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["month_num"] / 12)

    monthly = monthly.sort_values(["route", "type", "position_bin", "month"]).reset_index(drop=True)

    pattern_cols = ["route", "type", "position_bin"]
    grp = monthly.groupby(pattern_cols, group_keys=False)

    for lag, name in [(1, "prev_month_count"), (2, "prev2_month_count"), (3, "prev3_month_count")]:
        monthly[name] = grp["alarm_count"].shift(lag).fillna(0)

    monthly["prev_avg_resolution_hrs"] = grp["avg_resolution_hrs"].shift(1).fillna(0)

    def _roll_safe(s, win, fn):
        """Like _roll but explicitly fills NaN after aggregation.
        Handles pandas version differences where std() on window<2 returns NaN.
        """
        rolled = s.shift(1).rolling(window=win, min_periods=1)
        result = getattr(rolled, fn)()
        # std() returns NaN for single-element windows across some pandas versions
        return result.fillna(0)

    monthly["roll3_mean"]       = grp["alarm_count"].transform(lambda s: _roll_safe(s, 3, "mean"))
    monthly["roll3_std"]        = grp["alarm_count"].transform(lambda s: _roll_safe(s, 3, "std"))
    monthly["roll6_mean"]       = grp["alarm_count"].transform(lambda s: _roll_safe(s, 6, "mean"))
    monthly["roll6_std"]        = grp["alarm_count"].transform(lambda s: _roll_safe(s, 6, "std"))
    monthly["roll3_unresolved"] = grp["unresolved_share"].transform(lambda s: _roll_safe(s, 3, "mean"))
    monthly["ewma3_count"] = grp["alarm_count"].transform(
        lambda s: s.shift(1).ewm(span=3, min_periods=1, adjust=False).mean().fillna(0).infer_objects(copy=False)
    )
    monthly["momentum_diff"] = monthly["roll3_mean"] - monthly["roll6_mean"]

    monthly["past_total_count"]   = grp["alarm_count"].cumsum() - monthly["alarm_count"]
    monthly["past_months_active"] = grp.cumcount()
    monthly["past_avg_count"]     = np.where(
        monthly["past_months_active"] > 0,
        monthly["past_total_count"] / monthly["past_months_active"],
        0,
    )

    all_months   = np.sort(monthly["month"].unique())
    month_to_idx = {m: i for i, m in enumerate(all_months)}
    monthly["months_observed"]         = monthly["month"].map(month_to_idx).astype(float)
    monthly["active_ratio"]            = np.where(
        monthly["months_observed"] > 0,
        monthly["past_months_active"] / monthly["months_observed"], 0,
    )
    monthly["had_prev_month"]          = (monthly["prev_month_count"] > 0).astype(int)
    monthly["trend_diff"]              = monthly["prev_month_count"] - monthly["prev2_month_count"]
    monthly["months_since_last_seen"]  = np.where(
        monthly["past_months_active"] == 0, 999,
        np.where(monthly["prev_month_count"] > 0, 0, 1),
    )
    monthly["months_since_first_seen"] = monthly["months_observed"]

    rt_month = (
        monthly.groupby(["route", "type", "month"])["alarm_count"]
        .sum().reset_index().rename(columns={"alarm_count": "_rt_total"})
    )
    rt_month = rt_month.sort_values(["route", "type", "month"])
    rt_month["route_type_popularity"] = (
        rt_month.groupby(["route", "type"], group_keys=False)["_rt_total"]
        .shift(1).fillna(0)
    )
    monthly = monthly.merge(
        rt_month[["route", "type", "month", "route_type_popularity"]],
        on=["route", "type", "month"], how="left",
    )
    monthly["route_type_popularity"] = monthly["route_type_popularity"].fillna(0)

    lookup = monthly.set_index(["route", "type", "position_bin", "month"])["alarm_count"]

    def _neighbor_sum(route, ev_type, bin_idx, prev_m):
        total = 0.0
        for delta in (-1, 1):
            key = (route, ev_type, bin_idx + delta, prev_m)
            if key in lookup.index:
                total += float(lookup.loc[key])
        return total

    monthly["_prev_month"] = grp["month"].shift(1)
    has_prev = monthly["_prev_month"].notna()
    monthly["neighbor_bin_count"] = 0.0
    if has_prev.any():
        vals = monthly.loc[has_prev].apply(
            lambda r: _neighbor_sum(r["route"], r["type"], int(r["position_bin"]), r["_prev_month"]),
            axis=1,
        )
        monthly.loc[has_prev, "neighbor_bin_count"] = vals.values
    monthly.drop(columns=["_prev_month", "months_observed"], errors="ignore", inplace=True)

    monthly["position_range"] = monthly["position_bin"].apply(
        lambda b: _pos_range_label(int(b), POSITION_BIN_SIZE)
    )

    return monthly


def predict_next_month_table(df_clean: pd.DataFrame) -> pd.DataFrame:
    if MODEL is None:
        return pd.DataFrame()

    monthly = build_monthly_feature_table(df_clean)
    if monthly.empty:
        return pd.DataFrame()

    latest_month = monthly["month"].max()
    latest = monthly[monthly["month"] == latest_month].copy()
    if latest.empty:
        return pd.DataFrame()

    missing_features = [f for f in MODEL_FEATURES if f not in latest.columns]
    extra_features = [f for f in latest.columns if f not in MODEL_FEATURES]

    if missing_features:
        print(f"[WARN] {len(missing_features)} model features missing; filling with 0:")
        print(f"       {missing_features}")
        for col in missing_features:
            latest[col] = 0.0

    if extra_features:
        print(f"[INFO] {len(extra_features)} extra columns in latest data will be ignored by the model.")

    X_next = latest[MODEL_FEATURES].copy()

    try:
        probs = MODEL.predict_proba(X_next)[:, 1]
    except Exception as e:
        log_error("predict_next_month_table", e, "ERROR")
        return pd.DataFrame()

    latest["repeat_probability"]     = probs
    latest["pred_repeat_next_month"] = (probs >= MODEL_THRESHOLD).astype(int)
    latest["risk_level"] = pd.cut(
        latest["repeat_probability"],
        bins=[-0.01, 0.40, 0.70, 1.00],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    print(f"[predict] scored {len(latest):,} patterns → predicting month after {latest_month}")
    return latest.sort_values("repeat_probability", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Month filter helpers
# ──────────────────────────────────────────────────────────────────────────────
def available_months(df: pd.DataFrame) -> list[dict]:
    if df.empty or "_ym" not in df.columns:
        return [{"label": "All Months", "value": ALL_MONTHS_VALUE}]
    ym = df["_ym"].dropna().unique().tolist()
    ym = sorted([str(x) for x in ym if str(x).lower() not in _NA_STRINGS])
    opts = [{"label": "All Months", "value": ALL_MONTHS_VALUE}]
    for key in ym:
        try:
            dt = pd.to_datetime(key + "-01")
            opts.append({"label": dt.strftime("%B %Y"), "value": key})
        except Exception:
            continue
    return opts


def filter_by_month(df: pd.DataFrame, month_value: str | None) -> pd.DataFrame:
    if df.empty or not month_value or month_value == ALL_MONTHS_VALUE:
        return df
    if "_ym" not in df.columns:
        return df
    return df[df["_ym"] == month_value]


def month_label(month_value: str | None) -> str:
    if not month_value or month_value == ALL_MONTHS_VALUE:
        return "All Months"
    try:
        return pd.to_datetime(month_value + "-01").strftime("%B %Y")
    except Exception:
        return str(month_value)


# ──────────────────────────────────────────────────────────────────────────────
# Figure helpers
# ──────────────────────────────────────────────────────────────────────────────
def empty_fig(text: str = "No data available", height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=f"<span style='font-size:14px;color:{COLORS['text_4']}'>{text}</span>",
        x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
    )
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def polish(fig: go.Figure, height: int = 300, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="'Inter','DM Sans','Segoe UI',sans-serif", size=12,
                  color=COLORS["text_2"]),
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.28,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=COLORS["border"], borderwidth=1,
            font=dict(size=11, color=COLORS["text_3"]),
        ),
        hoverlabel=dict(
            bgcolor="white", font_size=12,
            font_family="'DM Sans',sans-serif",
            bordercolor=COLORS["border_2"],
        ),
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        linecolor=COLORS["border"], tickfont=dict(size=11),
    )
    fig.update_yaxes(
        gridcolor="#f1f5f9", zeroline=False,
        linecolor=COLORS["border"], tickfont=dict(size=11),
    )
    return fig


def count_col(df: pd.DataFrame, col: str, label: str,
              top: int | None = None) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[label, "Count"])
    out = df[col].dropna().value_counts().reset_index()
    out.columns = [label, "Count"]
    return out.head(top) if top else out


def make_clean_donut(
    df_count: pd.DataFrame,
    label_col: str,
    value_col: str = "Count",
    colors: list[str] | None = None,
    height: int = 300,
    min_label_pct: float = 1.0,
    legend_pct_decimals: int = 3,
) -> go.Figure:
    if (
        df_count.empty
        or value_col not in df_count.columns
        or df_count[value_col].sum() == 0
    ):
        return empty_fig("No data available", height)

    chart_df = df_count.copy()
    total = chart_df[value_col].sum()
    chart_df["pct"] = chart_df[value_col] / total * 100
    chart_df["inside_label"] = chart_df["pct"].apply(
        lambda x: f"{x:.3f}%" if x >= min_label_pct else ""
    )
    chart_df["legend_label"] = chart_df.apply(
        lambda r: (
            f"{r[label_col]} — {int(r[value_col]):,} "
            f"({r['pct']:.{legend_pct_decimals}f}%)"
        ),
        axis=1,
    )

    fig = go.Figure(go.Pie(
        labels=chart_df["legend_label"],
        values=chart_df[value_col],
        hole=0.6,
        marker=dict(
            colors=colors if colors else PALETTE,
            line=dict(color="white", width=2),
        ),
        text=chart_df["inside_label"],
        textinfo="text",
        textposition="inside",
        insidetextfont=dict(color="white", size=12, family="DM Sans"),
        hovertemplate="<b>%{label}</b><extra></extra>",
        sort=False,
    ))
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=5, b=5),
        font=dict(size=11),
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=11)),
    )
    return fig


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "_dt" not in df.columns or df["_dt"].notna().sum() == 0:
        return pd.DataFrame(columns=["period", "Count"])
    tmp = df.dropna(subset=["_dt"]).copy()
    use_ym_ts = (
        "_ym_ts" in tmp.columns
        and tmp["_ym_ts"].notna().any()
        and pd.api.types.is_datetime64_any_dtype(tmp["_ym_ts"])
    )
    if use_ym_ts:
        tmp["_period_ts"] = tmp["_ym_ts"]
    else:
        tmp["_period_ts"] = tmp["_dt"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        tmp.groupby("_period_ts").size().reset_index(name="Count")
        .sort_values("_period_ts")
    )
    monthly["period"] = monthly["_period_ts"].dt.strftime("%b %Y")
    return monthly[["period", "Count", "_period_ts"]]


def time_of_day_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "_dt" not in df.columns or df["_dt"].notna().sum() == 0:
        return pd.DataFrame(columns=["Time Period", "Count"])
    tmp = df.dropna(subset=["_dt"]).copy()
    tmp["_h"] = tmp["_dt"].dt.hour
    conditions = [
        tmp["_h"].between(0, 5),
        tmp["_h"].between(6, 11),
        tmp["_h"].between(12, 17),
        tmp["_h"].between(18, 23),
    ]
    labels = ["Midnight", "Morning", "Afternoon", "Night"]
    tmp["Time Period"] = np.select(conditions, labels, default="Unknown")
    order = pd.CategoricalDtype(
        categories=["Midnight", "Morning", "Afternoon", "Night"], ordered=True,
    )
    tmp["Time Period"] = tmp["Time Period"].astype(order)
    result = (
        tmp.groupby("Time Period", observed=False).size().reset_index(name="Count")
    )
    return result[result["Count"] > 0]


# ──────────────────────────────────────────────────────────────────────────────
# Position range computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_position_ranges(
    df: pd.DataFrame,
    bin_size: int = 5000,
    max_position: Optional[float] = None,
) -> pd.DataFrame:
    if df.empty or "_pos" not in df.columns:
        return pd.DataFrame()

    pos_df = df.copy()
    if "_event_class" in pos_df.columns:
        pos_df = pos_df[
            pos_df["_event_class"].astype(str).str.strip().str.lower().eq("alarm")
        ].copy()

    pos_df = pos_df.dropna(subset=["_pos"]).copy()
    if pos_df.empty:
        return pd.DataFrame()

    pos_df["_pos"] = pd.to_numeric(pos_df["_pos"], errors="coerce")
    pos_df = pos_df.dropna(subset=["_pos"])
    pos_df = pos_df[pos_df["_pos"] >= 0].copy()
    if pos_df.empty:
        return pd.DataFrame()

    bin_size = max(100, int(bin_size)) if bin_size else 5000
    upper = (
        float(max_position)
        if max_position is not None and float(max_position) > 0
        else float(pos_df["_pos"].max())
    )
    upper_bound = (int(upper) // bin_size + 1) * bin_size
    edges = np.arange(0, upper_bound + bin_size, bin_size)

    pos_df["bin_idx"] = pd.cut(
        pos_df["_pos"],
        bins=edges, right=False,
        labels=list(range(len(edges) - 1)),
        include_lowest=True,
    )
    pos_df = pos_df.dropna(subset=["bin_idx"]).copy()
    pos_df["bin_idx"] = pos_df["bin_idx"].astype(int)

    def _top_route(s: pd.Series) -> str:
        s = s.dropna().astype(str).str.strip()
        s = s[(s != "") & (s.str.lower() != "nan")]
        if s.empty:
            return "—"
        vc = s.value_counts()
        return f"{vc.index[0]} ({vc.iloc[0]:,})"

    def _high_threats(s: pd.Series) -> int:
        return int(s.astype(str).str.strip().str.lower().eq("red").sum())

    grouped = (
        pos_df.groupby("bin_idx", observed=False)
        .agg(
            alarm_count=("_pos", "size"),
            avg_lat=("_lat", "mean"),
            avg_lon=("_lon", "mean"),
            top_route=("_route", _top_route),
            high_threats=("_threat_key", _high_threats),
        )
        .reset_index()
    )

    grouped = grouped[grouped["alarm_count"] > 0].copy()
    if grouped.empty:
        return pd.DataFrame()

    grouped["Position Range"] = grouped["bin_idx"].apply(
        lambda idx: _pos_range_label(int(idx), bin_size)
    )
    grouped["Range Start"] = grouped["bin_idx"] * bin_size
    grouped = grouped.rename(columns={
        "alarm_count":  "Alarm Count",
        "avg_lat":      "Avg Latitude",
        "avg_lon":      "Avg Longitude",
        "top_route":    "Top Route",
        "high_threats": "Red Threats",
    })
    grouped["Avg Latitude"]  = grouped["Avg Latitude"].round(5)
    grouped["Avg Longitude"] = grouped["Avg Longitude"].round(5)

    return grouped[[
        "Position Range", "Alarm Count", "Red Threats",
        "Avg Latitude", "Avg Longitude", "Top Route", "Range Start",
    ]].sort_values("Range Start").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Report builder
# ──────────────────────────────────────────────────────────────────────────────
REPORT_SECTIONS = [
    {"label": "Executive Summary KPIs",        "value": "summary",         "icon": "📊", "desc": "Total records, alarms, incidents, red threats, top route & type"},
    {"label": "Events Over Time",               "value": "events_over_time","icon": "📈", "desc": "Monthly or daily event count trend line"},
    {"label": "Event Type Distribution",        "value": "event_types",     "icon": "🗂️", "desc": "Horizontal bar chart of the top 10 event types"},
    {"label": "Status Overview",                "value": "status",          "icon": "✅", "desc": "Pie chart of resolved vs pending alarms"},
    {"label": "Threat Level Distribution",      "value": "threat",          "icon": "⚠️", "desc": "Bar chart of threat level frequencies"},
    {"label": "Day-of-Week Pattern",            "value": "day_of_week",     "icon": "📅", "desc": "Which days drive the most alarm volume"},
    {"label": "Top Routes",                     "value": "routes",          "icon": "🛤️", "desc": "Horizontal bar chart of the top 10 busiest routes"},
    {"label": "Position Range Analysis",        "value": "position",        "icon": "📍", "desc": "Alarm density across 5 km position bins"},
    {"label": "Threat × Route Matrix",          "value": "threat_matrix",   "icon": "🔥", "desc": "Heat table crossing threat levels with top routes"},
    {"label": "Resolution Performance",         "value": "resolution",      "icon": "⏱️", "desc": "Median resolution time per event type"},
]
ALL_SECTION_VALUES = [s["value"] for s in REPORT_SECTIONS]


def _report_value_counts(df: pd.DataFrame, col: str, limit: int = 10) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=["Label", "Count"])
    out = (
        df[col].dropna().astype(str).str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .dropna().value_counts().head(limit).reset_index()
    )
    if out.empty:
        return pd.DataFrame(columns=["Label", "Count"])
    out.columns = ["Label", "Count"]
    return out


def _report_event_split(df: pd.DataFrame) -> tuple[int, int, int]:
    total = int(len(df))
    if df.empty or "_event_class" not in df.columns:
        return total, total, 0
    cls = df["_event_class"].astype(str).str.strip().str.lower()
    incidents = int(cls.eq("incident").sum())
    return total, total - incidents, incidents


def _report_plot_bytes(df: pd.DataFrame, section: str, bin_size: int = 5000) -> bytes | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except Exception:
        return None

    if df.empty:
        return None

    BRAND_BLUE   = COLORS["blue"]
    BRAND_TEAL   = COLORS["teal"]
    BRAND_RED    = COLORS["red"]
    BRAND_AMBER  = COLORS["amber_dark"]
    BRAND_GREEN  = COLORS["green"]
    BRAND_INDIGO = COLORS["indigo"]
    BRAND_VIOLET = COLORS["violet"]
    BG           = "#ffffff"
    GRID_COLOR   = "#f1f5f9"
    TEXT_COLOR   = COLORS["text"]
    LABEL_COLOR  = "#5f7893"

    def _style_ax(ax, title):
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12, color=TEXT_COLOR)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#d8e8f2")
        ax.spines["bottom"].set_color("#d8e8f2")
        ax.tick_params(colors=LABEL_COLOR, labelsize=8)
        ax.yaxis.label.set_color(LABEL_COLOR)
        ax.xaxis.label.set_color(LABEL_COLOR)

    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    try:
        if section == "events_over_time":
            if "_dt" not in df.columns or df["_dt"].notna().sum() == 0:
                return None
            tmp = df.dropna(subset=["_dt"]).copy()
            if "_ym" in tmp.columns and tmp["_ym"].nunique() == 1:
                trend = tmp.groupby(tmp["_dt"].dt.date).size().reset_index(name="Count")
                trend.columns = ["Period", "Count"]
                x_vals = trend["Period"].astype(str).tolist()
            else:
                trend = monthly_counts(tmp)
                if trend.empty:
                    return None
                x_vals = trend["period"].tolist()
            y_vals = trend["Count"].tolist()
            ax.fill_between(range(len(x_vals)), y_vals, alpha=0.18, color=BRAND_BLUE)
            ax.plot(range(len(x_vals)), y_vals, marker="o", linewidth=2.2,
                    color=BRAND_BLUE, markersize=5, markerfacecolor="white",
                    markeredgewidth=2)
            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels(x_vals, rotation=35, ha="right", fontsize=7.5)
            ax.set_ylabel("Events", fontsize=9)
            ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
            ax.set_axisbelow(True)
            _style_ax(ax, "Events Over Time")

        elif section == "event_types":
            counts = _report_value_counts(df, "_type", 10)
            if counts.empty:
                return None
            counts = counts.sort_values("Count")
            bar_colors = [BRAND_BLUE if i == len(counts) - 1 else BRAND_TEAL
                          for i in range(len(counts))]
            bars = ax.barh(counts["Label"], counts["Count"], color=bar_colors,
                           height=0.65, edgecolor="none")
            for bar in bars:
                w = bar.get_width()
                ax.text(w + max(counts["Count"]) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{int(w):,}", va="center", fontsize=7.5, color=TEXT_COLOR)
            ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_xlabel("Records", fontsize=9)
            _style_ax(ax, "Top Event Types")

        elif section == "status":
            counts = _report_value_counts(df, "_status", 6)
            if counts.empty:
                return None
            status_colors = []
            for lbl in counts["Label"]:
                if "resolved" in lbl.lower():
                    status_colors.append(BRAND_GREEN)
                elif "pending" in lbl.lower() or "open" in lbl.lower() or "unresolved" in lbl.lower():
                    status_colors.append(BRAND_RED)
                else:
                    status_colors.append(BRAND_AMBER)
            wedges, texts, autotexts = ax.pie(
                counts["Count"], labels=counts["Label"],
                autopct="%1.1f%%", startangle=90, colors=status_colors,
                textprops={"fontsize": 8, "color": TEXT_COLOR},
                wedgeprops={"edgecolor": "white", "linewidth": 2},
            )
            for at in autotexts:
                at.set_color("white")
                at.set_fontsize(8)
                at.set_fontweight("bold")
            ax.set_title("Status Overview", fontsize=12, fontweight="bold", pad=12, color=TEXT_COLOR)
            ax.axis("equal")

        elif section == "threat":
            counts = _report_value_counts(df, "_threat", 8)
            if counts.empty:
                return None
            color_map_t = {
                "red": BRAND_RED, "high": BRAND_RED, "critical": BRAND_RED,
                "amber": BRAND_AMBER, "medium": BRAND_AMBER, "yellow": BRAND_AMBER,
                "green": BRAND_GREEN, "low": BRAND_GREEN,
            }
            bar_colors = [
                color_map_t.get(lbl.lower(), BRAND_TEAL)
                for lbl in counts["Label"]
            ]
            bars = ax.bar(counts["Label"], counts["Count"], color=bar_colors,
                          edgecolor="none", width=0.6)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + counts["Count"].max() * 0.01,
                        f"{int(h):,}", ha="center", fontsize=8, color=TEXT_COLOR)
            ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", rotation=20)
            ax.set_ylabel("Records", fontsize=9)
            _style_ax(ax, "Threat Level Distribution")

        elif section == "day_of_week":
            if "_dow" not in df.columns:
                return None
            tmp = df.dropna(subset=["_dow"]).copy()
            if tmp.empty:
                return None
            tmp["Day"] = tmp["_dow"].astype(int).map(dict(enumerate(DAYS_OF_WEEK)))
            counts = tmp["Day"].value_counts().reindex(DAYS_OF_WEEK).fillna(0)
            peak_day = counts.idxmax()
            bar_colors = [BRAND_INDIGO if d == peak_day else COLORS["blue_light"] for d in counts.index]
            bars = ax.bar(counts.index, counts.values, color=bar_colors,
                          edgecolor="none", width=0.65)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + counts.max() * 0.01,
                        f"{int(h):,}", ha="center", fontsize=7.5, color=TEXT_COLOR)
            ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_ylabel("Records", fontsize=9)
            _style_ax(ax, "Day-of-Week Pattern")

        elif section == "routes":
            counts = _report_value_counts(df, "_route", 10)
            if counts.empty:
                return None
            counts = counts.sort_values("Count")
            bar_colors = [
                BRAND_INDIGO if i == len(counts) - 1 else BRAND_VIOLET
                for i in range(len(counts))
            ]
            bars = ax.barh(counts["Label"], counts["Count"],
                           color=bar_colors, height=0.65, edgecolor="none")
            for bar in bars:
                w = bar.get_width()
                ax.text(w + counts["Count"].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{int(w):,}", va="center", fontsize=7.5, color=TEXT_COLOR)
            ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_xlabel("Records", fontsize=9)
            _style_ax(ax, "Top Routes")

        elif section == "position":
            pos = compute_position_ranges(df, bin_size=bin_size)
            if pos.empty:
                return None
            top = pos.sort_values("Alarm Count", ascending=False).head(12).sort_values("Alarm Count")
            ax.barh(top["Position Range"], top["Alarm Count"],
                    color=BRAND_TEAL, height=0.65, edgecolor="none", label="Alarms")
            if (top["Red Threats"] > 0).any():
                ax.barh(top["Position Range"], top["Red Threats"],
                        color=BRAND_RED, height=0.65, edgecolor="none",
                        alpha=0.85, label="Red Threats")
                ax.legend(loc="lower right", fontsize=8)
            ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_xlabel("Alarms", fontsize=9)
            _style_ax(ax, "Top Position Ranges by Alarm Count")

        elif section == "threat_matrix":
            if "_threat" not in df.columns or "_route" not in df.columns:
                return None
            sub = df.dropna(subset=["_threat", "_route"]).copy()
            if sub.empty:
                return None
            top_routes  = sub["_route"].value_counts().head(8).index.tolist()
            top_threats = sub["_threat"].value_counts().head(6).index.tolist()
            sub = sub[sub["_route"].isin(top_routes) & sub["_threat"].isin(top_threats)]
            if sub.empty:
                return None
            pivot = (
                sub.groupby(["_threat", "_route"]).size()
                .unstack(fill_value=0)
                .reindex(index=top_threats, columns=top_routes, fill_value=0)
            )
            cmap = LinearSegmentedColormap.from_list(
                "das", ["#F8FAFC", "#DBEAFE", "#60A5FA", "#1E3A8A"]
            )
            plt.close(fig)
            fig, ax = plt.subplots(
                figsize=(max(7.4, len(top_routes) * 0.9 + 2),
                         max(3.6, len(top_threats) * 0.6 + 1.2)),
                dpi=150,
            )
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([c[:25] for c in pivot.columns],
                               rotation=35, ha="right", fontsize=7.5)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    ax.text(j, i, str(int(val)), ha="center", va="center",
                            fontsize=8, color="white" if val > pivot.values.max() * 0.5 else TEXT_COLOR,
                            fontweight="bold")
            plt.colorbar(im, ax=ax, shrink=0.8, label="Count")
            _style_ax(ax, "Threat Level × Route Matrix")

        elif section == "resolution":
            if "_res_minutes" not in df.columns or "_type" not in df.columns:
                return None
            tmp = df.dropna(subset=["_res_minutes", "_type"]).copy()
            tmp["_type"] = tmp["_type"].astype(str).str.strip().str.title()
            tmp["_res_minutes"] = pd.to_numeric(tmp["_res_minutes"], errors="coerce")
            tmp = tmp[tmp["_res_minutes"] >= 0].copy()
            if tmp.empty:
                return None
            med = (
                tmp.groupby("_type")["_res_minutes"]
                .median().sort_values(ascending=False).head(10)
            )
            med_hours = med / 60
            bars = ax.barh(
                [t[:30] for t in med.index],
                med_hours.values,
                color=BRAND_TEAL, height=0.65, edgecolor="none",
            )
            for bar in bars:
                w = bar.get_width()
                ax.text(w + med_hours.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{w:.1f}h", va="center", fontsize=7.5, color=TEXT_COLOR)
            ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_xlabel("Median Resolution (hours)", fontsize=9)
            _style_ax(ax, "Median Resolution Time by Event Type")

        else:
            return None

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        return None


def build_report_pdf(
    df: pd.DataFrame,
    sections: list[str],
    period_label: str,
    bin_size: int = 5000,
    data_source: str = "-",
) -> bytes:
    from io import BytesIO
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, HRFlowable, KeepTogether,
    )

    buffer = BytesIO()
    W, H = A4

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title=f"DAS Intelligence Report — {period_label}",
    )

    styles = getSampleStyleSheet()
    brand_dark  = colors.HexColor("#10233F")
    brand_blue  = colors.HexColor("#2F62B3")
    brand_teal  = colors.HexColor("#4AB6E8")
    brand_light = colors.HexColor("#EDF6FB")
    text_2      = colors.HexColor("#294564")
    text_3      = colors.HexColor("#5f7893")
    border_c    = colors.HexColor("#D8E8F2")
    red_c       = colors.HexColor("#D93D4F")
    green_c     = colors.HexColor("#2E7228")
    amber_c     = colors.HexColor("#C8A800")
    surface_2   = colors.HexColor("#F7FBFF")

    styles.add(ParagraphStyle(
        name="DAS_Title", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=22, leading=26,
        textColor=brand_dark, spaceAfter=2, alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        name="DAS_Sub", parent=styles["Normal"],
        fontSize=9.5, leading=13, textColor=text_3, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="DAS_Section", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=13, leading=16,
        textColor=brand_dark, spaceBefore=14, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="DAS_Small", parent=styles["Normal"],
        fontSize=8.2, leading=11, textColor=text_2,
    ))
    styles.add(ParagraphStyle(
        name="DAS_Muted", parent=styles["Normal"],
        fontSize=7.5, leading=10, textColor=text_3,
    ))
    styles.add(ParagraphStyle(
        name="DAS_Footer", parent=styles["Normal"],
        fontSize=7.5, leading=9, textColor=colors.HexColor("#90A9BD"),
        alignment=TA_CENTER,
    ))

    generated_at = datetime.now().strftime("%d %b %Y, %I:%M %p")

    def _footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(border_c)
        canvas.line(16 * mm, 13 * mm, W - 16 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#90A9BD"))
        canvas.drawString(16 * mm, 8 * mm, f"DAS Intelligence Report — {period_label}")
        canvas.drawRightString(W - 16 * mm, 8 * mm, f"Page {document.page}")
        canvas.restoreState()

    def _header(canvas, document):
        _footer(canvas, document)
        if document.page > 1:
            canvas.saveState()
            canvas.setFillColor(brand_dark)
            canvas.setFont("Helvetica-Bold", 8)
            canvas.drawString(16 * mm, H - 10 * mm, "DAS Intelligence")
            canvas.drawRightString(W - 16 * mm, H - 10 * mm, period_label)
            canvas.setStrokeColor(border_c)
            canvas.line(16 * mm, H - 12 * mm, W - 16 * mm, H - 12 * mm)
            canvas.restoreState()

    story = []
    total, alarms, incidents = _report_event_split(df)
    red_threats = 0
    if "_threat_key" in df.columns:
        red_threats = int(df["_threat_key"].astype(str).str.lower().eq("red").sum())

    top_route = "—"
    rc = _report_value_counts(df, "_route", 1)
    if not rc.empty:
        top_route = f"{rc.iloc[0]['Label']} ({int(rc.iloc[0]['Count']):,})"

    top_type = "—"
    tc = _report_value_counts(df, "_type", 1)
    if not tc.empty:
        top_type = f"{tc.iloc[0]['Label']} ({int(tc.iloc[0]['Count']):,})"

    story.append(Paragraph("DAS Intelligence", styles["DAS_Title"]))
    story.append(Paragraph(
        f"<font color='#5f7893'>Fibre Monitoring Report &nbsp;·&nbsp; {period_label}</font>",
        styles["DAS_Sub"],
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=brand_blue,
                            spaceAfter=10))

    meta_data = [
        ["Reporting Period", period_label,     "Generated",      generated_at],
        ["Data Source",      data_source or "—","Position Bin",  f"{bin_size:,} m"],
        ["Total Records",    f"{total:,}",      "Sections",      str(len(sections))],
    ]
    meta_tbl = Table(meta_data, colWidths=[34*mm, 65*mm, 30*mm, 47*mm])
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), surface_2),
        ("FONTNAME",     (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",     (2, 0), (2, -1),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8.2),
        ("TEXTCOLOR",    (0, 0), (-1, -1), text_2),
        ("GRID",         (0, 0), (-1, -1), 0.3, border_c),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 10))

    if "summary" in sections:
        story.append(Paragraph("Executive Summary", styles["DAS_Section"]))

        kpi_data = [
            ["Total Records",  "Alarms",       "Incidents",  "Red Threats"],
            [f"{total:,}",     f"{alarms:,}",  f"{incidents:,}", f"{red_threats:,}"],
        ]
        kpi_tbl = Table(kpi_data, colWidths=[43*mm, 43*mm, 43*mm, 43*mm])
        kpi_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0),  brand_dark),
            ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, 0),  8),
            ("BACKGROUND",   (0, 1), (-1, 1),  brand_light),
            ("FONTNAME",     (0, 1), (-1, 1),  "Helvetica-Bold"),
            ("FONTSIZE",     (0, 1), (-1, 1),  20),
            ("TEXTCOLOR",    (0, 1), (-1, 1),  brand_dark),
            ("BACKGROUND",   (3, 1), (3, 1),   colors.HexColor("#F9D7DC")),
            ("TEXTCOLOR",    (3, 1), (3, 1),   red_c),
            ("GRID",         (0, 0), (-1, -1), 0.35, border_c),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",   (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 9),
        ]))
        story.append(kpi_tbl)
        story.append(Spacer(1, 6))

        detail_data = [
            ["Top Route", top_route, "Top Event Type", top_type],
        ]
        detail_tbl = Table(detail_data, colWidths=[28*mm, 58*mm, 35*mm, 51*mm])
        detail_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), surface_2),
            ("FONTNAME",     (0, 0), (0, -1),  "Helvetica-Bold"),
            ("FONTNAME",     (2, 0), (2, -1),  "Helvetica-Bold"),
            ("TEXTCOLOR",    (0, 0), (-1, -1), text_2),
            ("FONTSIZE",     (0, 0), (-1, -1), 8.2),
            ("GRID",         (0, 0), (-1, -1), 0.3, border_c),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",   (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
            ("LEFTPADDING",  (0, 0), (-1, -1), 7),
        ]))
        story.append(detail_tbl)
        story.append(Spacer(1, 8))

    chart_meta = {
        "events_over_time":  "Events Over Time",
        "event_types":       "Event Type Distribution",
        "status":            "Status Overview",
        "threat":            "Threat Level Distribution",
        "day_of_week":       "Day-of-Week Pattern",
        "routes":            "Top Routes",
        "position":          "Position Range Analysis",
        "threat_matrix":     "Threat Level × Route Matrix",
        "resolution":        "Resolution Performance",
    }

    for section in sections:
        if section == "summary":
            continue
        title = chart_meta.get(section)
        if not title:
            continue

        section_elements = [Paragraph(title, styles["DAS_Section"])]

        img_bytes = _report_plot_bytes(df, section, bin_size=bin_size)
        if img_bytes:
            from io import BytesIO as _BIO
            img_h = 72 * mm if section == "threat_matrix" else 68 * mm
            section_elements.append(Image(_BIO(img_bytes), width=168*mm, height=img_h))
        else:
            section_elements.append(
                Paragraph("No chart data available for this section.", styles["DAS_Muted"])
            )

        if section == "position":
            pos = compute_position_ranges(df, bin_size=bin_size)
            if not pos.empty:
                section_elements.append(Spacer(1, 5))
                top15 = pos.sort_values("Alarm Count", ascending=False).head(15)
                headers = ["Position Range", "Alarm Count", "Red Threats", "Top Route"]
                rows = [headers] + [
                    [str(r["Position Range"]), f"{int(r['Alarm Count']):,}",
                     f"{int(r['Red Threats']):,}", str(r["Top Route"])]
                    for _, r in top15.iterrows()
                ]
                pos_tbl = Table(rows, colWidths=[44*mm, 28*mm, 28*mm, 72*mm], repeatRows=1)
                pos_tbl.setStyle(TableStyle([
                    ("BACKGROUND",   (0, 0), (-1, 0),  brand_dark),
                    ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
                    ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
                    ("FONTSIZE",     (0, 0), (-1, -1), 7.4),
                    ("GRID",         (0, 0), (-1, -1), 0.25, border_c),
                    ("BACKGROUND",   (0, 1), (-1, -1), surface_2),
                    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [surface_2, colors.white]),
                    ("VALIGN",       (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING",  (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING",   (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
                ]))
                section_elements.append(pos_tbl)

        elif section in ("event_types", "routes"):
            col_key = "_type" if section == "event_types" else "_route"
            cnt = _report_value_counts(df, col_key, 10)
            if not cnt.empty:
                section_elements.append(Spacer(1, 5))
                rows = [["Event Type" if section == "event_types" else "Route", "Count", "% of Total"]]
                total_cnt = cnt["Count"].sum()
                for _, r in cnt.iterrows():
                    pct = r["Count"] / total_cnt * 100 if total_cnt else 0
                    rows.append([str(r["Label"]), f"{int(r['Count']):,}", f"{pct:.1f}%"])
                cnt_tbl = Table(rows, colWidths=[90*mm, 30*mm, 52*mm], repeatRows=1)
                cnt_tbl.setStyle(TableStyle([
                    ("BACKGROUND",   (0, 0), (-1, 0),  brand_dark),
                    ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
                    ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
                    ("FONTSIZE",     (0, 0), (-1, -1), 7.8),
                    ("GRID",         (0, 0), (-1, -1), 0.25, border_c),
                    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [surface_2, colors.white]),
                    ("LEFTPADDING",  (0, 0), (-1, -1), 6),
                    ("TOPPADDING",   (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
                ]))
                section_elements.append(cnt_tbl)

        section_elements.append(Spacer(1, 8))
        story.append(KeepTogether(section_elements[:3]))
        for el in section_elements[3:]:
            story.append(el)

    doc.build(story, onFirstPage=_header, onLaterPages=_header)
    buffer.seek(0)
    return buffer.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# CSS — polished v6 with map expand panel
# ──────────────────────────────────────────────────────────────────────────────
STYLES = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #f5f8ff;
  --surface: #ffffff;
  --surface-2: #f8fafc;
  --surface-3: #f1f5f9;
  --border: #e2e8f0;
  --border-2: #cbd5e1;
  --text: #0f172a;
  --text-2: #334155;
  --text-3: #64748b;
  --text-4: #94a3b8;
  --blue: #2F62B3;
  --blue-light: #D7ECF8;
  --indigo: #334F96;
  --violet: #6C63C9;
  --cyan: #1F9AD1;
  --teal: #4AB6E8;
  --green: #2E7228;
  --green-light: #DDECCB;
  --sea: #2E9E8F;
  --amber: #C8A800;
  --amber-light: #FEF3C7;
  --red: #D93D4F;
  --red-light: #F9D7DC;
  --pink: #9C0E43;
  --radius: 14px;
  --radius-sm: 9px;
  --shadow-sm: 0 1px 2px rgba(15,23,42,.04);
  --shadow: 0 1px 3px rgba(15,23,42,.06), 0 4px 12px rgba(15,23,42,.04);
  --shadow-lg: 0 4px 6px rgba(15,23,42,.05), 0 12px 24px rgba(15,23,42,.07);
}

body {
  font-family: 'DM Sans', 'Inter', sans-serif;
  background: linear-gradient(135deg, #f3f8fc 0%, #e7f3fb 50%, #f7fbff 100%);
  background-attachment: fixed;
  color: var(--text);
  -webkit-font-smoothing: antialiased;
}

/* ── Shell ────────────────────────────────────── */
.shell { display: flex; min-height: 100vh; }
.shell.closed .sidebar { display: none; }
.shell.closed .toggle  { left: 16px; }
.shell.open  .toggle   { left: 292px; }

/* ── Sidebar ──────────────────────────────────── */
.sidebar {
  width: 280px; min-height: 100vh;
  padding: 18px 16px;
  background: linear-gradient(180deg, #081126 0%, #111843 48%, #1b1750 100%);
  color: #e5e7eb;
  display: flex; flex-direction: column;
  position: sticky; top: 0; overflow-y: auto;
  z-index: 100;
  border-right: 1px solid rgba(255,255,255,.08);
  box-shadow: 10px 0 35px rgba(15,23,42,.16);
}
.sidebar::before {
  content: ""; position: absolute; inset: 10px;
  border-radius: 28px;
  border: 1px solid rgba(255,255,255,.07);
  pointer-events: none;
}

.brand {
  position: relative; display: flex; align-items: center; gap: 14px;
  padding: 18px 12px 22px; margin-bottom: 18px;
  border-bottom: 1px solid rgba(255,255,255,.09);
}
.brand-logo {
  width: 52px; height: 52px; flex-shrink: 0; border-radius: 16px;
  background: radial-gradient(circle at 35% 25%, #9da7ff, transparent 35%),
              linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
  box-shadow: 0 12px 28px rgba(79,70,229,.42), inset 0 1px 0 rgba(255,255,255,.25);
}
.brand-title   { font-size: 17px; font-weight: 800; color: #ffffff; letter-spacing: -.2px; line-height: 1.1; }
.brand-subtitle{ font-size: 12px; color: #9ca3af; margin-top: 4px; font-weight: 500; }

.nav-section { padding: 8px 12px 12px; font-size: 10.5px; font-weight: 800; color: #7f8aa8; text-transform: uppercase; letter-spacing: 1.8px; }
.nav-list    { display: flex; flex-direction: column; gap: 6px; padding: 0 2px; }

.nav-item {
  position: relative; display: flex; align-items: center; gap: 13px;
  min-height: 44px; padding: 10px 12px;
  border-radius: 13px; border: 1px solid transparent;
  background: transparent; color: #a8b3cf;
  font: 700 13.5px 'DM Sans', sans-serif;
  cursor: pointer; width: 100%; text-align: left;
  transition: all .2s ease;
}
.nav-item:hover   { background: rgba(255,255,255,.07); color: #ffffff; border-color: rgba(255,255,255,.07); transform: translateX(3px); }
.nav-item.active  { background: linear-gradient(90deg,rgba(37,99,235,.95),rgba(124,58,237,.92)); color:#fff; border-color:rgba(255,255,255,.18); box-shadow:0 12px 26px rgba(79,70,229,.32),inset 0 1px 0 rgba(255,255,255,.22); }
.nav-item.active::before { content:""; position:absolute; left:-3px; top:10px; width:4px; height:24px; border-radius:999px; background:#38bdf8; box-shadow:0 0 16px rgba(56,189,248,.9); }

.nav-icon { width:32px;height:32px;border-radius:10px;display:inline-flex;align-items:center;justify-content:center;flex-shrink:0;font-size:16px;color:#dbeafe;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.07); }
.nav-item.active .nav-icon { background:rgba(255,255,255,.18);color:#fff;border-color:rgba(255,255,255,.18); }

.nav-spacer { flex: 1; }
.nav-footer { position:relative;margin-top:20px;padding:14px;border-radius:16px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.09); }
.footer-row { display:flex;align-items:center;gap:12px; }
.footer-icon { width:40px;height:40px;border-radius:13px;display:flex;align-items:center;justify-content:center;color:#dbeafe;font-size:19px;background:rgba(99,102,241,.18);border:1px solid rgba(255,255,255,.14); }
.footer-label { font-size:10.5px;color:#8b95b5;font-weight:700; }
.footer-value { margin-top:2px;font-size:13px;color:#fff;font-weight:800; }
.footer-version { margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,.08);font-size:11px;color:#9ca3af;line-height:1.6; }

.toggle { position:fixed;top:18px;width:34px;height:34px;background:var(--surface);border:1px solid var(--border);border-radius:8px;font-size:14px;cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:var(--shadow);transition:all .2s;z-index:200;color:var(--text-2); }
.toggle:hover { background:var(--blue-light);border-color:var(--blue);color:var(--blue); }

.content { flex:1;min-width:0; }
.pages { padding:28px 32px 56px;max-width:1600px;margin:0 auto; }

.page { animation:fade .28s ease; }
.page.hidden { display:none !important; }
@keyframes fade { from { opacity:0;transform:translateY(8px); } to { opacity:1;transform:none; } }

/* ── Header ───────────────────────────────────── */
.header { display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:24px; }
.header-info h1 { font-size:24px;font-weight:700;color:var(--text);letter-spacing:-.3px; }
.header-info p  { font-size:13px;color:var(--text-3);margin-top:4px; }
.header-chip { display:flex;align-items:center;gap:7px;padding:6px 13px;border-radius:999px;background:linear-gradient(135deg,#edf6fb,#d7ecf8);border:1px solid #b7daef;font-size:11px;font-weight:600;color:var(--blue);white-space:nowrap; }
.live-dot { width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2s infinite;box-shadow:0 0 0 0 rgba(46,114,40,.7); }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(46,114,40,.7);}70%{box-shadow:0 0 0 6px rgba(46,114,40,0);}100%{box-shadow:0 0 0 0 rgba(46,114,40,0);} }

/* ── KPI ──────────────────────────────────────── */
.kpi-grid { display:grid;grid-template-columns:repeat(auto-fit,minmax(195px,1fr));gap:14px;margin-bottom:18px; }
.kpi { background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px 18px;display:flex;justify-content:space-between;align-items:flex-start;box-shadow:var(--shadow);transition:all .2s;position:relative;overflow:hidden; }
.kpi::before { content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--blue),var(--violet));opacity:.85; }
.kpi.green::before  { background:linear-gradient(90deg,var(--green),var(--sea)); }
.kpi.red::before    { background:linear-gradient(90deg,var(--red),var(--pink)); }
.kpi.amber::before  { background:linear-gradient(90deg,var(--amber),#f97316); }
.kpi.indigo::before { background:linear-gradient(90deg,var(--indigo),var(--violet)); }
.kpi.cyan::before   { background:linear-gradient(90deg,var(--cyan),var(--teal)); }
.kpi:hover { transform:translateY(-3px);box-shadow:var(--shadow-lg);border-color:var(--border-2); }
.kpi-info { display:flex;flex-direction:column;gap:4px;min-width:0; }
.kpi-label { font-size:10px;font-weight:700;color:var(--text-3);text-transform:uppercase;letter-spacing:.6px; }
.kpi-value { font-size:26px;font-weight:800;color:var(--text);line-height:1.1;letter-spacing:-.5px;word-break:break-word; }
.kpi-value.small { font-size:16px; }
.kpi-note  { font-size:11px;color:var(--text-3);margin-top:2px; }
.kpi-icon  { width:40px;height:40px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:19px;flex-shrink:0; }
.kpi-icon.blue   { background:var(--blue-light);  color:var(--blue); }
.kpi-icon.green  { background:var(--green-light); color:var(--green); }
.kpi-icon.red    { background:var(--red-light);   color:var(--red); }
.kpi-icon.amber  { background:var(--amber-light); color:var(--amber); }
.kpi-icon.indigo { background:#eef2ff;             color:var(--indigo); }
.kpi-icon.cyan   { background:#cffafe;             color:var(--cyan); }

/* ── Insight strip ────────────────────────────── */
.insight-strip { display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-bottom:24px; }
.insight { background:linear-gradient(135deg,#fff 0%,#f7fbff 100%);border:1px solid var(--border);border-left:3px solid var(--blue);border-radius:var(--radius-sm);padding:12px 14px;display:flex;gap:12px;align-items:flex-start; }
.insight.amber  { border-left-color:var(--amber); }
.insight.red    { border-left-color:var(--red); }
.insight.green  { border-left-color:var(--green); }
.insight.indigo { border-left-color:var(--indigo); }
.insight-icon { width:26px;height:26px;border-radius:6px;background:var(--blue-light);color:var(--blue);display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0; }
.insight.amber  .insight-icon { background:var(--amber-light); color:var(--amber); }
.insight.red    .insight-icon { background:var(--red-light);   color:var(--red); }
.insight.green  .insight-icon { background:var(--green-light); color:var(--green); }
.insight.indigo .insight-icon { background:#eef2ff;             color:var(--indigo); }
.insight-text { font-size:12.5px;color:var(--text-2);line-height:1.45;min-width:0; }
.insight-text strong { color:var(--text);font-weight:700; }

/* ── Cards ────────────────────────────────────── */
.card { background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px;box-shadow:var(--shadow);transition:box-shadow .2s; }
.card:hover { box-shadow:var(--shadow-lg); }
.card-head { margin-bottom:14px;display:flex;align-items:flex-start;justify-content:space-between;gap:12px;padding-bottom:12px;border-bottom:1px solid var(--border); }
.card-head-text { flex:1;min-width:0; }
.card-head h3 { font-size:14.5px;font-weight:700;color:var(--text);letter-spacing:-.1px; }
.card-head p  { font-size:11.5px;color:var(--text-3);margin-top:3px; }
.card-badge { padding:4px 10px;border-radius:999px;background:var(--blue-light);color:var(--blue);font-size:11px;font-weight:700;white-space:nowrap; }

.grid   { display:grid;gap:18px;margin-bottom:22px; }
.grid.cols-2 { grid-template-columns:repeat(auto-fit,minmax(400px,1fr)); }
.grid.cols-3 { grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }

/* ── Form elements ────────────────────────────── */
.field { display:flex;flex-direction:column;gap:5px; }
.field label { font-size:10.5px;font-weight:700;color:var(--text-2);text-transform:uppercase;letter-spacing:.5px; }
.input-grid { display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px; }

/* ── Prediction ───────────────────────────────── */
.prediction-result { min-height:180px;display:flex;flex-direction:column;justify-content:center;gap:10px; }
.prediction-badge { display:inline-flex;width:fit-content;padding:7px 14px;border-radius:999px;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.5px; }
.prediction-badge.red   { background:var(--red-light);   color:#b91c1c; }
.prediction-badge.amber { background:var(--amber-light); color:#8a6900; }
.prediction-badge.green { background:var(--green-light); color:#14532d; }
.prediction-score { font-size:36px;font-weight:800;color:var(--text);letter-spacing:-1px; }
.prediction-note  { font-size:13px;color:var(--text-2);line-height:1.5; }
.prediction-note.muted { color:var(--text-3);font-size:11px; }

.factor-list { display:flex;flex-direction:column;gap:6px;margin-top:6px; }
.factor-row  { display:flex;justify-content:space-between;align-items:center;padding:7px 10px;background:var(--surface-2);border-radius:7px;font-size:12px;border-left:3px solid var(--border-2); }
.factor-row.high   { border-left-color:var(--red); }
.factor-row.medium { border-left-color:var(--amber); }
.factor-row.low    { border-left-color:var(--green); }
.factor-name  { color:var(--text-2); }
.factor-value { font-weight:700;color:var(--text); }

.model-meta-grid { display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:10px;margin-top:8px; }
.model-meta-cell { padding:10px 12px;background:var(--surface-2);border-radius:8px;border:1px solid var(--border); }
.model-meta-cell .label { font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--text-3); }
.model-meta-cell .value { font-size:16px;font-weight:700;color:var(--text);margin-top:4px; }

.model-status-banner { padding:13px 16px;border-radius:var(--radius-sm);border:1px solid;font-size:13px;display:flex;align-items:flex-start;gap:12px;margin-bottom:18px; }
.model-status-banner.ok    { background:linear-gradient(135deg,#edf6fb,#d7ecf8);border-color:#86efac;color:var(--blue); }
.model-status-banner.error { background:linear-gradient(135deg,#fef2f2,#fee2e2);border-color:#fca5a5;color:#991b1b; }
.model-status-banner-icon  { width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;flex-shrink:0; }
.model-status-banner.ok    .model-status-banner-icon { background:#86efac;color:#14532d; }
.model-status-banner.error .model-status-banner-icon { background:#fca5a5;color:#7f1d1d; }

/* ── Buttons ──────────────────────────────────── */
.btn { padding:9px 18px;background:linear-gradient(135deg,var(--blue),var(--indigo));color:white;border:none;border-radius:var(--radius-sm);font:600 12.5px 'DM Sans',sans-serif;cursor:pointer;transition:all .15s;box-shadow:0 2px 6px rgba(47,98,179,.28); }
.btn:hover { transform:translateY(-1px);box-shadow:0 4px 12px rgba(59,130,246,.4); }
.btn:active { transform:translateY(0); }
.btn.secondary { background:var(--surface);color:var(--text-2);border:1px solid var(--border);box-shadow:var(--shadow-sm); }
.btn.secondary:hover { background:var(--surface-2);border-color:var(--border-2);color:var(--text);box-shadow:var(--shadow); }
.btn.success { background:linear-gradient(135deg,var(--green),#3a9e34); }
.btn.success:hover { box-shadow:0 4px 14px rgba(46,114,40,.45); }
.btn.icon-btn { padding:8px 12px;display:inline-flex;align-items:center;gap:7px;font-size:12px; }
.btn.danger { background:linear-gradient(135deg,var(--red),var(--pink)); }

/* ── Filters / chips ──────────────────────────── */
.filter-chip { display:inline-flex;align-items:center;gap:6px;padding:6px 12px;background:var(--surface);border:1px solid var(--border);border-radius:999px;font-size:12px;color:var(--text-2);font-weight:500; }
.filter-chip strong { color:var(--blue);font-weight:700; }
.page-filter-bar { display:flex;gap:14px;align-items:flex-end;flex-wrap:wrap;padding:14px 16px;background:linear-gradient(135deg,#edf6fb,#f7fbff);border:1px solid var(--blue-light);border-radius:var(--radius);margin-bottom:18px;box-shadow:var(--shadow-sm); }
.page-filter-bar .field { min-width:220px; }

/* ── Position controls ────────────────────────── */
.position-controls { display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr)) auto;gap:12px;align-items:end;padding:16px;background:linear-gradient(135deg,#edf6fb,#f7fbff);border-radius:var(--radius);border:1px solid var(--blue-light);margin-bottom:18px; }
.position-stat { padding:12px 14px;background:var(--surface);border-radius:var(--radius-sm);border:1px solid var(--border); }
.position-stat-label { font-size:10px;font-weight:700;color:var(--text-3);text-transform:uppercase;letter-spacing:.5px; }
.position-stat-value { font-size:18px;font-weight:700;color:var(--text);margin-top:3px; }

/* ── Map ──────────────────────────────────────── */
.map-controls { display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap;margin-bottom:14px;padding:14px;background:var(--surface-2);border-radius:var(--radius-sm);border:1px solid var(--border); }
.map-legend { display:flex;gap:18px;flex-wrap:wrap;margin-top:10px;padding:10px 14px;background:var(--surface-2);border-radius:var(--radius-sm);border:1px solid var(--border);font-size:12px;color:var(--text-2); }
.legend-item { display:flex;align-items:center;gap:7px; }
.legend-swatch { width:14px;height:14px;border-radius:4px; }
.legend-swatch.heat { background:linear-gradient(90deg,rgba(147,197,253,1),rgba(245,158,11,1),rgba(239,68,68,1)); }
.map-stat-row { display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px; }
.map-stat-pill { padding:5px 12px;border-radius:999px;background:var(--surface-2);border:1px solid var(--border);font-size:12px;color:var(--text-2);font-weight:500; }
.map-stat-pill strong { color:var(--text);font-weight:700; }
.map-stat-pill.accent { background:var(--blue-light);border-color:#93c5fd;color:#1d4ed8; }

/* ── Map expand / fullscreen overlay ─────────── */
.map-action-bar {
  display: flex; gap: 10px; align-items: center;
  margin-bottom: 12px; flex-wrap: wrap;
}
.map-expand-overlay {
  display: none;
  position: fixed; inset: 0; z-index: 9000;
  background: rgba(8,17,38,.82);
  backdrop-filter: blur(6px);
  animation: fadeOverlay .2s ease;
}
.map-expand-overlay.active { display: flex; flex-direction: column; }
@keyframes fadeOverlay { from { opacity: 0; } to { opacity: 1; } }
.map-expand-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 20px;
  background: linear-gradient(180deg, #081126 0%, #111843 100%);
  border-bottom: 1px solid rgba(255,255,255,.12);
  flex-shrink: 0;
}
.map-expand-title {
  font-size: 15px; font-weight: 700; color: #fff;
  display: flex; align-items: center; gap: 10px;
}
.map-expand-actions { display: flex; gap: 8px; }
.map-expand-body {
  flex: 1; overflow: hidden; position: relative;
  padding: 0;
}
.map-expand-body iframe {
  width: 100%; height: 100%; border: none;
}
.map-close-btn {
  padding: 8px 16px;
  background: rgba(255,255,255,.1);
  border: 1px solid rgba(255,255,255,.2);
  border-radius: 8px;
  color: #fff;
  font: 600 12px 'DM Sans', sans-serif;
  cursor: pointer;
  transition: all .15s;
  display: flex; align-items: center; gap: 6px;
}
.map-close-btn:hover { background: rgba(255,255,255,.2); }
.map-download-btn {
  padding: 8px 16px;
  background: linear-gradient(135deg,#2F62B3,#334F96);
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 8px;
  color: #fff;
  font: 600 12px 'DM Sans', sans-serif;
  cursor: pointer;
  transition: all .15s;
  display: flex; align-items: center; gap: 6px;
}
.map-download-btn:hover { background: linear-gradient(135deg,#3a72c8,#4060b0); box-shadow: 0 4px 14px rgba(47,98,179,.4); }
.map-download-note {
  font-size: 11px; color: rgba(255,255,255,.5);
  padding: 0 20px 10px;
  flex-shrink: 0;
}

/* ── Table ────────────────────────────────────── */
.table-container { overflow-x:auto;border-radius:var(--radius-sm); }
.dash-spreadsheet-container { font-family:'DM Sans',sans-serif !important; }
.dash-spreadsheet-container th { background:linear-gradient(180deg,#f8fafc,#f1f5f9) !important;font-weight:700 !important;color:var(--text) !important;font-size:11px !important;padding:10px 12px !important;border:1px solid var(--border) !important;text-transform:uppercase;letter-spacing:.5px; }
.dash-spreadsheet-container td { padding:9px 12px !important;border:1px solid var(--border) !important;font-size:12px !important;color:var(--text-2) !important; }
.dash-spreadsheet-container tr:nth-child(odd) td  { background:var(--surface-2) !important; }
.dash-spreadsheet-container tr:hover td { background:var(--blue-light) !important; }

/* ── Report page ──────────────────────────────── */
.report-layout { display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px; }
.report-section-grid { display:flex;flex-direction:column;gap:6px; }
.report-section-item { display:flex;align-items:flex-start;gap:10px;padding:10px 12px;border-radius:var(--radius-sm);border:1px solid var(--border);background:var(--surface);transition:all .15s;cursor:pointer; }
.report-section-item:hover { border-color:var(--blue);background:var(--blue-light); }
.report-section-icon { font-size:16px;flex-shrink:0;margin-top:1px; }
.report-section-text { flex:1;min-width:0; }
.report-section-label { font-size:13px;font-weight:600;color:var(--text); }
.report-section-desc  { font-size:11px;color:var(--text-3);margin-top:2px;line-height:1.4; }
.report-preview { background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius);padding:18px; }
.report-preview-kpi { display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px; }
.report-kpi-cell { padding:12px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);text-align:center; }
.report-kpi-cell .val { font-size:22px;font-weight:800;color:var(--text); }
.report-kpi-cell .lbl { font-size:10px;font-weight:700;color:var(--text-3);text-transform:uppercase;letter-spacing:.5px;margin-top:3px; }
.report-action-bar { display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:16px;background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius);margin-top:18px; }
.report-status { font-size:12.5px;color:var(--text-3); }
.report-status.ok    { color:var(--green);font-weight:600; }
.report-status.error { color:var(--red); font-weight:600; }

/* ── Misc ─────────────────────────────────────── */
@media (max-width:900px) {
  .pages { padding:18px; }
  .grid.cols-2, .grid.cols-3 { grid-template-columns:1fr; }
  .input-grid { grid-template-columns:1fr 1fr; }
  .kpi-grid { grid-template-columns:repeat(2,1fr); }
  .header-info h1 { font-size:20px; }
  .report-layout { grid-template-columns:1fr; }
}
@media (max-width:600px) {
  .sidebar { width:100%;position:fixed;height:100vh; }
  .kpi-grid { grid-template-columns:1fr; }
  .input-grid { grid-template-columns:1fr; }
}

::-webkit-scrollbar { width:10px;height:10px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border-2);border-radius:5px;border:2px solid transparent;background-clip:content-box; }
::-webkit-scrollbar-thumb:hover { background:var(--text-4);border:2px solid transparent;background-clip:content-box; }
"""


# ──────────────────────────────────────────────────────────────────────────────
# UI component helpers
# ──────────────────────────────────────────────────────────────────────────────
def kpi_card(label, value, note, icon, tone="blue", value_class="kpi-value") -> html.Div:
    return html.Div(className=f"kpi {tone}", children=[
        html.Div(className="kpi-info", children=[
            html.Div(label, className="kpi-label"),
            html.Div(value, className=value_class),
            html.Div(note,  className="kpi-note"),
        ]),
        html.Div(icon, className=f"kpi-icon {tone}"),
    ])


def insight_card(icon, text_children, tone="blue") -> html.Div:
    return html.Div(className=f"insight {tone}", children=[
        html.Div(icon, className="insight-icon"),
        html.Div(text_children, className="insight-text"),
    ])


def page_header(title, subtitle) -> html.Div:
    return html.Div(className="header", children=[
        html.Div(className="header-info", children=[html.H1(title), html.P(subtitle)]),
        html.Div(className="header-chip", children=[
            html.Span(className="live-dot"),
            html.Span("Live"),
        ]),
    ])


def graph_card(title, note, graph_id, badge=None) -> html.Div:
    head = [html.Div(className="card-head-text", children=[html.H3(title), html.P(note)])]
    if badge:
        head.append(html.Div(badge, className="card-badge"))
    return html.Div(className="card", children=[
        html.Div(className="card-head", children=head),
        dcc.Graph(id=graph_id, config={"displayModeBar": False}, style={"margin": "-8px"}),
    ])


def table_card(table_id, title="Data", note="", page_size=10,
               header_id=None, note_id=None) -> html.Div:
    return html.Div(className="card", children=[
        html.Div(className="card-head", children=[
            html.Div(className="card-head-text", children=[
                html.H3(title, id=header_id) if header_id else html.H3(title),
                html.P(note, id=note_id) if note_id else html.P(note),
            ]),
        ]),
        html.Div(className="table-container", children=[
            dash_table.DataTable(
                id=table_id,
                page_size=page_size,
                sort_action="native",
                filter_action="none",
                style_table={"overflowX": "auto"},
                style_cell={
                    "fontFamily": "'DM Sans',sans-serif",
                    "fontSize": "12px", "padding": "9px 12px",
                    "border": f"1px solid {COLORS['border']}",
                    "maxWidth": "240px", "overflow": "hidden", "textOverflow": "ellipsis",
                },
                style_header={
                    "backgroundColor": COLORS["surface_2"], "fontWeight": "700",
                    "color": COLORS["text"], "border": f"1px solid {COLORS['border']}",
                    "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": ".5px",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": COLORS["surface_2"]},
                ],
            ),
        ]),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────────────────────
def overview_page() -> html.Div:
    return html.Div(id="page-overview", className="page", children=[
        page_header("Overview", "Key metrics, event trends, type distribution and status."),
        html.Div(id="kpi-overview",  className="kpi-grid"),
        html.Div(id="insight-strip", className="insight-strip"),
        html.Div(className="grid cols-2", children=[
            graph_card("Events Over Time",     "Monthly alarm trend",                         "fig-monthly"),
            graph_card("Most Active Time",     "Morning / Afternoon / Night / Midnight",      "fig-time-period"),
            graph_card("Alarms by Hour of Day", "24-hour distribution — spot peak hours",     "fig-hour"),
            graph_card("Event Types",          "Top categories",                              "fig-type"),
            graph_card("Threat Level",         "Red / Amber / Green breakdown",               "fig-status"),
        ]),
        table_card("data-preview", "Latest Incidents",
                   "10 most recent incident records", 10,
                   header_id="data-preview-title", note_id="data-preview-note"),
    ])


def trends_page() -> html.Div:
    return html.Div(id="page-trends", className="page hidden", children=[
        page_header("Trends", "Temporal patterns — what's driving volume and when."),
        html.Div(className="grid cols-2", children=[
            graph_card("Monthly Count",    "Total events per month",                         "fig-monthly-bar"),
            graph_card("Event Type Trend", "Volume breakdown by type per month",             "fig-type-trend", badge="Stacked"),
        ]),
        html.Div(className="grid cols-2", children=[
            graph_card("Day-of-Week Pattern", "Which days drive the most alarms",             "fig-dow"),
            graph_card("Resolution Time by Type", "Median resolution hours by event type",    "fig-resolution"),
        ]),
    ])


def hotspots_page() -> html.Div:
    return html.Div(id="page-hotspots", className="page hidden", children=[
        page_header("Hotspots & Position Analysis",
                    "Route hotspots, event-type matrix, and custom bin-range analysis."),
        html.Div(className="page-filter-bar", children=[
            html.Div(className="field", children=[
                html.Label("Filter by Month"),
                dcc.Dropdown(id="hotspot-month", value=ALL_MONTHS_VALUE,
                             clearable=False, placeholder="Choose a month"),
            ]),
            html.Div(id="hotspot-filter-chip", className="filter-chip"),
        ]),
        html.Div(className="grid cols-2", children=[
            graph_card("Top Routes",         "Most frequent routes",                         "fig-route"),
            graph_card("Route × Event Type", "Threat dominance by route segment",            "fig-route-heatmap", badge="Matrix"),
            graph_card("Top Marker Names",   "Markers with the highest alarm frequency",     "fig-marker"),
        ]),
        html.Div(className="position-controls", children=[
            html.Div(className="field", children=[
                html.Label("Bin Size (m)"),
                dcc.Input(id="pos-bin-size", type="number", value=5000, min=100, step=100),
            ]),
            html.Div(className="field", children=[
                html.Label("Max Position (optional)"),
                dcc.Input(id="pos-max", type="number", placeholder="Auto-detect", min=0, step=1000),
            ]),
            html.Div(className="field", children=[
                html.Label("Sort By"),
                dcc.Dropdown(
                    id="pos-sort",
                    options=[
                        {"label": "Range Start (asc)",  "value": "range"},
                        {"label": "Alarm Count (desc)", "value": "count"},
                        {"label": "Red Threats (desc)", "value": "red"},
                    ],
                    value="range", clearable=False,
                ),
            ]),
            html.Button("⬇ Export CSV", id="pos-download-btn", className="btn secondary"),
            dcc.Download(id="pos-download"),
        ]),
        html.Div(id="position-summary", style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit,minmax(170px,1fr))",
            "gap": "12px", "marginBottom": "18px",
        }),
        graph_card("Alarms by Position Range",
                   "Distribution across your chosen bin size", "fig-position-bar"),
        table_card("position-table", "Position Range Breakdown",
                   "Alarm count, GPS centroid, and dominant route per bin",
                   page_size=20),
    ])


def map_page() -> html.Div:
    return html.Div(id="page-map", className="page hidden", children=[
        page_header("Geographic Map", "Alarm heatmap and incident location overlay."),

        # ── Fullscreen expand overlay ──────────────────────────────────────
        html.Div(id="map-expand-overlay", className="map-expand-overlay", children=[
            html.Div(className="map-expand-header", children=[
                html.Div(className="map-expand-title", children=[
                    html.Span("⌖", style={"fontSize": "20px"}),
                    html.Span("Incident Map — Expanded View"),
                ]),
                html.Div(className="map-expand-actions", children=[
                    html.Button(
                        ["⬇  Download Map (JPEG)"],
                        id="map-download-img-btn",
                        className="map-download-btn",
                    ),
                    html.Button(
                        ["✕  Close"],
                        id="map-close-btn",
                        className="map-close-btn",
                    ),
                ]),
            ]),
            html.Div(
                "Download saves a static JPEG image of the current map view. "
                "JPEG export needs Selenium, Pillow, and a headless browser driver installed.",
                className="map-download-note",
            ),
            html.Div(className="map-expand-body", children=[
                html.Iframe(
                    id="folium-map-expanded",
                    srcDoc="",
                    style={"width": "100%", "height": "100%", "border": "none"},
                ),
            ]),
        ]),

        html.Div(className="page-filter-bar", children=[
            html.Div(className="field", children=[
                html.Label("Filter by Month"),
                dcc.Dropdown(id="map-month", value=ALL_MONTHS_VALUE,
                             clearable=False, placeholder="Choose a month"),
            ]),
            html.Div(id="map-filter-chip", className="filter-chip"),
        ]),
        html.Div(className="card", children=[
            html.Div(className="card-head", children=[
                html.Div(className="card-head-text", children=[
                    html.H3("Incident Map"),
                    html.P("Alarms rendered as heatmap; incidents as warning-triangle markers."),
                ]),
            ]),
            html.Div(className="map-controls", children=[
                html.Div(className="field", children=[
                    html.Label("Show"),
                    dcc.Dropdown(
                        id="map-mode",
                        options=[
                            {"label": "Alarms only",    "value": "alarm"},
                            {"label": "Incidents only", "value": "incident"},
                            {"label": "Both",           "value": "both"},
                        ],
                        value="both", clearable=False, style={"width": "180px"},
                    ),
                ]),
                html.Div(className="field", children=[
                    html.Label("Map Style"),
                    dcc.Dropdown(
                        id="map-style",
                        options=[
                            {"label": "Esri Satellite", "value": "esri"},
                            {"label": "OpenStreetMap",  "value": "osm"},
                            {"label": "Carto Light",    "value": "carto"},
                        ],
                        value="esri", clearable=False, style={"width": "180px"},
                    ),
                ]),
                html.Div(className="field", style={"flex": "1", "minWidth": "220px"}, children=[
                    html.Label("Heatmap Intensity"),
                    dcc.Slider(
                        id="map-intensity", min=5, max=40, step=1, value=18,
                        marks={5: "Low", 18: "Default", 40: "High"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ]),
                html.Div(className="field", children=[
                    html.Label("\u00A0"),
                    html.Button("↻ Reset View", id="map-reset", className="btn secondary"),
                ]),
            ]),

            # ── Map action bar (expand + download) ───────────────────────
            html.Div(className="map-action-bar", children=[
                html.Button(
                    "⛶  Expand Map",
                    id="map-expand-btn",
                    className="btn icon-btn",
                    n_clicks=0,
                ),
                html.Button(
                    "⬇  Download Map (JPEG)",
                    id="map-download-btn",
                    className="btn secondary icon-btn",
                    n_clicks=0,
                ),
                html.Div(id="map-download-status", style={
                    "fontSize": "12px", "color": COLORS["text_3"], "alignSelf": "center",
                }),
                dcc.Download(id="map-download-file"),
            ]),

            html.Div(id="map-stats", className="map-stat-row"),
            html.Iframe(
                id="folium-map", srcDoc="",
                style={"width": "100%", "height": "620px", "border": "none",
                       "borderRadius": "12px"},
            ),
            html.Div(className="map-legend", children=[
                html.Div(className="legend-item", children=[
                    html.Span(className="legend-swatch heat"),
                    html.Span("Alarm density (heatmap)"),
                ]),
                html.Div(className="legend-item", children=[
                    html.Span("⚠", style={"fontSize": "18px", "color": COLORS["red"]}),
                    html.Span("Incident / landslide marker"),
                ]),
                html.Div(className="legend-item", children=[
                    html.Span("💡", style={"fontSize": "14px"}),
                    html.Span("Tip: Click ⛶ Expand for fullscreen view",
                              style={"fontStyle": "italic", "color": COLORS["text_3"]}),
                ]),
            ]),
        ]),

    ])


def prediction_page() -> html.Div:
    return html.Div(id="page-prediction", className="page hidden", children=[
        page_header(
            "Risk Prediction",
            "ML model (v4) — predicts whether each (Route × Type × Position) pattern repeats next month.",
        ),
        html.Div(id="model-status-banner"),
        html.Div(className="grid cols-2", children=[
            html.Div(className="card", children=[
                html.Div(className="card-head", children=[
                    html.Div(className="card-head-text", children=[
                        html.H3("Check a Specific Pattern"),
                        html.P("Select Route, Event Type, and Position Range."),
                    ]),
                ]),
                html.Div(className="input-grid", children=[
                    html.Div(className="field", children=[
                        html.Label("Route"),
                        dcc.Dropdown(id="pred-route", placeholder="Choose route"),
                    ]),
                    html.Div(className="field", children=[
                        html.Label("Event Type"),
                        dcc.Dropdown(id="pred-type", placeholder="Choose type"),
                    ]),
                    html.Div(className="field", children=[
                        html.Label("Position Range (5 km bin)"),
                        dcc.Dropdown(id="pred-position", placeholder="Choose range"),
                    ]),
                ]),
                html.Div(id="pred-result", className="prediction-result"),
            ]),
            html.Div(className="card", children=[
                html.Div(className="card-head", children=[
                    html.Div(className="card-head-text", children=[
                        html.H3("Model Information"),
                        html.P(f"v4 — HistGradientBoosting ensemble, isotonic calibration, {len(MODEL_FEATURES)} features."),
                    ]),
                ]),
                html.Div(id="model-meta-grid", className="model-meta-grid"),
                html.Div(style={"marginTop": "14px", "fontSize": "12px",
                                "color": COLORS["text_3"], "lineHeight": "1.6"}, children=[
                    html.P(
                        "Groups events into monthly buckets per (Route × Type × 5 km bin) "
                        "and predicts whether the pattern recurs next month."
                    ),
                    html.P(
                        "Features: 3-month lag, 6-month rolling stats, EWMA, momentum, "
                        "active ratio, route-type popularity, neighbor-bin activity — "
                        "all leak-free (shifted, no future data)."
                    ),
                ]),
            ]),
        ]),
        html.Div(className="card", style={"marginTop": "18px"}, children=[
            html.Div(className="card-head", children=[
                html.Div(className="card-head-text", children=[
                    html.H3("High-Risk Patterns for Next Month"),
                    html.P("Patterns ranked by predicted repeat probability."),
                ]),
                html.Div("Auto-ranked", className="card-badge"),
            ]),
            html.Div(style={"display": "flex", "gap": "12px", "marginBottom": "12px",
                            "alignItems": "flex-end", "flexWrap": "wrap"}, children=[
                html.Div(className="field", style={"minWidth": "170px"}, children=[
                    html.Label("Show top"),
                    dcc.Dropdown(
                        id="pred-topn",
                        options=[
                            {"label": "Top 10",  "value": 10},
                            {"label": "Top 25",  "value": 25},
                            {"label": "Top 50",  "value": 50},
                            {"label": "Top 100", "value": 100},
                        ],
                        value=25, clearable=False,
                    ),
                ]),
                html.Div(className="field", style={"minWidth": "170px"}, children=[
                    html.Label("Risk Filter"),
                    dcc.Dropdown(
                        id="pred-risk-filter",
                        options=[
                            {"label": "All Risks",     "value": "all"},
                            {"label": "High Only",     "value": "High"},
                            {"label": "High + Medium", "value": "HM"},
                        ],
                        value="all", clearable=False,
                    ),
                ]),
                html.Button("⬇ Download CSV", id="pred-download-btn", className="btn"),
                dcc.Download(id="pred-download"),
            ]),
            html.Div(className="table-container", children=[
                dash_table.DataTable(
                    id="pred-table",
                    page_size=20, sort_action="native", filter_action="none",
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "fontFamily": "'DM Sans',sans-serif",
                        "fontSize": "12px", "padding": "9px 12px",
                        "border": f"1px solid {COLORS['border']}",
                        "maxWidth": "240px", "overflow": "hidden", "textOverflow": "ellipsis",
                    },
                    style_header={
                        "backgroundColor": COLORS["surface_2"], "fontWeight": "700",
                        "color": COLORS["text"], "border": f"1px solid {COLORS['border']}",
                        "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": ".5px",
                    },
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": COLORS["surface_2"]},
                        {"if": {"filter_query": '{risk_level} = "High"'},
                         "backgroundColor": "#fee2e2", "color": "#7f1d1d", "fontWeight": "600"},
                        {"if": {"filter_query": '{risk_level} = "Medium"'},
                         "backgroundColor": "#fef3c7", "color": "#8a6900"},
                    ],
                ),
            ]),
        ]),
    ])


def explorer_page() -> html.Div:
    return html.Div(id="page-explorer", className="page hidden", children=[
        page_header("Report Builder",
                    "Configure and download a professional PDF summary of your alarm data."),

        html.Div(className="report-layout", children=[
            html.Div(children=[
                html.Div(className="card", style={"marginBottom": "18px"}, children=[
                    html.Div(className="card-head", children=[
                        html.Div(className="card-head-text", children=[
                            html.H3("Report Settings"),
                            html.P("Choose the period and layout bin size."),
                        ]),
                    ]),
                    html.Div(className="input-grid", children=[
                        html.Div(className="field", children=[
                            html.Label("Reporting Period"),
                            dcc.Dropdown(id="report-month", value=ALL_MONTHS_VALUE,
                                         clearable=False, placeholder="Choose period"),
                        ]),
                        html.Div(className="field", children=[
                            html.Label("Position Bin Size (m)"),
                            dcc.Input(id="report-bin-size", type="number",
                                      value=5000, min=100, step=100),
                        ]),
                    ]),
                ]),

                html.Div(className="card", children=[
                    html.Div(className="card-head", children=[
                        html.Div(className="card-head-text", children=[
                            html.H3("Select Report Sections"),
                            html.P("Check the sections to include in the PDF."),
                        ]),
                        html.Div(children=[
                            html.Button("All",  id="report-select-all",  className="btn secondary",
                                        n_clicks=0,
                                        style={"padding": "5px 10px", "fontSize": "11px",
                                               "marginRight": "6px"}),
                            html.Button("None", id="report-select-none", className="btn secondary",
                                        n_clicks=0,
                                        style={"padding": "5px 10px", "fontSize": "11px"}),
                        ]),
                    ]),
                    dcc.Checklist(
                        id="report-sections",
                        options=[{"label": f" {s['icon']}  {s['label']}", "value": s["value"]}
                                 for s in REPORT_SECTIONS],
                        value=ALL_SECTION_VALUES,
                        labelStyle={
                            "display": "flex", "alignItems": "center", "gap": "8px",
                            "padding": "8px 6px", "fontSize": "13px",
                            "color": COLORS["text_2"], "cursor": "pointer",
                            "borderBottom": f"1px solid {COLORS['border']}",
                        },
                        inputStyle={"width": "15px", "height": "15px", "cursor": "pointer"},
                    ),
                ]),
            ]),

            html.Div(children=[
                html.Div(className="card report-preview", children=[
                    html.Div(className="card-head", children=[
                        html.Div(className="card-head-text", children=[
                            html.H3("Live Preview"),
                            html.P("Summary of what will appear in the report."),
                        ]),
                    ]),
                    html.Div(id="report-preview-content"),
                ]),
            ]),
        ]),

        html.Div(className="report-action-bar", children=[
            html.Button("⬇  Generate & Download PDF",
                        id="report-generate-btn", className="btn success"),
            html.Span(id="report-status", className="report-status"),
        ]),
        dcc.Download(id="report-download"),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────
_INDEX_TEMPLATE = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>__STYLES__</style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
    <script>
    (function() {
      // Escape key closes map overlay
      document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
          var ov = document.getElementById('map-expand-overlay');
          if (ov && ov.classList.contains('active')) {
            ov.classList.remove('active');
            document.body.style.overflow = '';
          }
        }
      });
    })();
    </script>
  </body>
</html>"""

app.index_string = _INDEX_TEMPLATE.replace("__STYLES__", STYLES)

app.layout = html.Div([
    dcc.Store(id="sidebar-state", data=True),
    dcc.Store(id="active-page",   data="overview"),
    dcc.Store(id="data-store",    data=load_initial()),
    dcc.Store(id="pred-store",    data=None),
    dcc.Store(id="map-overlay-state", data=False),  # False=closed, True=open

    html.Div(id="shell", className="shell open", children=[
        html.Aside(className="sidebar", children=[
            html.Div(className="brand", children=[
                html.Div("◆", className="brand-logo"),
                html.Div(className="brand-text", children=[
                    html.Div("DAS Intelligence", className="brand-title"),
                    html.Div("Fibre Monitoring",  className="brand-subtitle"),
                ]),
            ]),
            html.Div("Navigation", className="nav-section"),
            html.Nav(className="nav-list", children=[
                html.Button(
                    [html.Span(p["icon"], className="nav-icon"), html.Span(p["label"])],
                    id={"type": "nav", "page": p["key"]},
                    className="nav-item" + (" active" if p["key"] == "overview" else ""),
                    n_clicks=0,
                )
                for p in PAGES
            ]),
            html.Div(className="nav-spacer"),
            html.Div(className="nav-footer", children=[
                html.Div(className="footer-row", children=[
                    html.Div("◉", className="footer-icon"),
                    html.Div(children=[
                        html.Div("Dataset",              className="footer-label"),
                        html.Div(DATA_SOURCE_FILENAME,   className="footer-value"),
                    ]),
                ]),
                html.Div(className="footer-version", children=[
                    "Version",
                    html.Br(),
                    "v6.0 · ML-powered",
                ]),
            ]),
        ]),
        html.Main(className="content", children=[
            html.Button("✕", id="sidebar-toggle", className="toggle", n_clicks=0),
            html.Div(className="pages", children=[
                overview_page(),
                trends_page(),
                hotspots_page(),
                map_page(),
                prediction_page(),
                explorer_page(),
            ]),
        ]),
    ]),
])


# ──────────────────────────────────────────────────────────────────────────────
# Map expand / close — clientside callbacks (pure JS, no server round-trip)
# ──────────────────────────────────────────────────────────────────────────────
app.clientside_callback(
    """
    function(n_expand, n_close, current) {
        var triggered = window.dash_clientside.callback_context.triggered;
        if (!triggered || triggered.length === 0) { return window.dash_clientside.no_update; }
        var prop_id = triggered[0].prop_id;

        var overlay = document.getElementById('map-expand-overlay');
        if (!overlay) { return current; }

        if (prop_id.indexOf('map-expand-btn') >= 0) {
            overlay.classList.add('active');
            document.body.style.overflow = 'hidden';
            // sync expanded iframe
            var src = document.getElementById('folium-map');
            var exp = document.getElementById('folium-map-expanded');
            if (src && exp) { exp.srcdoc = src.srcdoc; }
            return true;
        }
        if (prop_id.indexOf('map-close-btn') >= 0) {
            overlay.classList.remove('active');
            document.body.style.overflow = '';
            return false;
        }
        return current;
    }
    """,
    Output("map-overlay-state", "data"),
    Input("map-expand-btn",  "n_clicks"),
    Input("map-close-btn",   "n_clicks"),
    State("map-overlay-state", "data"),
    prevent_initial_call=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Navigation callbacks
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("shell",          "className"),
    Output("sidebar-toggle", "children"),
    Input("sidebar-state",   "data"),
)
def update_sidebar(is_open):
    open_state = bool(is_open) if is_open is not None else True
    return ("shell open", "✕") if open_state else ("shell closed", "☰")


@app.callback(
    Output("sidebar-state", "data"),
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar-state",  "data"),
    prevent_initial_call=True,
)
def toggle_sidebar(_, current):
    current_bool = bool(current) if current is not None else True
    return not current_bool


@app.callback(
    Output("active-page", "data"),
    Input({"type": "nav", "page": ALL}, "n_clicks"),
    State("active-page", "data"),
    prevent_initial_call=True,
)
def set_page(_, current: str) -> str:
    trigger = callback_context.triggered_id
    return trigger.get("page", current) if isinstance(trigger, dict) else current


@app.callback(
    *[Output({"type": "nav", "page": p["key"]}, "className") for p in PAGES],
    Input("active-page", "data"),
)
def update_nav(active: str):
    return tuple(
        "nav-item active" if p["key"] == active else "nav-item"
        for p in PAGES
    )


@app.callback(
    *[Output(f"page-{p['key']}", "className") for p in PAGES],
    Input("active-page", "data"),
)
def show_page(active: str):
    return tuple(
        "page" if p["key"] == active else "page hidden"
        for p in PAGES
    )


# ──────────────────────────────────────────────────────────────────────────────
# Month dropdowns
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("hotspot-month", "options"),
    Output("map-month",     "options"),
    Output("report-month",  "options"),
    Input("data-store", "data"),
)
def populate_month_dropdowns(records):
    df = from_store(records)
    opts = available_months(df)
    return opts, opts, opts


# ──────────────────────────────────────────────────────────────────────────────
# Overview KPIs + insights
# ──────────────────────────────────────────────────────────────────────────────
def _fmt_minutes(m) -> str:
    try:
        if m is None or pd.isna(m):
            return "—"
        m = float(m)
        if math.isinf(m) or math.isnan(m):
            return "—"
        if m < 60:
            return f"{m:.1f} min"
        h = m / 60
        if h < 24:
            return f"{h:.1f} hr"
        return f"{h/24:.1f} d"
    except (TypeError, ValueError):
        return "—"


def _build_kpis(df: pd.DataFrame) -> list:
    total_records = len(df)

    if "_event_class" in df.columns:
        incident_mask = df["_event_class"].astype(str).str.strip().str.lower().eq("incident")
    else:
        incident_mask = pd.Series(False, index=df.index)

    incident_count = int(incident_mask.sum())
    alarm_df       = df.loc[~incident_mask].copy()
    alarm_count    = len(alarm_df)

    resolved = 0
    if "_status" in alarm_df.columns and len(alarm_df) > 0:
        resolved = int(
            alarm_df["_status"].fillna("").astype(str)
            .str.strip().str.lower().eq("resolved").sum()
        )

    high = 0
    if "_threat_key" in alarm_df.columns and len(alarm_df) > 0:
        high = int(alarm_df["_threat_key"].astype(str).str.lower().eq("red").sum())

    avg_res = np.nan
    if "_res_minutes" in alarm_df.columns and len(alarm_df) > 0:
        res_series = pd.to_numeric(alarm_df["_res_minutes"], errors="coerce")
        if res_series.notna().any():
            avg_res = res_series.median(skipna=True)

    # Count ROWS where Route itself is missing/null/blank.
    # This fixes inflated counts caused by also counting distance/marker gaps.

    busiest_route = "—"
    if "_route" in alarm_df.columns and alarm_df["_route"].notna().any():
        busiest_route = alarm_df["_route"].dropna().value_counts().index[0]
        if len(busiest_route) > 28:
            busiest_route = busiest_route[:26] + "…"

    return [
        kpi_card("Total Alarms", f"{alarm_count:,}",
                 f"from {total_records:,} total records", "▦", "blue"),
        kpi_card("Resolved",    f"{resolved:,}",
                 "resolved alarm cases", "✓", "green"),
        kpi_card("High Threat", f"{high:,}",
                 "red-level alarms", "!", "red"),
        kpi_card("Incidents",   f"{incident_count:,}",
                 "incident records excluded", "⚠", "amber"),
        kpi_card("Median Resolution", _fmt_minutes(avg_res),
                 "alarm → resolved", "⏱", "indigo"),
        kpi_card("Busiest Route", busiest_route,
                 "highest alarm volume", "◎", "cyan",
                 value_class="kpi-value small"),
    ]


def _build_insights(df: pd.DataFrame) -> list:
    insights = []
    if df.empty:
        return insights

    monthly = monthly_counts(df)
    if not monthly.empty:
        top   = monthly.loc[monthly["Count"].idxmax()]
        share = top["Count"] / monthly["Count"].sum() * 100
        insights.append(insight_card(
            "📈",
            [html.Span("Peak month: "), html.Strong(str(top["period"])),
             html.Span(f" — {int(top['Count']):,} events ({share:.1f}% of total).")],
            tone="indigo",
        ))

    if "_type" in df.columns and df["_type"].notna().any():
        tc       = df["_type"].value_counts()
        top_type = tc.index[0]
        share    = tc.iloc[0] / tc.sum() * 100
        insights.append(insight_card(
            "🎯",
            [html.Strong(str(top_type)),
             html.Span(f" accounts for {share:.1f}% of all events"),
             html.Span(f" ({int(tc.iloc[0]):,} of {int(tc.sum()):,}).")],
            tone="amber" if share > 30 else "blue",
        ))

    if "_hour" in df.columns and df["_hour"].notna().any():
        hc        = df["_hour"].value_counts().sort_index()
        peak_hour = int(hc.idxmax())
        next_hour = (peak_hour + 1) % 24
        insights.append(insight_card(
            "🕒",
            [html.Span("Peak hour: "), html.Strong(f"{peak_hour:02d}:00–{next_hour:02d}:00"),
             html.Span(f" with {int(hc.max()):,} events.")],
            tone="red",
        ))

    return insights[:3]


@app.callback(
    Output("kpi-overview",  "children"),
    Output("insight-strip", "children"),
    Input("data-store", "data"),
)
def update_kpis(records):
    df = from_store(records)
    return _build_kpis(df), _build_insights(df)


# ──────────────────────────────────────────────────────────────────────────────
# Overview figures — improved aesthetics
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("fig-monthly",     "figure"),
    Output("fig-time-period", "figure"),
    Output("fig-hour",        "figure"),
    Output("fig-type",        "figure"),
    Output("fig-status",      "figure"),
    Input("data-store",  "data"),
    Input("active-page", "data"),
)
def update_overview(records, active):
    if active != "overview":
        return no_update, no_update, no_update, no_update, no_update

    df = from_store(records)
    if df.empty:
        return empty_fig(height=260), empty_fig(height=260), empty_fig(height=260), empty_fig(height=260), empty_fig(height=260)

    if "_event_class" in df.columns:
        alarm_mask = ~df["_event_class"].astype(str).str.lower().eq("incident")
        chart_df   = df.loc[alarm_mask].copy()
    else:
        chart_df = df.copy()

    # Monthly trend — gradient area chart
    monthly = monthly_counts(chart_df)
    if not monthly.empty:
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(
            x=monthly["period"], y=monthly["Count"],
            mode="lines+markers",
            line=dict(color=COLORS["blue"], width=2.8, shape="spline"),
            marker=dict(size=7, color="#ffffff", line=dict(color=COLORS["blue"], width=2.5)),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.12)",
            hovertemplate="<b>%{x}</b><br>Events: %{y:,}<extra></extra>",
            showlegend=False,
        ))
        fig_m.update_layout(xaxis_type="category")
        fig_m = polish(fig_m, 260, False)
    else:
        fig_m = empty_fig("No date data", 260)

    # Time of day — gradient bars
    time_df = time_of_day_counts(chart_df)
    if not time_df.empty:
        tod_colors = [COLORS["slate"], COLORS["blue"], COLORS["cyan"], COLORS["teal"]]
        fig_time = go.Figure(go.Bar(
            x=time_df["Time Period"], y=time_df["Count"],
            marker_color=tod_colors[:len(time_df)],
            marker_line_width=0,
            text=time_df["Count"], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Alarms: %{y:,}<extra></extra>",
        ))
        busiest_row = time_df.loc[time_df["Count"].idxmax()]
        fig_time.update_layout(
            annotations=[dict(
                text=f"Peak: {busiest_row['Time Period']} ({int(busiest_row['Count']):,})",
                x=0.5, y=1.12, xref="paper", yref="paper", showarrow=False,
                font=dict(size=12, color=COLORS["text_3"]),
            )],
            xaxis_type="category",
        )
        fig_time.update_xaxes(title="")
        fig_time.update_yaxes(title="Count")
        fig_time = polish(fig_time, 260, False)
    else:
        fig_time = empty_fig("No time data", 260)


    # Alarms by hour of day — 24-hour distribution
    if "_hour" in chart_df.columns and chart_df["_hour"].notna().any():
        hc = (
            chart_df.dropna(subset=["_hour"])
            .assign(_hour_int=lambda d: d["_hour"].astype(int))
            .groupby("_hour_int")
            .size()
            .reindex(range(24), fill_value=0)
            .reset_index(name="Count")
            .rename(columns={"_hour_int": "Hour"})
        )
        hc["Label"] = hc["Hour"].apply(lambda h: f"{int(h):02d}:00")

        def _hour_color(h: int) -> str:
            if 0 <= h < 6:
                return COLORS["indigo"]
            if 6 <= h < 12:
                return COLORS["green"]
            if 12 <= h < 18:
                return COLORS["blue"]
            return COLORS["teal"]

        peak_hour = int(hc.loc[hc["Count"].idxmax(), "Hour"]) if not hc.empty else 0
        peak_count = int(hc.loc[hc["Hour"] == peak_hour, "Count"].iloc[0]) if not hc.empty else 0

        bar_colors = [
            COLORS["amber_dark"] if int(h) == peak_hour else _hour_color(int(h))
            for h in hc["Hour"]
        ]

        # Use an empty string for bars without labels.
        # This prevents Plotly from rendering None as the literal text "null".
        hc["Bar Text"] = [
            f"{int(count):,}" if int(hour) == peak_hour and int(count) > 0 else ""
            for hour, count in zip(hc["Hour"], hc["Count"])
        ]

        fig_hour = go.Figure(go.Bar(
            x=hc["Label"],
            y=hc["Count"],
            marker_color=bar_colors,
            marker_line_width=0,
            text=hc["Bar Text"],
            textposition="outside",
            textfont=dict(color=COLORS["amber_dark"], size=11, family="DM Sans"),
            hovertemplate="<b>%{x}</b><br>Alarms: %{y:,}<extra></extra>",
        ))
        fig_hour.update_layout(
            xaxis_type="category",
            annotations=[dict(
                text=f"Peak: {peak_hour:02d}:00 ({peak_count:,} alarms)",
                x=0.5,
                y=1.12,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12, color=COLORS["text_3"]),
            )],
        )
        fig_hour.update_xaxes(title="Hour of Day", tickangle=-45)
        fig_hour.update_yaxes(title="Alarms")
        fig_hour = polish(fig_hour, 260, False)
    else:
        fig_hour = empty_fig("No hour-of-day data", 260)

    # Event types — indigo gradient bars
    types_df = count_col(df, "_type", "Type", 10)
    if not types_df.empty:
        n = len(types_df)
        bar_colors = [
            BAR_SEQ_INDIGO[min(i, len(BAR_SEQ_INDIGO) - 1)]
            for i in range(n)
        ]
        fig_t = go.Figure(go.Bar(
            x=types_df["Count"],
            y=types_df.sort_values("Count")["Type"],
            orientation="h",
            marker_color=bar_colors[::-1],
            marker_line_width=0,
            hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
        ))
        fig_t = polish(fig_t, 260, False)
    else:
        fig_t = empty_fig("No type data", 260)

    # Threat level donut
    if "_threat_key" in df.columns and df["_threat_key"].notna().any():
        threat_counts = (
            df["_threat_key"]
            .fillna("unknown").astype(str).str.strip().str.lower()
            .replace("", "unknown")
            .value_counts().reset_index()
        )
        threat_counts.columns = ["Threat", "Count"]
        threat_counts["Label"] = threat_counts["Threat"].str.title()
        threat_colors = [
            THREAT_MAP.get(t, COLORS["text_4"]) for t in threat_counts["Threat"]
        ]
        fig_s = make_clean_donut(
            df_count=threat_counts, label_col="Label", value_col="Count",
            colors=threat_colors, height=260, min_label_pct=1.0,
        )
    else:
        fig_s = empty_fig("No threat data", 260)

    return fig_m, fig_time, fig_hour, fig_t, fig_s


# ──────────────────────────────────────────────────────────────────────────────
# Trends — improved aesthetics
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("fig-monthly-bar", "figure"),
    Output("fig-type-trend",  "figure"),
    Output("fig-dow",         "figure"),
    Output("fig-resolution",  "figure"),
    Input("data-store",  "data"),
    Input("active-page", "data"),
)
def update_trends(records, active):
    if active != "trends":
        return no_update, no_update, no_update, no_update

    df = from_store(records)
    if df.empty:
        return empty_fig(height=320), empty_fig(height=320), empty_fig(height=320), empty_fig(height=320)

    # Aesthetic stacked bar palette
    TREND_PALETTE = [
        COLORS["blue"], COLORS["teal"], COLORS["indigo"], COLORS["cyan"],
        COLORS["violet"], COLORS["green"], COLORS["slate"], COLORS["pink"],
    ]

    # Monthly bar — gradient teal→blue
    monthly = monthly_counts(df)
    if not monthly.empty:
        n = len(monthly)
        bar_colors = [
            BAR_SEQ_BLUE[min(i, len(BAR_SEQ_BLUE) - 1)]
            for i in range(n)
        ]
        fig_m = go.Figure(go.Bar(
            x=monthly["period"], y=monthly["Count"],
            marker_color=bar_colors,
            marker_line_width=0,
            text=monthly["Count"], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Events: %{y:,}<extra></extra>",
        ))
        fig_m.update_layout(xaxis_type="category")
        fig_m.update_xaxes(title="")
        fig_m.update_yaxes(title="Count")
        fig_m = polish(fig_m, 320, False)
    else:
        fig_m = empty_fig(height=320)

    # Stacked type trend
    if "_dt" in df.columns and df["_dt"].notna().any() and "_type" in df.columns:
        tmp = df.dropna(subset=["_dt", "_type"]).copy()
        use_ym_ts = (
            "_ym_ts" in tmp.columns
            and tmp["_ym_ts"].notna().any()
            and pd.api.types.is_datetime64_any_dtype(tmp["_ym_ts"])
        )
        tmp["_period_ts"] = (
            tmp["_ym_ts"] if use_ym_ts
            else tmp["_dt"].dt.to_period("M").dt.to_timestamp()
        )
        tmp["period"] = tmp["_period_ts"].dt.strftime("%b %Y")
        grouped = (
            tmp.groupby(["_period_ts", "period", "_type"]).size()
            .reset_index(name="Count").sort_values("_period_ts")
        )
        top_types = tmp["_type"].value_counts().head(6).index.tolist()
        grouped["_type_grp"] = grouped["_type"].where(grouped["_type"].isin(top_types), "Other")
        grouped = (
            grouped.groupby(["_period_ts", "period", "_type_grp"], as_index=False)["Count"]
            .sum().sort_values("_period_ts")
        )
        unique_types = list(grouped["_type_grp"].unique())
        color_map = {t: TREND_PALETTE[i % len(TREND_PALETTE)] for i, t in enumerate(unique_types)}

        fig_tt = px.bar(
            grouped, x="period", y="Count", color="_type_grp",
            color_discrete_map=color_map,
            category_orders={"period": grouped["period"].unique().tolist()},
        )
        fig_tt.update_traces(
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:,}<extra></extra>",
        )
        fig_tt.update_layout(barmode="stack", xaxis_type="category", legend_title_text="Type")
        fig_tt.update_xaxes(title="")
        fig_tt.update_yaxes(title="Count")
        fig_tt = polish(fig_tt, 320, True)
    else:
        fig_tt = empty_fig("Need date + type data", 320)

    # Day of week — highlight peak with accent color
    if "_dow" in df.columns and df["_dow"].notna().any():
        dow_df = (
            df.dropna(subset=["_dow"]).groupby("_dow").size().reset_index(name="Count")
        )
        dow_df["Day"] = dow_df["_dow"].astype(int).apply(lambda i: DAYS_OF_WEEK[i])
        dow_df = dow_df.sort_values("_dow")
        peak_day   = dow_df.loc[dow_df["Count"].idxmax(), "Day"]
        bar_colors = [
            COLORS["blue"] if d == peak_day else COLORS["slate_light"]
            for d in dow_df["Day"]
        ]
        fig_dow = go.Figure(go.Bar(
            x=dow_df["Day"], y=dow_df["Count"],
            marker_color=bar_colors, marker_line_width=0,
            text=dow_df["Count"], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Events: %{y:,}<extra></extra>",
        ))
        fig_dow.update_layout(
            xaxis_type="category",
            annotations=[dict(
                text=f"Peak: {peak_day} ({int(dow_df.loc[dow_df['Day']==peak_day,'Count'].iloc[0]):,})",
                x=0.5, y=1.12, xref="paper", yref="paper", showarrow=False,
                font=dict(size=12, color=COLORS["text_3"]),
            )],
        )
        fig_dow.update_xaxes(title="")
        fig_dow.update_yaxes(title="Count")
        fig_dow = polish(fig_dow, 320, False)
    else:
        fig_dow = empty_fig("No day-of-week data", 320)

    # Resolution time by event type — median hours
    if "_res_minutes" in df.columns and "_type" in df.columns:
        tmp_res = df.dropna(subset=["_res_minutes", "_type"]).copy()
        tmp_res["_type"] = tmp_res["_type"].astype(str).str.strip().str.title()
        tmp_res["_res_minutes"] = pd.to_numeric(tmp_res["_res_minutes"], errors="coerce")
        tmp_res = tmp_res[tmp_res["_res_minutes"] >= 0].copy()
        if not tmp_res.empty:
            med = (
                tmp_res.groupby("_type")["_res_minutes"]
                .median().sort_values(ascending=False).head(10)
            )
            med_hours = (med / 60).reset_index()
            med_hours.columns = ["Type", "Hours"]
            med_hours = med_hours.sort_values("Hours")
            n = len(med_hours)
            bar_colors = [
                BAR_SEQ_TEAL[min(i, len(BAR_SEQ_TEAL) - 1)]
                for i in range(n)
            ]
            fig_res = go.Figure(go.Bar(
                x=med_hours["Hours"], y=med_hours["Type"],
                orientation="h", marker_color=bar_colors, marker_line_width=0,
                text=med_hours["Hours"].apply(lambda h: f"{h:.1f}h"),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Median: %{x:.1f} hrs<extra></extra>",
            ))
            fig_res.update_xaxes(title="Median Resolution (hours)")
            fig_res.update_yaxes(title="")
            fig_res = polish(fig_res, 320, False)
        else:
            fig_res = empty_fig("No resolution time data", 320)
    else:
        fig_res = empty_fig("No resolution time data", 320)

    return fig_m, fig_tt, fig_dow, fig_res


# ──────────────────────────────────────────────────────────────────────────────
# Hotspots — improved aesthetics
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("fig-route",           "figure"),
    Output("fig-route-heatmap",   "figure"),
    Output("fig-marker",          "figure"),
    Output("hotspot-filter-chip", "children"),
    Input("data-store",    "data"),
    Input("active-page",   "data"),
    Input("hotspot-month", "value"),
)
def update_hotspots(records, active, month_value):
    if active != "hotspots":
        return no_update, no_update, no_update, no_update

    df = from_store(records)
    if df.empty:
        return empty_fig(height=360), empty_fig(height=360), empty_fig(height=360), "No data"

    df_filt = filter_by_month(df, month_value)
    chip = [
        html.Span("Showing: "),
        html.Strong(month_label(month_value)),
        html.Span(f" — {len(df_filt):,} records"),
    ]
    if df_filt.empty:
        msg = f"No data for {month_label(month_value)}"
        return empty_fig(msg, 360), empty_fig(msg, 360), empty_fig(msg, 360), chip

    # Routes — neutral business palette
    route_df = count_col(df_filt, "_route", "Route", 12)
    if not route_df.empty:
        sorted_df = route_df.sort_values("Count")
        n = len(sorted_df)
        bar_colors = [
            BAR_SEQ_ROUTE[min(i, len(BAR_SEQ_ROUTE) - 1)]
            for i in range(n)
        ]
        fig_ro = go.Figure(go.Bar(
            x=sorted_df["Count"], y=sorted_df["Route"],
            orientation="h",
            marker_color=bar_colors,
            marker_line_width=0,
            hovertemplate="<b>%{y}</b><br>Alarms: %{x:,}<extra></extra>",
        ))
        fig_ro = polish(fig_ro, 360, False)
    else:
        fig_ro = empty_fig(height=360)

    # Route × Type heatmap — cool blue colorscale
    if {"_route", "_type"}.issubset(df_filt.columns) and df_filt["_route"].notna().any():
        sub        = df_filt.dropna(subset=["_route", "_type"]).copy()
        top_routes = sub["_route"].value_counts().head(8).index.tolist()
        top_types  = sub["_type"].value_counts().head(6).index.tolist()
        sub        = sub[sub["_route"].isin(top_routes) & sub["_type"].isin(top_types)]
        if sub.empty:
            fig_rh = empty_fig("Insufficient data for matrix", 360)
        else:
            pivot = (
                sub.groupby(["_route", "_type"]).size()
                .unstack(fill_value=0)
                .reindex(index=top_routes, columns=top_types, fill_value=0)
            )
            short_routes = [r if len(r) <= 32 else r[:30] + "…" for r in pivot.index]
            fig_rh = go.Figure(go.Heatmap(
                z=pivot.values, x=list(pivot.columns), y=short_routes,
                colorscale=HEATMAP_SCALE,
                colorbar=dict(title="Count", thickness=12, len=0.8),
                text=pivot.values, texttemplate="%{text:,}",
                textfont=dict(size=10),
                hovertemplate="<b>%{y}</b><br>%{x}: %{z:,}<extra></extra>",
            ))
            fig_rh.update_xaxes(title="Event Type", side="bottom")
            fig_rh.update_yaxes(title="", autorange="reversed")
            fig_rh = polish(fig_rh, 360, False)
            fig_rh.update_layout(margin=dict(l=10, r=10, t=20, b=40))
    else:
        fig_rh = empty_fig("Need route + type data", 360)

    # Top marker names — useful for finding exact recurring locations
    if "Marker name" in df_filt.columns:
        marker_df = _report_value_counts(df_filt, "Marker name", 12)
    elif "_marker" in df_filt.columns:
        marker_df = _report_value_counts(df_filt, "_marker", 12)
    else:
        marker_df = pd.DataFrame(columns=["Label", "Count"])

    if not marker_df.empty:
        sorted_marker = marker_df.sort_values("Count")
        n = len(sorted_marker)
        marker_colors = [
            BAR_SEQ_BLUE[min(i, len(BAR_SEQ_BLUE) - 1)]
            for i in range(n)
        ]
        fig_marker = go.Figure(go.Bar(
            x=sorted_marker["Count"],
            y=sorted_marker["Label"],
            orientation="h",
            marker_color=marker_colors,
            marker_line_width=0,
            text=sorted_marker["Count"],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Records: %{x:,}<extra></extra>",
        ))
        fig_marker.update_xaxes(title="Records")
        fig_marker.update_yaxes(title="")
        fig_marker = polish(fig_marker, 360, False)
    else:
        fig_marker = empty_fig("No marker-name data", 360)

    return fig_ro, fig_rh, fig_marker, chip


# ──────────────────────────────────────────────────────────────────────────────
# Position
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("position-table",   "data"),
    Output("position-table",   "columns"),
    Output("fig-position-bar", "figure"),
    Output("position-summary", "children"),
    Input("data-store",    "data"),
    Input("active-page",   "data"),
    Input("pos-bin-size",  "value"),
    Input("pos-max",       "value"),
    Input("pos-sort",      "value"),
    Input("hotspot-month", "value"),
)
def update_position(records, active, bin_size, max_pos, sort_by, month_value):
    if active != "hotspots":
        return no_update, no_update, no_update, no_update

    df = from_store(records)
    if df.empty:
        return [], [], empty_fig("No data", 320), []

    df = filter_by_month(df, month_value)
    if df.empty:
        return [], [], empty_fig(f"No data for {month_label(month_value)}", 320), []

    bin_size = _positive_int(bin_size, default=5000, minimum=100)
    max_pos  = _positive_float(max_pos, default=None)

    result = compute_position_ranges(df, bin_size=bin_size, max_position=max_pos)
    if result.empty:
        return [], [], empty_fig("No position data", 320), []

    if sort_by == "count":
        result = result.sort_values("Alarm Count", ascending=False)
    elif sort_by == "red":
        result = result.sort_values("Red Threats", ascending=False)
    else:
        result = result.sort_values("Range Start")

    total_alarms    = int(result["Alarm Count"].sum())
    total_ranges    = len(result)
    most_active_idx = result["Alarm Count"].idxmax()
    most_active_rng = result.loc[most_active_idx, "Position Range"]
    total_red       = int(result["Red Threats"].sum())

    def _stat(label, value, color=None) -> html.Div:
        val_style = {"fontSize": "18px", "fontWeight": "700", "color": COLORS["text"], "marginTop": "3px"}
        if color:
            val_style["color"] = color
        return html.Div(className="position-stat", children=[
            html.Div(label, className="position-stat-label"),
            html.Div(value, style=val_style),
        ])

    summary = [
        _stat("Bin Size",         f"{bin_size:,} m"),
        _stat("Ranges Generated", f"{total_ranges:,}"),
        _stat("Total Alarms",     f"{total_alarms:,}"),
        _stat("Busiest Range",    most_active_rng),
        _stat("High Threats",     f"{total_red:,}", COLORS["red"]),
    ]

    plot_df = result.sort_values("Range Start").copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df["Position Range"], y=plot_df["Alarm Count"],
        name="Total Alarms",
        marker_color=COLORS["blue"], marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Alarms: %{y}<extra></extra>",
    ))
    if (plot_df["Red Threats"] > 0).any():
        fig.add_trace(go.Bar(
            x=plot_df["Position Range"], y=plot_df["Red Threats"],
            name="Red Threats",
            marker_color=COLORS["red"], marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Red: %{y}<extra></extra>",
        ))
    fig.update_layout(barmode="overlay", xaxis_type="category")
    fig.update_xaxes(tickangle=-45)
    fig = polish(fig, 380, legend=True)

    display_df = result.drop(columns=["Range Start"])
    table_data = display_df.to_dict("records")
    table_cols = [
        {"name": "Position Range", "id": "Position Range"},
        {"name": "Alarm Count",    "id": "Alarm Count",   "type": "numeric"},
        {"name": "Red Threats",    "id": "Red Threats",   "type": "numeric"},
        {"name": "Avg Latitude",   "id": "Avg Latitude",  "type": "numeric"},
        {"name": "Avg Longitude",  "id": "Avg Longitude", "type": "numeric"},
        {"name": "Top Route",      "id": "Top Route"},
    ]
    return table_data, table_cols, fig, summary


@app.callback(
    Output("pos-download", "data"),
    Input("pos-download-btn", "n_clicks"),
    State("data-store",    "data"),
    State("pos-bin-size",  "value"),
    State("pos-max",       "value"),
    State("hotspot-month", "value"),
    prevent_initial_call=True,
)
def download_position_csv(_, records, bin_size, max_pos, month_value):
    df       = from_store(records)
    df       = filter_by_month(df, month_value)
    bin_size = _positive_int(bin_size, default=5000, minimum=100)
    max_pos  = _positive_float(max_pos, default=None)
    result   = compute_position_ranges(df, bin_size=bin_size, max_position=max_pos)
    if result.empty:
        return no_update
    result = result.drop(columns=["Range Start"])
    suffix = f"_{month_value}" if month_value and month_value != ALL_MONTHS_VALUE else ""
    return dcc.send_data_frame(
        result.to_csv, f"position_ranges_{bin_size}m{suffix}.csv",
        index=False, encoding="utf-8-sig",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Map
# ──────────────────────────────────────────────────────────────────────────────
def build_folium_map(
    df: pd.DataFrame,
    map_mode: str = "both",
    map_style: str = "esri",
    intensity: int = 18,
    return_obj: bool = False,
) -> object:
    geo_df = df.dropna(subset=["_lat", "_lon"]).copy()
    if geo_df.empty:
        if return_obj:
            return None
        return (
            "<div style='padding:40px;font-family:DM Sans,Arial;color:#64748b;text-align:center;'>"
            "No geo-coded data for this selection.</div>"
        )

    alarm_df    = geo_df[geo_df["_event_class"] == "Alarm"].copy()
    incident_df = geo_df[geo_df["_event_class"] == "Incident"].copy()

    map_df = (
        alarm_df if map_mode == "alarm" else
        incident_df if map_mode == "incident" else
        geo_df
    )

    if map_df.empty:
        if return_obj:
            return None
        return (
            "<div style='padding:40px;font-family:DM Sans,Arial;color:#64748b;text-align:center;'>"
            f"No <b>{map_mode}</b> records with GPS for this selection.</div>"
        )

    center_lat = float(map_df["_lat"].mean())
    center_lon = float(map_df["_lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11,
                   tiles=None, control_scale=True)

    esri_sat  = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                 "World_Imagery/MapServer/tile/{z}/{y}/{x}")
    esri_attr = ("Tiles © Esri — Esri, i-cubed, USDA, USGS, AEX, GeoEye, "
                 "Getmapping, Aerogrid, IGN, IGP, UPR-EGP")

    for tiles, attr, name, show in [
        (esri_sat, esri_attr, "Esri Satellite", map_style == "esri"),
        ("OpenStreetMap",     None, "OpenStreetMap", map_style == "osm"),
        ("CartoDB positron",  None, "Carto Light",   map_style == "carto"),
    ]:
        kw = dict(name=name, overlay=False, control=True, show=show)
        if attr:
            folium.TileLayer(tiles=tiles, attr=attr, **kw).add_to(m)
        else:
            folium.TileLayer(tiles=tiles, **kw).add_to(m)

    for tiles, attr, name in [
        ("https://server.arcgisonline.com/ArcGIS/rest/services/"
         "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
         "Labels © Esri", "Location Names"),
        ("https://server.arcgisonline.com/ArcGIS/rest/services/"
         "Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
         "Roads © Esri", "Road Names"),
    ]:
        folium.TileLayer(tiles=tiles, attr=attr, name=name,
                         overlay=True, control=True,
                         show=(map_style == "esri")).add_to(m)

    if map_mode in ("alarm", "both") and not alarm_df.empty:
        sample    = alarm_df.sample(min(4000, len(alarm_df)), random_state=42)
        heat_data = sample[["_lat", "_lon"]].dropna().values.tolist()
        HeatMap(
            heat_data, name="Alarm Heatmap",
            radius=intensity, blur=max(10, int(intensity * 1.5)),
            min_opacity=0.3,
            gradient={
                "0.2": "#93c5fd", "0.45": "#fcd34d",
                "0.7": "#f97316", "1.0": "#dc2626",
            },
        ).add_to(m)

    if map_mode in ("incident", "both") and not incident_df.empty:
        truncated = len(incident_df) > INCIDENT_MAP_LIMIT
        render_df = incident_df.head(INCIDENT_MAP_LIMIT) if truncated else incident_df
        acc_layer = folium.FeatureGroup(name="Incident Markers", show=True)

        if truncated:
            folium.Marker(
                location=[center_lat, center_lon],
                icon=folium.DivIcon(
                    html=(
                        f"<div style='background:#fef3c7;border:1px solid #f59e0b;"
                        f"border-radius:6px;padding:6px 10px;font-size:11px;"
                        f"font-family:DM Sans,Arial;color:#92400e;white-space:nowrap;'>"
                        f"⚠ Showing first {INCIDENT_MAP_LIMIT:,} of {len(incident_df):,}</div>"
                    ),
                    icon_size=(300, 36), icon_anchor=(150, -10),
                ),
            ).add_to(acc_layer)

        for _, row in render_df.iterrows():
            ts_str = "—"
            if "_dt" in row and pd.notna(row.get("_dt")):
                try:
                    ts_str = pd.to_datetime(row["_dt"]).strftime("%d %b %Y, %H:%M")
                except Exception:
                    ts_str = str(row.get("_dt"))
            popup_html = (
                "<div style='font-family:DM Sans,Arial;font-size:12px;'>"
                "<b style='color:#dc2626;font-size:14px;'>⚠ Incident</b><br>"
                f"<b>When:</b> {ts_str}<br>"
                f"<b>Type:</b> {row.get('_type','—')}<br>"
                f"<b>Threat:</b> {row.get('_threat','—')}<br>"
                f"<b>Route:</b> {row.get('_route','—')}<br>"
                f"<b>Position:</b> {row.get('_pos','—')} m"
                "</div>"
            )
            folium.Marker(
                location=[row["_lat"], row["_lon"]],
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"⚠ {row.get('_type','Incident')}",
                icon=folium.DivIcon(
                    html=(
                        '<div style="width:34px;height:31px;'
                        'filter:drop-shadow(0 1px 3px rgba(0,0,0,.5));">'
                        '<svg width="34" height="31" viewBox="0 0 34 31" xmlns="http://www.w3.org/2000/svg">'
                        '<path d="M17 2 L32 29 H2 Z" fill="white" stroke="#dc2626" stroke-width="3"/>'
                        '<text x="17" y="23" text-anchor="middle" font-size="14" font-weight="bold" fill="#dc2626">!</text>'
                        '</svg></div>'
                    ),
                    icon_size=(34, 31), icon_anchor=(17, 27),
                ),
            ).add_to(acc_layer)

        acc_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    if return_obj:
        return m
    return m.get_root().render()


def folium_map_to_jpeg_bytes(m, delay: int = 4, quality: int = 90) -> bytes:
    """Render a folium.Map to JPEG bytes.

    Requires Selenium + a headless browser driver such as Firefox/geckodriver
    or Chrome/chromedriver, plus Pillow.
    """
    from io import BytesIO
    from PIL import Image

    try:
        if hasattr(m, "_to_png"):
            png_bytes = m._to_png(delay=delay)
        else:
            print("[WARN] folium._to_png() not available; using Selenium fallback")
            import tempfile
            import time as _time
            from selenium import webdriver
            from selenium.webdriver.firefox.options import Options as FirefoxOptions

            with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tmp:
                tmp.write(m.get_root().render())
                tmp_path = tmp.name

            options = FirefoxOptions()
            options.add_argument("--headless")
            driver = webdriver.Firefox(options=options)
            try:
                driver.get(f"file://{tmp_path}")
                _time.sleep(delay)
                png_bytes = driver.get_screenshot_as_png()
            finally:
                driver.quit()
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        img = Image.open(BytesIO(png_bytes)).convert("RGB")
        out = BytesIO()
        img.save(out, format="JPEG", quality=quality)
        out.seek(0)
        return out.getvalue()
    except Exception as e:
        log_error("folium_map_to_jpeg_bytes", e, "ERROR")
        raise


def _stat_pill(label: str, value: str, accent: bool = False) -> html.Div:
    return html.Div(
        className=f"map-stat-pill{' accent' if accent else ''}",
        children=[f"{label}: ", html.Strong(str(value))],
    )


@app.callback(
    Output("map-mode",      "value"),
    Output("map-style",     "value"),
    Output("map-intensity", "value"),
    Input("map-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_map_controls(_):
    return "both", "esri", 18


@app.callback(
    Output("folium-map",      "srcDoc"),
    Output("map-stats",       "children"),
    Output("map-filter-chip", "children"),
    Input("data-store",    "data"),
    Input("active-page",   "data"),
    Input("map-mode",      "value"),
    Input("map-style",     "value"),
    Input("map-intensity", "value"),
    Input("map-month",     "value"),
)
def update_map(records, active, map_mode, map_style, map_intensity, month_value):
    if active != "map":
        return no_update, no_update, no_update

    df = from_store(records)
    map_mode      = (map_mode  or "both").lower()
    map_style     = (map_style or "esri").lower()
    map_intensity = int(map_intensity) if map_intensity else 18

    if df.empty:
        empty_html = "<div style='padding:40px;text-align:center;color:#94a3b8;'>No data.</div>"
        return empty_html, [_stat_pill("Status", "no data")], "No data"

    df_filt = filter_by_month(df, month_value)
    chip = [
        html.Span("Showing: "), html.Strong(month_label(month_value)),
        html.Span(f" — {len(df_filt):,} records"),
    ]

    if df_filt.empty:
        no_data_html = (
            f"<div style='padding:40px;text-align:center;color:#94a3b8;font-family:DM Sans,Arial;'>"
            f"No records for <b>{month_label(month_value)}</b>.</div>"
        )
        return (
            no_data_html,
            [_stat_pill("Month", month_label(month_value), accent=True),
             _stat_pill("Total mapped", "0")],
            chip,
        )

    geo_df = (
        df_filt.dropna(subset=["_lat", "_lon"])
        if {"_lat", "_lon"}.issubset(df_filt.columns) else df_filt.iloc[0:0]
    )
    alarm_count    = int((geo_df["_event_class"] == "Alarm").sum())    if "_event_class" in geo_df.columns else 0
    incident_count = int((geo_df["_event_class"] == "Incident").sum()) if "_event_class" in geo_df.columns else 0
    total_geo      = alarm_count + incident_count
    total_mapped   = (
        alarm_count if map_mode == "alarm" else
        incident_count if map_mode == "incident" else
        total_geo
    )
    pct = (total_mapped / total_geo * 100) if total_geo else 0

    stat_pills = [
        _stat_pill("Mode",      map_mode.capitalize()),
        _stat_pill("Month",     month_label(month_value), accent=(month_value != ALL_MONTHS_VALUE)),
        _stat_pill("Alarms",    f"{alarm_count:,}"    if map_mode in ("alarm", "both")    else "0"),
        _stat_pill("Incidents", f"{incident_count:,}" if map_mode in ("incident", "both") else "0"),
        _stat_pill("Mapped",    f"{total_mapped:,} ({pct:.1f}%)", accent=True),
    ]

    map_html = build_folium_map(
        df_filt, map_mode=map_mode, map_style=map_style, intensity=map_intensity
    )
    return map_html, stat_pills, chip

# ── Map: download static JPEG file
@app.callback(
    Output("map-download-file",   "data"),
    Output("map-download-status", "children"),
    Input("map-download-btn",     "n_clicks"),
    Input("map-download-img-btn", "n_clicks"),
    State("data-store",    "data"),
    State("map-mode",      "value"),
    State("map-style",     "value"),
    State("map-intensity", "value"),
    State("map-month",     "value"),
    prevent_initial_call=True,
)
def download_map_file(n1, n2, records, map_mode, map_style, map_intensity, month_value):
    df = from_store(records)
    if df.empty:
        return no_update, html.Span(
            "⚠ No data to download.",
            style={"color": COLORS["red"], "fontWeight": "600"},
        )

    df_filt = filter_by_month(df, month_value)
    if df_filt.empty:
        return no_update, html.Span(
            f"⚠ No records for {month_label(month_value)}.",
            style={"color": COLORS["red"], "fontWeight": "600"},
        )

    map_mode      = (map_mode  or "both").lower()
    map_style     = (map_style or "esri").lower()
    map_intensity = int(map_intensity) if map_intensity else 18

    # Build the folium Map OBJECT, not HTML, so it can be rasterized to JPEG.
    m = build_folium_map(
        df_filt,
        map_mode=map_mode,
        map_style=map_style,
        intensity=map_intensity,
        return_obj=True,
    )
    if m is None:
        return no_update, html.Span(
            "⚠ No mappable records for this selection.",
            style={"color": COLORS["red"], "fontWeight": "600"},
        )

    # Early check before attempting export
    if not _PILLOW_OK or not _SELENIUM_OK:
        missing = []
        if not _PILLOW_OK:
            missing.append("Pillow")
        if not _SELENIUM_OK:
            missing.append("Selenium")
        return no_update, html.Span(
            f"⚠ JPEG export unavailable. Missing: {', '.join(missing)}. "
            "Also ensure Firefox + geckodriver are installed.",
            style={"color": COLORS["amber_dark"], "fontWeight": "600"},
        )

    try:
        jpeg_bytes = folium_map_to_jpeg_bytes(m)
    except Exception as e:
        traceback.print_exc()
        return no_update, html.Span(
            f"⚠ JPEG export failed: {e}",
            style={"color": COLORS["red"], "fontWeight": "600"},
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    suffix    = "all_months" if (not month_value or month_value == ALL_MONTHS_VALUE) else str(month_value)
    filename  = f"DAS_Map_{suffix}_{timestamp}.jpeg"

    return (
        dcc.send_bytes(jpeg_bytes, filename),
        html.Span(f"✓ Map downloaded as {filename}",
                  style={"color": COLORS["green"], "fontWeight": "600"}),
    )

# ──────────────────────────────────────────────────────────────────────────────
# Prediction callbacks
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("pred-store", "data"),
    Input("active-page", "data"),
    Input("data-store",  "data"),
    State("pred-store",  "data"),
)
def compute_predictions(active, records, existing):
    if active != "prediction":
        return no_update
    if MODEL is None:
        return []
    df = from_store(records)
    if df.empty:
        return []
    try:
        result = predict_next_month_table(df)
        if result.empty:
            return []

        # IMPORTANT:
        # The ML feature pipeline intentionally uses lowercase `route` and `type`
        # because MODEL_FEATURES was trained with those lowercase names.
        # The dashboard UI/callbacks use `Route` and `Type`.
        # Rename only after prediction so the model input stays unchanged, but
        # pred-store still contains the columns expected by update_pred_result(),
        # update_pred_table(), CSV download, and the dropdown fallback logic.
        result = result.rename(columns={"route": "Route", "type": "Type"})

        keep = [
            "Route", "Type", "position_bin", "position_range",
            "alarm_count", "prev_month_count", "past_total_count",
            "past_months_active", "past_avg_count",
            "avg_lat", "avg_lon",
            "repeat_probability", "pred_repeat_next_month", "risk_level",
        ]
        keep = [c for c in keep if c in result.columns]
        return result[keep].to_dict("records")
    except Exception as e:
        print(f"[predict] error: {e}")
        traceback.print_exc()
        return []


@app.callback(
    Output("model-status-banner", "children"),
    Output("model-meta-grid",     "children"),
    Input("active-page", "data"),
    Input("pred-store",  "data"),
)
def update_model_status(active, pred_records):
    if active != "prediction":
        return no_update, no_update

    if MODEL is None:
        banner = html.Div(className="model-status-banner error", children=[
            html.Div("!", className="model-status-banner-icon"),
            html.Div([
                html.Strong("Model not loaded. "),
                html.Span(MODEL_ERROR or "Unknown error."),
                html.Br(),
                html.Span(
                    "Run train_repeated_alarm_model_v4.py first, then place "
                    "repeated_alarm_model.joblib and repeated_alarm_metadata.json "
                    "next to this app file.",
                    style={"fontSize": "11px", "opacity": "0.85"},
                ),
            ]),
        ])
        return banner, []

    n_preds = len(pred_records) if pred_records else 0
    high_n  = sum(1 for r in (pred_records or []) if r.get("risk_level") == "High")

    if pred_records is None:
        banner = html.Div(className="model-status-banner ok", children=[
            html.Div("…", className="model-status-banner-icon"),
            html.Div(html.Strong("Computing predictions…")),
        ])
    elif n_preds == 0:
        banner = html.Div(className="model-status-banner error", children=[
            html.Div("!", className="model-status-banner-icon"),
            html.Div([
                html.Strong("Model loaded but no predictions produced. "),
                html.Span("The latest month may not have scoreable patterns."),
            ]),
        ])
    else:
        banner = html.Div(className="model-status-banner ok", children=[
            html.Div("✓", className="model-status-banner-icon"),
            html.Div([
                html.Strong("Model ready. "),
                html.Span(
                    f"Scored {n_preds:,} patterns; "
                    f"{high_n:,} flagged High risk for next month."
                ),
            ]),
        ])

    def _cell(label, value):
        return html.Div(className="model-meta-cell", children=[
            html.Div(label, className="label"),
            html.Div(value, className="value"),
        ])

    def _fmt(v):
        if v is None:
            return "—"
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return "—"

    meta  = MODEL_META or {}
    cells = [
        _cell("Algorithm",     str(meta.get("winning_model", "—"))),
        _cell("CV F1",         _fmt(meta.get("cv_f1"))),
        _cell("Test F1",       _fmt(meta.get("test_f1"))),
        _cell("Test Accuracy", _fmt(meta.get("test_accuracy"))),
        _cell("ROC AUC",       _fmt(meta.get("test_roc_auc"))),
        _cell("Threshold",     f"{meta.get('chosen_threshold', 0.5):.2f}"),
        _cell("Calibration",   str(meta.get("calibration_method", "—"))),
        _cell("Trainer Ver.",  str(meta.get("trainer_version", "—"))),
        _cell("Test Month",    str(meta.get("test_month", "—"))),
        _cell("Bin Size",      f"{meta.get('position_bin_size', POSITION_BIN_SIZE):,} m"),
    ]
    return banner, cells


@app.callback(
    Output("pred-route",    "options"),
    Output("pred-type",     "options"),
    Output("pred-position", "options"),
    Input("pred-store",  "data"),
    Input("data-store",  "data"),
    Input("pred-route",  "value"),
    Input("pred-type",   "value"),
)
def update_pred_dropdowns(pred_records, raw_records, selected_route, selected_type):
    raw_df  = from_store(raw_records)
    pred_df = pd.DataFrame(pred_records) if pred_records else pd.DataFrame()

    if raw_df.empty and pred_df.empty:
        return [], [], []

    if not raw_df.empty and {"_route", "_type"}.issubset(raw_df.columns):
        base = raw_df.dropna(subset=["_route", "_type"]).copy()
        base["Route"] = base["_route"].astype(str).str.strip()
        base["Type"]  = base["_type"].astype(str).str.strip()
        base["Position (m)"] = (
            pd.to_numeric(base["_pos"], errors="coerce")
            if "_pos" in base.columns else np.nan
        )
    elif not pred_df.empty:
        base = pred_df.copy()
        base["Position (m)"] = (
            pd.to_numeric(base["position_bin"], errors="coerce") * POSITION_BIN_SIZE
            if "position_bin" in base.columns else np.nan
        )
    else:
        return [], [], []

    base = base[
        (base["Route"].notna()) & (base["Type"].notna())
        & (base["Route"].astype(str).str.strip() != "")
        & (base["Type"].astype(str).str.strip() != "")
        & (base["Route"].astype(str).str.lower() != "nan")
        & (base["Type"].astype(str).str.lower()  != "nan")
    ].copy()

    if base.empty:
        return [], [], []

    routes     = sorted(base["Route"].dropna().astype(str).unique().tolist())
    route_opts = [{"label": r, "value": r} for r in routes]

    df_t      = base if not selected_route else base[base["Route"] == selected_route]
    types     = sorted(df_t["Type"].dropna().astype(str).unique().tolist())
    type_opts = [{"label": t, "value": t} for t in types]

    df_p = (df_t if not selected_type else df_t[df_t["Type"] == selected_type]).copy()

    # ── Priority: use pred-store position_range labels directly so they
    #    always match what was scored, regardless of POSITION_BIN_SIZE changes.
    pred_df = pd.DataFrame(pred_records) if pred_records else pd.DataFrame()
    if not pred_df.empty and {"Route", "Type", "position_range", "position_bin"}.issubset(pred_df.columns):
        sub_pred = pred_df.copy()
        if selected_route:
            sub_pred = sub_pred[sub_pred["Route"] == selected_route]
        if selected_type:
            sub_pred = sub_pred[sub_pred["Type"] == selected_type]
        if not sub_pred.empty:
            range_df = (
                sub_pred[["position_bin", "position_range"]]
                .drop_duplicates()
                .sort_values("position_bin")
            )
            pos_opts = [
                {"label": r["position_range"], "value": r["position_range"]}
                for _, r in range_df.iterrows()
            ]
            return route_opts, type_opts, pos_opts

    # ── Fallback: build from raw data when pred-store is unavailable
    df_p = df_p.dropna(subset=["Position (m)"])
    df_p = df_p[df_p["Position (m)"] >= 0].copy()

    if df_p.empty:
        return route_opts, type_opts, []

    df_p["_pos_num"]       = pd.to_numeric(df_p["Position (m)"], errors="coerce")
    df_p                   = df_p.dropna(subset=["_pos_num"]).copy()
    df_p["position_bin"]   = (df_p["_pos_num"] // POSITION_BIN_SIZE).astype(float).astype(int)
    df_p["position_range"] = df_p["position_bin"].apply(
        lambda b: _pos_range_label(int(b), POSITION_BIN_SIZE)
    )

    range_df = (
        df_p[["position_bin", "position_range"]]
        .drop_duplicates()
        .sort_values("position_bin")
    )
    pos_opts = [
        {"label": r["position_range"], "value": r["position_range"]}
        for _, r in range_df.iterrows()
    ]
    return route_opts, type_opts, pos_opts


@app.callback(
    Output("pred-result", "children"),
    Input("pred-route",    "value"),
    Input("pred-type",     "value"),
    Input("pred-position", "value"),
    State("pred-store",    "data"),
)
def update_pred_result(route, ev_type, pos_range, pred_records):
    if MODEL is None:
        return html.Div([
            html.Div("Model unavailable", className="prediction-badge red"),
            html.Div("Train and load the model to enable predictions.",
                     className="prediction-note muted"),
        ])
    if not pred_records:
        return html.Div([
            html.Div("No predictions", className="prediction-badge amber"),
            html.Div("Predictions could not be computed.", className="prediction-note muted"),
        ])
    if not (route and ev_type and pos_range):
        return html.Div([
            html.Div("Awaiting input", className="prediction-badge amber"),
            html.Div("Select Route, Type, and Position Range to see the prediction.",
                     className="prediction-note muted"),
        ])

    df = pd.DataFrame(pred_records)
    match = df[
        (df["Route"] == route)
        & (df["Type"] == ev_type)
        & (df["position_range"] == pos_range)
    ]

    if match.empty:
        return html.Div([
            html.Div("No latest-month data", className="prediction-badge amber"),
            html.Div(
                "This pattern exists in the dataset but not in the latest scored month.",
                className="prediction-note",
            ),
        ])

    row  = match.iloc[0]
    prob = float(row["repeat_probability"])
    risk = str(row["risk_level"])
    badge_tone = "red" if risk == "High" else ("amber" if risk == "Medium" else "green")

    def _num(key, default=0.0):
        try:
            if key not in row.index:
                return default
            v = row[key]
            if v is None:
                return default
            f = float(v)
            return default if (math.isnan(f) or math.isinf(f)) else f
        except (TypeError, ValueError):
            return default

    factors = [
        ("Alarms in latest month",   f"{int(_num('alarm_count')):,}",
         "high" if _num("alarm_count") >= 5 else ("medium" if _num("alarm_count") >= 2 else "low")),
        ("Previous month count",     f"{int(_num('prev_month_count')):,}",
         "high" if _num("prev_month_count") >= 3 else ("medium" if _num("prev_month_count") >= 1 else "low")),
        ("Past months active",       f"{int(_num('past_months_active')):,}",
         "high" if _num("past_months_active") >= 3 else ("medium" if _num("past_months_active") >= 1 else "low")),
        ("Past total alarms",        f"{int(_num('past_total_count')):,}",
         "high" if _num("past_total_count") >= 10 else ("medium" if _num("past_total_count") >= 3 else "low")),
        ("Past avg / active month",  f"{_num('past_avg_count'):.2f}",
         "medium" if _num("past_avg_count") >= 2 else "low"),
    ]

    return html.Div([
        html.Div(f"{risk} Risk", className=f"prediction-badge {badge_tone}"),
        html.Div(f"{prob * 100:.1f}%", className="prediction-score"),
        html.Div(
            f"Probability this pattern repeats next month. "
            f"Decision threshold: {MODEL_THRESHOLD:.2f}.",
            className="prediction-note muted",
        ),
        html.Div(className="factor-list", children=[
            html.Div(className=f"factor-row {tone}", children=[
                html.Div(name,  className="factor-name"),
                html.Div(value, className="factor-value"),
            ])
            for name, value, tone in factors
        ]),
    ])


@app.callback(
    Output("pred-table",   "data"),
    Output("pred-table",   "columns"),
    Input("pred-store",       "data"),
    Input("pred-topn",        "value"),
    Input("pred-risk-filter", "value"),
)
def update_pred_table(pred_records, topn, risk_filter):
    if not pred_records:
        return [], []

    df = pd.DataFrame(pred_records)
    if risk_filter == "High":
        df = df[df["risk_level"] == "High"]
    elif risk_filter == "HM":
        df = df[df["risk_level"].isin(["High", "Medium"])]

    if df.empty:
        return [], []

    df = df.sort_values("repeat_probability", ascending=False).head(int(topn or 25)).copy()
    df["repeat_probability"] = (df["repeat_probability"].astype(float) * 100).round(2)

    if "pred_repeat_next_month" in df.columns:
        df["pred_repeat_next_month"] = df["pred_repeat_next_month"].map(
            lambda v: "Yes" if (v == 1 or v == "1" or v is True) else "No"
        )

    display_order = [
        "Route", "Type", "position_range",
        "alarm_count", "prev_month_count", "past_total_count",
        "repeat_probability", "risk_level", "pred_repeat_next_month",
    ]
    display_order = [c for c in display_order if c in df.columns]
    df = df[display_order]

    nice = {
        "position_range":         "Position Range",
        "alarm_count":            "Latest Month",
        "prev_month_count":       "Prev Month",
        "past_total_count":       "Past Total",
        "repeat_probability":     "Probability (%)",
        "risk_level":             "Risk",
        "pred_repeat_next_month": "Will Repeat?",
    }
    numeric_cols = {"alarm_count", "prev_month_count", "past_total_count", "repeat_probability"}
    cols = []
    for c in df.columns:
        col_def = {"name": nice.get(c, c.replace("_", " ").title()), "id": c}
        if c in numeric_cols:
            col_def["type"] = "numeric"
        cols.append(col_def)

    return df.to_dict("records"), cols


@app.callback(
    Output("pred-download", "data"),
    Input("pred-download-btn", "n_clicks"),
    State("pred-store",        "data"),
    State("pred-risk-filter",  "value"),
    prevent_initial_call=True,
)
def download_pred_csv(_, pred_records, risk_filter):
    if not pred_records:
        return no_update
    df = pd.DataFrame(pred_records)
    if risk_filter == "High":
        df = df[df["risk_level"] == "High"]
    elif risk_filter == "HM":
        df = df[df["risk_level"].isin(["High", "Medium"])]
    df = df.sort_values("repeat_probability", ascending=False)
    return dcc.send_data_frame(
        df.to_csv, "next_month_predictions.csv", index=False, encoding="utf-8-sig",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────────────────────────────────────
def _visible_df(df: pd.DataFrame, limit: int) -> tuple[list, list]:
    vis = df.drop(
        columns=[c for c in df.columns if c.startswith("_")], errors="ignore"
    ).head(limit).copy()
    for col in vis.select_dtypes(include=["datetimetz"]).columns:
        vis[col] = vis[col].dt.tz_localize(None)
    for col in vis.select_dtypes(include=["datetime64"]).columns:
        vis[col] = vis[col].dt.strftime("%Y-%m-%d %H:%M").where(vis[col].notna(), "—")
    return vis.to_dict("records"), [{"name": c, "id": c} for c in vis.columns]


@app.callback(
    Output("data-preview",       "data"),
    Output("data-preview",       "columns"),
    Output("data-preview-title", "children"),
    Output("data-preview-note",  "children"),
    Input("data-store", "data"),
)
def update_preview_table(records):
    df = from_store(records)
    if df.empty:
        return [], [], "Latest Records", "No data loaded."

    if "_event_class" in df.columns:
        acc = df[df["_event_class"].astype(str).str.lower() == "incident"]
    else:
        acc = df.iloc[0:0]

    if not acc.empty:
        acc = acc.sort_values("_dt", ascending=False) if "_dt" in acc.columns else acc
        data, cols = _visible_df(acc, 10)
        return data, cols, "Latest Incidents", f"10 most recent ({len(acc):,} total incidents)"

    fallback = df.sort_values("_dt", ascending=False) if "_dt" in df.columns else df
    data, cols = _visible_df(fallback, 10)
    return data, cols, "Latest Records", "No incidents found — showing 10 most recent records."


# ──────────────────────────────────────────────────────────────────────────────
# Report callbacks
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("report-sections", "value"),
    Input("report-select-all",  "n_clicks"),
    Input("report-select-none", "n_clicks"),
    State("report-sections",    "value"),
    prevent_initial_call=True,
)
def toggle_all_sections(n_all, n_none, current):
    trigger = callback_context.triggered_id
    if trigger == "report-select-all":
        return ALL_SECTION_VALUES
    if trigger == "report-select-none":
        return []
    return current


@app.callback(
    Output("report-preview-content", "children"),
    Input("active-page",   "data"),
    Input("data-store",    "data"),
    Input("report-month",  "value"),
    Input("report-sections", "value"),
)
def update_report_preview(active, records, month_value, sections):
    if active != "explorer":
        return no_update

    df = from_store(records)
    df_f = filter_by_month(df, month_value)
    total, alarms, incidents = _report_event_split(df_f)
    red_threats = 0
    if "_threat_key" in df_f.columns:
        red_threats = int(df_f["_threat_key"].astype(str).str.lower().eq("red").sum())

    sections_selected = len(sections) if sections else 0
    section_labels = {s["value"]: s["label"] for s in REPORT_SECTIONS}
    included = [section_labels.get(s, s) for s in (sections or [])]

    kpi_cells = [
        html.Div(className="report-kpi-cell", children=[
            html.Div(f"{total:,}",        className="val"),
            html.Div("Total Records",     className="lbl"),
        ]),
        html.Div(className="report-kpi-cell", children=[
            html.Div(f"{alarms:,}",       className="val"),
            html.Div("Alarms",            className="lbl"),
        ]),
        html.Div(className="report-kpi-cell", children=[
            html.Div(f"{incidents:,}",    className="val"),
            html.Div("Incidents",         className="lbl"),
        ]),
        html.Div(className="report-kpi-cell", children=[
            html.Div(f"{red_threats:,}",  className="val",
                     style={"color": COLORS["red"]}),
            html.Div("Red Threats",       className="lbl"),
        ]),
    ]

    return html.Div([
        html.Div([
            html.Span("Period: "),
            html.Strong(month_label(month_value)),
        ], style={"fontSize": "13px", "color": COLORS["text_2"], "marginBottom": "10px"}),

        html.Div(className="report-preview-kpi", children=kpi_cells),

        html.Div(
            f"{sections_selected} section{'s' if sections_selected != 1 else ''} selected",
            style={"fontSize": "11.5px", "color": COLORS["text_3"],
                   "marginTop": "12px", "fontWeight": "600"},
        ),
        html.Ul(
            [html.Li(lbl, style={"fontSize": "11.5px", "color": COLORS["text_2"],
                                 "padding": "2px 0"}) for lbl in included],
            style={"paddingLeft": "18px", "marginTop": "6px"},
        ) if included else html.Div(
            "No sections selected — select at least one section.",
            style={"fontSize": "11.5px", "color": COLORS["red"], "marginTop": "8px"},
        ),
    ])


@app.callback(
    Output("report-download", "data"),
    Output("report-status",   "children"),
    Input("report-generate-btn", "n_clicks"),
    State("data-store",      "data"),
    State("report-month",    "value"),
    State("report-sections", "value"),
    State("report-bin-size", "value"),
    prevent_initial_call=True,
)
def generate_report(_, records, month_value, sections, bin_size):
    df = from_store(records)
    if df.empty:
        return no_update, html.Span("No data available.", className="report-status error")
    if not sections:
        return no_update, html.Span("Select at least one section.", className="report-status error")

    df_f = filter_by_month(df, month_value)
    if df_f.empty:
        return (
            no_update,
            html.Span(f"No records for {month_label(month_value)}.",
                      className="report-status error"),
        )

    bin_size = _validate_numeric_input(
        bin_size, default=5000, minimum=100, maximum=50000, step=100
    )

    period = month_label(month_value)
    try:
        pdf_bytes = build_report_pdf(
            df_f, sections,
            period_label=period,
            bin_size=bin_size,
            data_source=DATA_SOURCE_FILENAME,
        )
    except Exception as e:
        traceback.print_exc()
        return (
            no_update,
            html.Span(f"Report failed: {e}", className="report-status error"),
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    suffix    = "all_months" if month_value == ALL_MONTHS_VALUE else str(month_value)
    filename  = f"DAS_Report_{suffix}_{timestamp}.pdf"

    return (
        dcc.send_bytes(pdf_bytes, filename),
        html.Span(f"✓ Report ready — {period} — {len(sections)} sections.",
                  className="report-status ok"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[startup] Data source : {DATA_SOURCE_FILENAME}")
    print(f"[startup] Model loaded: {MODEL is not None}")
    print(f"[startup] Pillow OK   : {_PILLOW_OK}")
    print(f"[startup] Selenium OK : {_SELENIUM_OK}")
    if MODEL is None:
        print(f"[startup] Reason      : {MODEL_ERROR}")
    else:
        meta = MODEL_META or {}
        print(f"[startup] Algorithm   : {meta.get('winning_model','—')}")
        print(f"[startup] Calibration : {meta.get('calibration_method','—')}")
        print(f"[startup] Trainer ver : {meta.get('trainer_version','—')}")
        print(f"[startup] Threshold   : {MODEL_THRESHOLD}")
        print(f"[startup] Features    : {len(MODEL_FEATURES)}")

    _debug = os.getenv("DAS_DEBUG", "0") == "1"
    app.run(
        debug=_debug,
        dev_tools_ui=_debug,
        dev_tools_hot_reload=_debug,
        dev_tools_hot_reload_interval=1000,
        dev_tools_hot_reload_watch_interval=0.5,
        host=os.getenv("DAS_HOST", "127.0.0.1"),
        port=int(os.getenv("DAS_PORT", "8050")),
    )