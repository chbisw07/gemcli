# tools/charting.py
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt  # NOTE: matplotlib only; no seaborn
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


@dataclass
class PlotResult:
    image_path: str
    rows: int
    cols: int
    columns: List[str]
    kind: str
    x: Optional[str]
    y: List[str]
    resample: Optional[str]
    agg: Optional[str]
    title: Optional[str]


# ------------------------- helpers -------------------------

_NUMERIC_KINDS = {"i", "u", "f", "c"}  # int, uint, float, complex
_TS_FORMATTERS = {
    "M": mdates.DateFormatter("%Y-%m"),
    "W": mdates.DateFormatter("%Y-%m-%d"),
    "D": mdates.DateFormatter("%Y-%m-%d"),
}

def _is_numeric(s: pd.Series) -> bool:
    return s.dtype.kind in _NUMERIC_KINDS

def _ensure_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        return pd.to_datetime(series, errors="raise", utc=False, infer_datetime_format=True)
    except Exception:
        return series  # leave as-is if it truly isn't datetime parseable

def _pick_x(df: pd.DataFrame, x: Optional[str]) -> Tuple[pd.Index, Optional[str]]:
    if x and x in df.columns:
        sx = _ensure_datetime(df[x]) if "date" in x.lower() or "time" in x.lower() else df[x]
        return pd.Index(sx), x
    # heuristic: prefer first datetime-looking column, else index, else first column
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            sx = _ensure_datetime(df[c])
            return pd.Index(sx), c
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index, None
    # fallback: first column
    c0 = df.columns[0]
    sx = _ensure_datetime(df[c0]) if "date" in c0.lower() or "time" in c0.lower() else df[c0]
    return pd.Index(sx), c0

def _pick_y(df: pd.DataFrame, y: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if y is None:
        # numeric columns only
        cols = [c for c in df.columns if _is_numeric(df[c])]
        return cols or [c for c in df.columns if c != df.columns[0]][:1]
    if isinstance(y, str):
        y = [c.strip() for c in y.split(",") if c.strip()]
    return [c for c in y if c in df.columns]

def _resample(df: pd.DataFrame, freq: str, agg: str) -> pd.DataFrame:
    agg = (agg or "mean").lower()
    fn = {"mean": "mean", "sum": "sum", "min": "min", "max": "max", "median": "median"}.get(agg, "mean")
    return getattr(df.resample(freq), fn)()

def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in (s or "").strip()) or "chart"

def _mk_out_path(out_dir: Path, title: Optional[str]) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = f"{_slug(title) or 'chart'}-{ts}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


# ------------------------- core API -------------------------

def draw_chart_from_csv(
    csv_path: Union[str, Path],
    *,
    kind: str = "line",
    x: Optional[str] = None,
    y: Optional[Union[str, Sequence[str]]] = None,
    parse_dates: bool = True,
    resample: Optional[str] = None,    # e.g., "D", "W", "M"
    agg: Optional[str] = "mean",
    title: Optional[str] = None,
    out_dir: Union[str, Path] = "charts",
    width: int = 1200,
    height: int = 800,
    dpi: int = 144,
    grid: bool = True,
) -> PlotResult:
    """
    Draw a chart from a CSV. Saves a PNG locally and returns metadata.
    - kind: line|bar|scatter|hist
    - x: column to use for x-axis (optional; auto-picks date-like column)
    - y: one or many columns (if None -> all numeric)
    - resample: "D"/"W"/"M" when x is datetime-like (optional)
    - agg: mean|sum|min|max|median when resampling
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV has no rows")

    return _draw_chart_from_df(
        df, kind=kind, x=x, y=y, parse_dates=parse_dates, resample=resample, agg=agg,
        title=title or csv_path.stem, out_dir=out_dir, width=width, height=height, dpi=dpi, grid=grid
    )


def draw_chart_from_json_records(
    data_json: str,
    *,
    kind: str = "line",
    x: Optional[str] = None,
    y: Optional[Union[str, Sequence[str]]] = None,
    parse_dates: bool = True,
    resample: Optional[str] = None,
    agg: Optional[str] = "mean",
    title: Optional[str] = None,
    out_dir: Union[str, Path] = "charts",
    width: int = 1200,
    height: int = 800,
    dpi: int = 144,
    grid: bool = True,
) -> PlotResult:
    """
    Draw a chart from a JSON string of records: '[{"date":"...", "price": ...}, ...]'.
    Saves PNG and returns metadata.
    """
    try:
        recs = json.loads(data_json or "[]")
    except Exception as e:
        raise ValueError(f"Invalid JSON records: {e}")
    if not isinstance(recs, list) or not recs:
        raise ValueError("JSON records must be a non-empty list of objects")
    df = pd.DataFrame.from_records(recs)
    return _draw_chart_from_df(
        df, kind=kind, x=x, y=y, parse_dates=parse_dates, resample=resample, agg=agg,
        title=title or "chart", out_dir=out_dir, width=width, height=height, dpi=dpi, grid=grid
    )


def _draw_chart_from_df(
    df: pd.DataFrame,
    *,
    kind: str,
    x: Optional[str],
    y: Optional[Union[str, Sequence[str]]],
    parse_dates: bool,
    resample: Optional[str],
    agg: Optional[str],
    title: Optional[str],
    out_dir: Union[str, Path],
    width: int,
    height: int,
    dpi: int,
    grid: bool,
) -> PlotResult:
    # detect x/y
    X, x_name = _pick_x(df, x)
    y_cols = _pick_y(df, y)

    # Build working frame
    work = pd.DataFrame(index=pd.Index(range(len(df))))  # placeholder index; we’ll set below
    work[x_name or "x"] = X.values
    for col in y_cols:
        work[col] = df[col].values

    # If datetime index is desired for resample, set it
    idx = work[x_name or "x"]
    is_dt = pd.api.types.is_datetime64_any_dtype(idx)
    if parse_dates and not is_dt:
        try:
            idx = pd.to_datetime(idx, errors="raise", utc=False, infer_datetime_format=True)
            is_dt = True
        except Exception:
            pass

    if is_dt:
        work = work.set_index(idx)
        # drop the helper x column if duplicated
        if x_name and x_name in work.columns:
            work = work.drop(columns=[x_name], errors="ignore")
        if resample:
            work = _resample(work[y_cols], resample, agg)  # only numeric cols for resample
    else:
        # no resample on non-datetime x
        work = work.set_index(idx)

    # Plot
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    k = (kind or "line").lower()
    if k == "line":
        work[y_cols].plot(ax=ax)
    elif k == "bar":
        work[y_cols].plot(kind="bar", ax=ax)
    elif k == "scatter":
        # scatter supports one y; if many, scatter first
        y0 = y_cols[0]
        ax.scatter(work.index.values, work[y0].values, s=16)
    elif k == "hist":
        work[y_cols].plot(kind="hist", ax=ax, bins=30, alpha=0.7)
    else:
        work[y_cols].plot(ax=ax)  # fallback to line

    # Axis formatting
    if is_dt:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fmt = _TS_FORMATTERS.get(resample or "", mdates.AutoDateFormatter(mdates.AutoDateLocator()))
        ax.xaxis.set_major_formatter(fmt)
        fig.autofmt_xdate()

    ax.grid(grid)
    ax.set_title(title or "Chart")
    ax.set_xlabel(x_name or ("index" if not is_dt else "date"))
    if len(y_cols) == 1:
        ax.set_ylabel(y_cols[0])
    else:
        ax.set_ylabel("value")

    out_dir = Path(out_dir)
    out_path = _mk_out_path(out_dir, title or (x_name or "chart"))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return PlotResult(
        image_path=str(out_path),
        rows=int(work.shape[0]),
        cols=int(work.shape[1]),
        columns=list(work.columns),
        kind=k,
        x=x_name,
        y=list(y_cols),
        resample=resample,
        agg=agg,
        title=title,
    ).__dict__
