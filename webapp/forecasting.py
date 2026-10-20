"""Timestral 7-day temperature forecast and last week's track record, for any location.

Method (validated in ../backtest_covariates.py): Timestral reads the last 60 days of hourly
temperature plus the weather service's forecast as a known-future input, and forecasts 168 hours.
The track record reruns the same method from exactly 7 days ago, using only forecasts that had
already been issued by then, and scores it against what was observed.
"""

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd

MODEL_ID = os.environ.get("TIMESTRAL_MODEL", "mistralai/Timestral-28M")
HORIZON = 168  # hours = 7 days
CONTEXT = 60 * 24  # hours of history the model reads
PAST_DAYS = 92  # Open-Meteo's maximum; covers the context for both the forecast and last week's rerun
RECENT_HOURS = 48  # observed hours shown before "now"
MAX_LEAD_DAY = 7

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

_http = httpx.Client(timeout=30, headers={"User-Agent": "timestral-weather-demo"})


def _get(url: str, params: dict) -> dict:
    r = _http.get(url, params=params)
    r.raise_for_status()
    return r.json()


# ---- model ----------------------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            import torch
            from timestral import TimestralForecaster

            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
            _model = TimestralForecaster.from_pretrained(MODEL_ID)
    return _model


def model_name() -> str:
    return MODEL_ID.split("/")[-1]


# ---- data -----------------------------------------------------------------------------------------
def geocode(query: str, count: int = 6) -> list[dict]:
    data = _get(GEOCODE_URL, {"name": query, "count": count, "language": "en", "format": "json"})
    return [
        {"name": r["name"], "region": r.get("admin1"), "country": r.get("country"),
         "lat": r["latitude"], "lon": r["longitude"]}
        for r in data.get("results", []) or []
    ]


def _hourly_frame(payload: dict, cols: list[str]) -> pd.DataFrame:
    h = payload["hourly"]
    return pd.DataFrame({c: h[c] for c in cols}, index=pd.to_datetime(h["time"]), dtype=float)


def fetch(lat: float, lon: float):
    """All data in UTC, so different sources line up with no daylight-saving gaps."""
    loc = {"latitude": lat, "longitude": lon}
    prev_cols = [f"temperature_2m_previous_day{d}" for d in range(1, MAX_LEAD_DAY + 1)]
    with ThreadPoolExecutor(3) as ex:
        live = ex.submit(_get, FORECAST_URL, {**loc, "hourly": "temperature_2m", "timezone": "GMT",
                                              "past_days": PAST_DAYS, "forecast_days": 9})
        prev = ex.submit(_get, PREVIOUS_RUNS_URL, {**loc, "hourly": ",".join(prev_cols), "timezone": "GMT",
                                                   "past_days": PAST_DAYS, "forecast_days": 1})
        now = ex.submit(_get, FORECAST_URL, {**loc, "current": "temperature_2m", "timezone": "auto",
                                             "forecast_days": 1})
    now = now.result()
    return (_hourly_frame(live.result(), ["temperature_2m"])["temperature_2m"],
            _hourly_frame(prev.result(), prev_cols),
            now.get("timezone", "UTC"),
            (now.get("current") or {}).get("temperature_2m"))


# ---- forecasting ----------------------------------------------------------------------------------
def _pad_left(a: np.ndarray, n: int) -> np.ndarray:
    a = a[-n:]
    return np.concatenate([np.full(n - len(a), np.nan), a]) if len(a) < n else a


def _clean(values) -> list:
    return [None if v is None or not np.isfinite(v) else round(float(v), 1) for v in values]


def _mae(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    ok = np.isfinite(pred) & np.isfinite(truth)
    return round(float(np.abs(pred[ok] - truth[ok]).mean()), 2) if ok.any() else None


_cache: dict = {}
_cache_lock = threading.Lock()


def forecast(lat: float, lon: float) -> dict:
    key = (round(lat, 2), round(lon, 2), pd.Timestamp.now(tz="UTC").floor("h"))
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    result = _compute(lat, lon)
    with _cache_lock:
        if len(_cache) > 256:
            _cache.clear()
        _cache[key] = result
    return result


def _compute(lat: float, lon: float) -> dict:
    t_start = time.perf_counter()
    temps, prev, tz, current = fetch(lat, lon)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    hist = temps[temps.index <= now]
    fut = temps[temps.index > now].iloc[:HORIZON]
    H = len(fut)
    day1 = prev["temperature_2m_previous_day1"].reindex(temps.index)

    # Next 7 days: history + the weather service's current forecast as a known-future input.
    ctx_now = hist.iloc[-CONTEXT:]
    known_now = np.concatenate([day1.reindex(ctx_now.index).to_numpy(), fut.to_numpy()])

    # Last week, rerun: start 7 days ago, and for an hour h ahead use only a forecast issued
    # at least h hours before it (Open-Meteo's archive of previous forecasts).
    t0 = now - pd.Timedelta(hours=H)
    ctx_past = hist[hist.index <= t0].iloc[-CONTEXT:]
    past_hours = pd.date_range(t0 + pd.Timedelta(hours=1), periods=H, freq="h")
    service_past = np.array([
        prev[f"temperature_2m_previous_day{min(math.ceil((i + 1) / 24), MAX_LEAD_DAY)}"].get(t, np.nan)
        for i, t in enumerate(past_hours)
    ], dtype=float)
    known_past = np.concatenate([day1.reindex(ctx_past.index).to_numpy(), service_past])
    observed_past = hist.reindex(past_hours).to_numpy()

    n = max(len(ctx_now), len(ctx_past))
    context = np.stack([_pad_left(ctx_now.to_numpy(), n), _pad_left(ctx_past.to_numpy(), n)])[:, None, :]
    known = np.stack([_pad_left(known_now, n + H), _pad_left(known_past, n + H)])[:, None, :]
    f = get_model().forecast(context, prediction_length=H, known_covariates=known, clip_nonnegative=False)
    med = f.median[:, 0].numpy()
    lo = f.quantile(0.1)[:, 0].numpy()
    hi = f.quantile(0.9)[:, 0].numpy()

    def local(index) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(index).tz_localize("UTC").tz_convert(tz).tz_localize(None)

    def stamps(index) -> list[str]:
        return [t.strftime("%Y-%m-%dT%H:%M") for t in local(index)]

    # Daily highs and lows, in local days.
    fdf = pd.DataFrame({"t": med[0], "s": fut.to_numpy()}, index=local(fut.index))
    daily = []
    for day, g in fdf.groupby(fdf.index.date):
        if len(g) >= 6:
            daily.append({"date": day.isoformat(), "high": round(float(g.t.max()), 1),
                          "low": round(float(g.t.min()), 1), "service_high": _clean([g.s.max()])[0],
                          "service_low": _clean([g.s.min()])[0], "hours": len(g)})
    daily = daily[:7]

    pdf = pd.DataFrame({"obs": observed_past, "t": med[1], "s": service_past}, index=local(past_hours))
    daily_errors = [{"date": day.isoformat(), "timestral": _mae(g.t, g.obs), "service": _mae(g.s, g.obs)}
                    for day, g in pdf.groupby(pdf.index.date) if len(g) >= 6]

    recent = hist.iloc[-RECENT_HOURS:]
    return {
        "location": {"lat": lat, "lon": lon, "timezone": tz},
        "model": model_name(),
        "generated_at": stamps([now])[0],
        "now": {"time": stamps([now])[0], "temperature": current if current is not None else _clean([hist.iloc[-1]])[0]},
        "recent": {"time": stamps(recent.index), "observed": _clean(recent.to_numpy())},
        "forecast": {"time": stamps(fut.index), "median": _clean(med[0]), "low": _clean(lo[0]),
                     "high": _clean(hi[0]), "service": _clean(fut.to_numpy())},
        "daily": daily,
        "track_record": {
            "issued_at": stamps([t0])[0],
            "time": stamps(past_hours), "observed": _clean(observed_past),
            "median": _clean(med[1]), "low": _clean(lo[1]), "high": _clean(hi[1]),
            "service": _clean(service_past),
            "mae": {"timestral": _mae(med[1], observed_past), "service": _mae(service_past, observed_past)},
            "daily": daily_errors,
        },
        "seconds": round(time.perf_counter() - t_start, 2),
    }
