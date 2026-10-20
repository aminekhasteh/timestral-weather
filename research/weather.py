"""Shared helpers: fetch Montreal hourly weather from Open-Meteo (free, no API key) and load Timestral."""

import json
import ssl
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import certifi
import numpy as np
import pandas as pd

LOCATION = {"latitude": 45.54, "longitude": -73.62}  # Montreal (Villeray)
TZ = "America/Toronto"
VAR = "temperature_2m"  # °C at 2 m; try "relative_humidity_2m" or "wind_speed_10m" too
DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


_SSL = ssl.create_default_context(cafile=certifi.where())  # python.org builds on macOS ship no CA certs


def _get(url: str, params: dict) -> dict:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(full, timeout=60, context=_SSL) as r:
        return json.load(r)


def _to_series(payload: dict, var: str) -> pd.Series:
    h = payload["hourly"]
    return pd.Series(h[var], index=pd.to_datetime(h["time"]), name=var, dtype=float)


def fetch_history(days: int = 400, var: str = VAR) -> pd.Series:
    """Hourly observations (ERA5 reanalysis) ending ~6 days ago; the archive lags real time."""
    end = date.today() - timedelta(days=6)
    start = end - timedelta(days=days)
    cache = DATA_DIR / f"history_{var}_{start}_{end}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)[var]
    payload = _get(
        "https://archive-api.open-meteo.com/v1/archive",
        {**LOCATION, "start_date": start.isoformat(), "end_date": end.isoformat(), "hourly": var, "timezone": TZ},
    )
    s = _to_series(payload, var)
    s.to_frame().to_csv(cache)
    return s


def fetch_archive(location: dict, variables: list[str], days: int, tz: str = "GMT") -> pd.DataFrame:
    """Hourly archive data for any location and several variables, cached to data/.

    Defaults to UTC timestamps so series from different cities line up with no daylight-saving gaps.
    """
    end = date.today() - timedelta(days=6)
    start = end - timedelta(days=days)
    tag = f"{location['latitude']}_{location['longitude']}_{'-'.join(variables)}_{start}_{end}_{tz}"
    cache = DATA_DIR / f"archive_{tag}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    payload = _get(
        "https://archive-api.open-meteo.com/v1/archive",
        {**location, "start_date": start.isoformat(), "end_date": end.isoformat(),
         "hourly": ",".join(variables), "timezone": tz},
    )
    h = payload["hourly"]
    df = pd.DataFrame({v: h[v] for v in variables}, index=pd.to_datetime(h["time"]), dtype=float)
    df.to_csv(cache)
    return df


UPSTREAM = {  # weather mostly moves west -> east into Montreal
    "Ottawa": {"latitude": 45.42, "longitude": -75.70},
    "Toronto": {"latitude": 43.65, "longitude": -79.38},
    "Detroit": {"latitude": 42.33, "longitude": -83.05},
    "Chicago": {"latitude": 41.88, "longitude": -87.63},
}


def fetch_previous_runs(location: dict, var: str, max_day: int, days: int | None = None,
                        past_days: int | None = None, tz: str = "GMT") -> pd.DataFrame:
    """Archived weather-service forecasts. Column `<var>_previous_dayN` holds, for each hour, the value
    that was forecast N x 24 h before that hour, i.e. what you could actually have known back then.

    Pass `days` for a backtest range ending where the archive ends, or `past_days` for the live window.
    """
    cols = [f"{var}_previous_day{d}" for d in range(1, max_day + 1)]
    params = {**location, "hourly": ",".join(cols), "timezone": tz}
    cache = None
    if past_days is None:
        end = date.today() - timedelta(days=6)
        start = end - timedelta(days=days)
        params.update(start_date=start.isoformat(), end_date=end.isoformat())
        cache = DATA_DIR / f"prevruns_{location['latitude']}_{location['longitude']}_{var}_d{max_day}_{start}_{end}.csv"
        if cache.exists():
            return pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        params.update(past_days=past_days, forecast_days=1)
    h = _get("https://previous-runs-api.open-meteo.com/v1/forecast", params)["hourly"]
    df = pd.DataFrame({c: h[c] for c in cols}, index=pd.to_datetime(h["time"]), dtype=float)
    if cache is not None:
        df.to_csv(cache)
    return df


def fetch_live(location: dict, variables: list[str], past_days: int = 92, forecast_days: int = 5,
               tz: str = "GMT") -> pd.DataFrame:
    """Recent values plus the current weather-service forecast, in UTC."""
    h = _get("https://api.open-meteo.com/v1/forecast",
             {**location, "hourly": ",".join(variables), "past_days": past_days,
              "forecast_days": forecast_days, "timezone": tz})["hourly"]
    return pd.DataFrame({v: h[v] for v in variables}, index=pd.to_datetime(h["time"]), dtype=float)


def load_model(size: str = "28M"):
    from timestral import TimestralForecaster

    print(f"Loading mistralai/Timestral-{size} (downloads once to the Hugging Face cache)...")
    return TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}")


def forecast(model, contexts: np.ndarray, horizon: int, **covariates) -> dict:
    """contexts: (B, T) or (B, 1, T) array, NaN = missing; optional past_covariates (B, C, T) and
    known_covariates (B, C, T + horizon). Returns numpy arrays of shape (B, horizon)."""
    # clip_nonnegative=False: otherwise an all-positive context (e.g. October) would stop the
    # model from ever forecasting below 0 °C.
    f = model.forecast(contexts, prediction_length=horizon, clip_nonnegative=False, **covariates)
    return {
        "median": f.median[:, 0].numpy(),
        "q10": f.quantile(0.1)[:, 0].numpy(),
        "q90": f.quantile(0.9)[:, 0].numpy(),
    }
