"""Forecast Montreal's temperature for the next few days.

Uses the setup that won in backtest_covariates.py: Montreal's temperature history plus the weather
service's own forecast as a known-future input. If backtest_covariates.py has saved a calibration
for this model, the uncertainty band is widened so that "80%" really holds about 80% of outcomes.

    uv run python forecast_next.py
    uv run python forecast_next.py --model 209M --horizon 96
    uv run python forecast_next.py --plain        # temperature history only (the original version)
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weather import LOCATION, OUT_DIR, TZ, fetch_live, fetch_previous_runs, forecast, load_model

p = argparse.ArgumentParser()
p.add_argument("--model", default="28M", choices=["5M", "28M", "123M", "209M"])
p.add_argument("--horizon", type=int, default=72, help="hours ahead (max 96)")
p.add_argument("--plain", action="store_true", help="temperature history only")
args = p.parse_args()
H = args.horizon
assert H <= 96, "the live weather-service forecast is fetched 5 days ahead"

live = fetch_live(LOCATION, ["temperature_2m"], past_days=92, forecast_days=5)
now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
hist, fut = live[live.index <= now], live[live.index > now].iloc[:H]
y = hist["temperature_2m"].to_numpy()
nwp = fut["temperature_2m"]
model = load_model(args.model)

plain = forecast(model, y[None, :], H)
if args.plain:
    best = plain
else:
    # History part: forecasts issued 1 day before each past hour; future part: the current forecast.
    prev = fetch_previous_runs(LOCATION, "temperature_2m", 1, past_days=92)["temperature_2m_previous_day1"]
    known = np.concatenate([prev.reindex(hist.index).to_numpy(), nwp.to_numpy()])[None, None]  # (1, 1, T+H)
    best = forecast(model, y[None, None, :], H, known_covariates=known)

# Calibrated 80% band (falls back to the raw 10-90% band when no calibration exists).
med, lo, hi = best["median"][0], best["q10"][0], best["q90"][0]
cal_path = OUT_DIR / f"calibration_{args.model}.json"
band_label = "10-90% band (uncalibrated)"
if not args.plain and cal_path.exists():
    ks = json.loads(cal_path.read_text())["widen_factor_by_day"]
    k = np.array([ks[min(h // 24, len(ks) - 1)] for h in range(H)])
    lo, hi = med - k * (med - lo), med + k * (hi - med)
    band_label = "80% range (calibrated)"
elif not args.plain:
    print("Tip: run backtest_covariates.py once to calibrate the uncertainty band.")


def local(index):
    return index.tz_localize("UTC").tz_convert(TZ).tz_localize(None)


future = local(nwp.index)
table = pd.DataFrame({"timestral": med, "low": lo, "high": hi, "weather_service": nwp.to_numpy()}, index=future)
if not args.plain:
    table.insert(3, "temp_only", plain["median"][0])
print(f"\nNext 24 h, every 3 h, Montreal local time (°C; low/high = {band_label}):")
print(table.iloc[:24:3].round(1).to_string())

fig, ax = plt.subplots(figsize=(12, 4.8))
tail = hist["temperature_2m"].iloc[-5 * 24 :]
ax.plot(local(tail.index), tail.values, color="0.3", label="last 5 days")
ax.plot(future, med, color="C0", lw=2,
        label=f"Timestral-{args.model} ({'temperature only' if args.plain else '+ weather-service forecast'})")
ax.fill_between(future, lo, hi, color="C0", alpha=0.22, label=band_label)
if not args.plain:
    ax.plot(future, plain["median"][0], color="C0", ls=":", label="Timestral, temperature only")
ax.plot(future, nwp.values, color="C3", ls="--", label="weather-service forecast (Open-Meteo)")
ax.axvline(local(pd.DatetimeIndex([hist.index[-1]]))[0], color="0.6", ls=":")
ax.set(title="Montreal hourly temperature forecast", ylabel="°C")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
path = OUT_DIR / f"forecast_next_{args.model}{'_plain' if args.plain else ''}.png"
fig.savefig(path, dpi=130)
print(f"\nSaved {path}")
