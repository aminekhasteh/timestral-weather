"""Which extra inputs make Timestral more accurate for Montreal temperature?

Compares, on identical forecast windows:
  - temperature only (the baseline from backtest.py)
  - + air pressure at Montreal         (falling pressure = a weather system is coming)
  - + temperatures in upstream cities  (Ottawa/Toronto/Detroit/Chicago see incoming weather first)
  - + the weather service's own forecast, as a known-future input
  - + everything
plus references: "repeat yesterday", the weather service's forecast on its own, and classical
ARIMA(2,1,1) models (with and without the weather-service forecast) fitted per window.

Fairness: pressure and city temperatures are past-only inputs. For the weather-service forecast,
an hour N hours after the start point only uses a forecast issued at least N hours before that
hour (Open-Meteo's archive of previous runs), so nothing is used before it would have existed.

    uv run python backtest_covariates.py
    uv run python backtest_covariates.py --model 209M --horizon 96
"""

import argparse
import json
import math
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weather import LOCATION, OUT_DIR, UPSTREAM, fetch_archive, fetch_previous_runs, load_model

p = argparse.ArgumentParser()
p.add_argument("--model", default="28M", choices=["5M", "28M", "123M", "209M"])
p.add_argument("--horizon", type=int, default=72, help="hours ahead (max 168)")
p.add_argument("--context-days", type=int, default=60, help="history the model sees")
p.add_argument("--windows", type=int, default=73, help="number of forecast origins (73 x 5 days = 1 year)")
p.add_argument("--step-days", type=int, default=5, help="days between forecast origins")
p.add_argument("--batch", type=int, default=16, help="windows per model call (lower = less RAM)")
p.add_argument("--no-arima", action="store_true", help="skip the classical ARIMA references (they take ~1-2 min)")
args = p.parse_args()

H, C = args.horizon, args.context_days * 24
assert H <= 168, "the forecast archive goes back at most 7 days per forecast"
days = args.context_days + args.windows * args.step_days + H // 24 + 2
max_day = math.ceil(H / 24)

# ---- data ---------------------------------------------------------------------------------------
print("Fetching data (cached after the first run)...")
mtl = fetch_archive(LOCATION, ["temperature_2m", "pressure_msl"], days)
idx = mtl.index
upstream = np.stack([fetch_archive(loc, ["temperature_2m"], days)["temperature_2m"].reindex(idx).to_numpy()
                     for loc in UPSTREAM.values()])
prev = fetch_previous_runs(LOCATION, "temperature_2m", max_day, days=days).reindex(idx)
nwp_by_day = {d: prev[f"temperature_2m_previous_day{d}"].to_numpy() for d in range(1, max_day + 1)}
y = mtl["temperature_2m"].to_numpy()
print(f"{len(y)} hourly points (UTC): {idx[0]} -> {idx[-1]}")

# ---- forecast windows (same for every setup) ----------------------------------------------------
last = len(y) - H
origins = np.array([last - 24 * args.step_days * i for i in range(args.windows)])[::-1]
origins = origins[origins >= C]
ctx = np.stack([y[o - C : o] for o in origins])[:, None, :]  # (B, 1, T)
truth = np.stack([y[o : o + H] for o in origins])  # (B, H)
lead_day = np.ceil(np.arange(1, H + 1) / 24).astype(int)  # hour h ahead -> forecast issued >= h hours earlier


def nwp_window(o):
    """Weather-service forecast for one window: (T + H,). History uses 1-day-old forecasts;
    each future hour uses the freshest forecast that already existed at the start point."""
    past = nwp_by_day[1][o - C : o]
    future = np.array([nwp_by_day[d][o + h] for h, d in enumerate(lead_day)])
    return np.concatenate([past, future])


nwp_known = np.stack([nwp_window(o) for o in origins])[:, None, :]  # (B, 1, T + H)
nwp_only = nwp_known[:, 0, C:]  # the weather service's forecast on its own, for reference
missing = np.isnan(nwp_only).mean()
if missing > 0.05:
    print(f"Warning: {missing:.0%} of the archived weather-service forecasts are missing for these dates.")
print(f"{len(origins)} forecast windows from {idx[origins[0]]:%Y-%m-%d} to {idx[origins[-1]]:%Y-%m-%d}\n")

past_inputs = {"pressure": mtl["pressure_msl"].to_numpy()[None, :], "upstream": upstream}
setups = {  # name -> (past-only inputs, use weather-service forecast)
    "temperature only": ([], False),
    "+ pressure": (["pressure"], False),
    "+ upstream cities": (["upstream"], False),
    "+ pressure + upstream": (["pressure", "upstream"], False),
    "+ weather-service forecast": ([], True),
    "+ everything": (["pressure", "upstream"], True),
}

model = load_model(args.model)


def run(past_keys, use_nwp):
    past = None
    if past_keys:
        series = np.concatenate([past_inputs[k] for k in past_keys])  # (n_cov, N)
        past = np.stack([series[:, o - C : o] for o in origins])  # (B, n_cov, T)
    med, q10, q90 = [], [], []
    for i in range(0, len(origins), args.batch):
        sl = slice(i, i + args.batch)
        f = model.forecast(ctx[sl], prediction_length=H, clip_nonnegative=False,
                           past_covariates=None if past is None else past[sl],
                           known_covariates=nwp_known[sl] if use_nwp else None)
        med.append(f.median[:, 0].numpy())
        q10.append(f.quantile(0.1)[:, 0].numpy())
        q90.append(f.quantile(0.9)[:, 0].numpy())
    return np.concatenate(med), np.concatenate(q10), np.concatenate(q90)


ARIMA_ORDER = (2, 1, 1)  # a standard, deliberately simple choice
FOURIER_K = 3  # sine/cosine pairs for the 24 h cycle


def fourier(start, n):
    t = np.arange(start, start + n)
    return np.column_stack([f(2 * np.pi * k * t / 24) for k in range(1, FOURIER_K + 1) for f in (np.sin, np.cos)])


def run_arima(use_nwp):
    """Classical baseline: fit ARIMA(2,1,1) + daily-cycle terms from scratch on each window's history.
    With use_nwp, the weather-service forecast is an extra regressor (ARIMAX), using the same
    no-peeking forecast values as the Timestral setup."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    B = len(origins)
    med, lo, hi = (np.full((B, H), np.nan) for _ in range(3))
    for i, o in enumerate(origins):
        X = fourier(o - C, C + H)
        if use_nwp:
            X = np.column_stack([X, pd.Series(nwp_known[i, 0]).ffill().bfill().to_numpy()])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(ctx[i, 0], exog=X[:C], order=ARIMA_ORDER).fit(disp=False, maxiter=50)
                fc = fit.get_forecast(H, exog=X[C:])
            med[i] = fc.predicted_mean
            ci = fc.conf_int(alpha=0.2)  # 80% interval
            lo[i], hi[i] = ci[:, 0], ci[:, 1]
        except Exception as e:  # a failed fit leaves that window as NaN
            print(f"    ARIMA failed on window {i}: {e}")
        if (i + 1) % 10 == 0:
            print(f"    ARIMA {i + 1}/{B} windows")
    return med, lo, hi


def day_maes(pred):
    err = np.abs(pred - truth)
    return [np.nanmean(err[:, d * 24 : (d + 1) * 24]) for d in range((H + 23) // 24)]


rows, lead_mae, preds = {}, {}, {}
seasonal = np.tile(ctx[:, 0, -24:], (1, H // 24 + 1))[:, :H]
references = {"ref: repeat yesterday": seasonal, "ref: weather-service forecast alone": nwp_only}
arima_setups = {} if args.no_arima else {"ARIMA, temperature only": False,
                                          "ARIMAX + weather-service forecast": True}
for name in [*setups, *arima_setups, *references]:
    t0 = time.perf_counter()
    if name in references:
        med, cover = references[name], np.nan
    else:
        med, q10, q90 = run_arima(arima_setups[name]) if name in arima_setups else run(*setups[name])
        preds[name] = (med, q10, q90)
        cover = np.nanmean((truth >= q10) & (truth <= q90))
    preds.setdefault(name, (med, None, None))
    secs = time.perf_counter() - t0
    rows[name] = {"MAE all": np.nanmean(np.abs(med - truth)),
                  **{f"MAE day {d + 1}": m for d, m in enumerate(day_maes(med))},
                  "80% band coverage": cover, "seconds": secs}
    lead_mae[name] = np.nanmean(np.abs(med - truth), axis=0)
    print(f"  done: {name} ({secs:.1f} s)")

res = pd.DataFrame(rows).T
base = res.loc["temperature only", "MAE all"]
res["vs baseline"] = (res["MAE all"] / base - 1).map(lambda x: f"{x:+.0%}")
pd.set_option("display.width", 160)
print(f"\nTimestral-{args.model}, {H} h horizon, {args.context_days} days of context (MAE in °C):\n")
print(res.round(2).to_string())
csv_path = OUT_DIR / f"covariates_{args.model}_{H}h.csv"
res.to_csv(csv_path)

# ---- error by season ------------------------------------------------------------------------------
SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
          6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
season = np.array([SEASON[idx[o].month] for o in origins])
by_season = pd.DataFrame({
    name: {s: np.nanmean(np.abs(preds[name][0][season == s] - truth[season == s]))
           for s in ["winter", "spring", "summer", "autumn"] if (season == s).any()}
    for name in ["temperature only", *(["ARIMA, temperature only"] if arima_setups else []),
                 "+ weather-service forecast", *(["ARIMAX + weather-service forecast"] if arima_setups else []),
                 "ref: weather-service forecast alone"]
}).T
by_season.columns = [f"{c} ({(season == c).sum()} win)" for c in by_season.columns]
print("\nMAE by season of the forecast start (°C):\n")
print(by_season.round(2).to_string())

# ---- calibrate the uncertainty range (for the setup forecast_next.py uses) -------------------------
# For each forecast day, find the factor k that widens the raw 10-90% band around the median so that
# 80% of actual temperatures land inside. Fit on even windows, check on odd ones, then refit on all.
CAL = "+ weather-service forecast"
med, q10, q90 = preds[CAL]
lo = (med - truth) / np.maximum(med - q10, 1e-6)
hi = (truth - med) / np.maximum(q90 - med, 1e-6)
k_needed = np.maximum(lo, hi)  # each point is inside the widened band iff k >= k_needed
n_days = math.ceil(H / 24)
day_cols = [slice(d * 24, min((d + 1) * 24, H)) for d in range(n_days)]


def fit(rows):
    return [float(np.nanquantile(k_needed[rows, c], 0.8)) for c in day_cols]


def coverage(rows, ks):
    out = []
    for c, k in zip(day_cols, ks):
        kn = k_needed[rows, c]
        out.append(np.mean(kn[np.isfinite(kn)] <= k))
    return out


all_rows = np.arange(len(origins))
train, test = all_rows[::2], all_rows[1::2]
k_train = fit(train)
cal = pd.DataFrame({"raw 10-90% coverage": coverage(test, [1.0] * n_days),
                    "widen factor": k_train,
                    "calibrated coverage": coverage(test, k_train)},
                   index=[f"day {d + 1}" for d in range(n_days)])
print(f"\nUncertainty calibration for '{CAL}' (target 80%, checked on {len(test)} held-out windows):\n")
print(cal.round(2).to_string())
k_all = fit(all_rows)
cal_path = OUT_DIR / f"calibration_{args.model}.json"
cal_path.write_text(json.dumps({"model": args.model, "setup": CAL, "target": 0.8,
                                "widen_factor_by_day": k_all, "windows": len(origins)}, indent=2))
print(f"Saved {cal_path} (forecast_next.py uses it automatically)")

# ---- plot -----------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
for name, m in lead_mae.items():
    style = {"ls": "--", "color": "0.5"} if "yesterday" in name else \
            {"ls": "--", "color": "C3"} if name.startswith("ref") else \
            {"ls": ":"} if name.startswith("ARIMA") else {}
    ax.plot(np.arange(1, H + 1), m, label=name, lw=2.2 if name == "+ everything" else 1.4, **style)
ax.set(xlabel="hours ahead", ylabel="MAE (°C)",
       title=f"Which inputs help? Timestral-{args.model}, {len(origins)} windows")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
png_path = OUT_DIR / f"covariates_{args.model}_{H}h.png"
fig.savefig(png_path, dpi=130)
print(f"\nSaved {png_path}\nSaved {csv_path}")
