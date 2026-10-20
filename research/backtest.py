"""Backtest: how good is zero-shot Timestral at forecasting Montreal temperature?

Picks many past "forecast origins", hides the next N hours, asks Timestral to forecast them,
and compares against two classic baselines:
  - persistence:     the temperature stays at its last observed value
  - seasonal naive:  tomorrow looks exactly like the last 24 hours

    uv run python backtest.py                     # 28M model, 72 h horizon
    uv run python backtest.py --model 209M --horizon 168
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from weather import OUT_DIR, VAR, fetch_history, forecast, load_model

p = argparse.ArgumentParser()
p.add_argument("--model", default="28M", choices=["5M", "28M", "123M", "209M"])
p.add_argument("--horizon", type=int, default=72, help="hours ahead")
p.add_argument("--context-days", type=int, default=60, help="history the model sees")
p.add_argument("--windows", type=int, default=30, help="number of forecast origins")
args = p.parse_args()

H, C = args.horizon, args.context_days * 24
s = fetch_history(days=args.context_days + 200)
y = s.to_numpy()
print(f"{len(y)} hourly points: {s.index[0]} -> {s.index[-1]}")

# Forecast origins spread over the last ~150 days, every 5 days at midnight-ish.
last = len(y) - H
origins = np.array([last - 24 * 5 * i for i in range(args.windows)])[::-1]
origins = origins[origins >= C]
ctx = np.stack([y[o - C : o] for o in origins])
truth = np.stack([y[o : o + H] for o in origins])

model = load_model(args.model)
t0 = time.perf_counter()
fc = forecast(model, ctx, H)
print(f"Forecast {len(origins)} windows x {H} h in {time.perf_counter() - t0:.1f} s")

persistence = np.repeat(ctx[:, -1:], H, axis=1)
seasonal = np.tile(ctx[:, -24:], (1, H // 24 + 1))[:, :H]
methods = {f"Timestral-{args.model}": fc["median"], "Seasonal naive (repeat last 24h)": seasonal,
           "Persistence (last value)": persistence}


def mae(pred, axis=None, hours=None):
    h = hours or pred.shape[1]
    return np.nanmean(np.abs(pred[:, :h] - truth[:, :h]), axis=axis)


print(f"\nMean absolute error over {H} h ({VAR}, °C):")
for name, pred in methods.items():
    print(f"  {name:36s} {mae(pred):5.2f}   (first 24 h: {mae(pred, hours=24):.2f})")
inside = (truth >= fc["q10"]) & (truth <= fc["q90"])
print(f"\n80% interval coverage: {np.nanmean(inside):.0%}  (well-calibrated ≈ 80%)")

# Plot 1: error by lead time. Plot 2: one example window with the uncertainty band.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.5))
for name, pred in methods.items():
    a1.plot(np.arange(1, H + 1), mae(pred, axis=0), label=name)
a1.set(xlabel="hours ahead", ylabel="MAE (°C)", title="Error by forecast lead time")
a1.legend()
a1.grid(alpha=0.3)

i = -1
o = origins[i]
past_t = s.index[o - 7 * 24 : o]
fut_t = s.index[o : o + H]
a2.plot(past_t, y[o - 7 * 24 : o], color="0.4", label="history")
a2.plot(fut_t, truth[i], color="black", label="actual")
a2.plot(fut_t, fc["median"][i], color="C0", label="Timestral median")
a2.fill_between(fut_t, fc["q10"][i], fc["q90"][i], color="C0", alpha=0.25, label="10-90% band")
a2.set(title=f"Latest backtest window (from {s.index[o]:%b %d %H:%M})", ylabel="°C")
a2.legend()
a2.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
path = OUT_DIR / f"backtest_{args.model}_{H}h.png"
fig.savefig(path, dpi=130)
print(f"\nSaved {path}")
