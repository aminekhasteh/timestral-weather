---
title: Timestral Weather
emoji: 🌡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: 7-day temperature forecast from a pretrained time-series model
---

# Timestral Weather

A 7-day hourly temperature forecast for any city, made by
[Timestral-28M](https://huggingface.co/mistralai/Timestral-28M), Mistral AI's pretrained
time-series model, and checked against what actually happened.

- **Forecast:** Timestral reads the last 60 days of hourly temperature plus the weather service's
  forecast as a known-future input, and predicts the next 168 hours with an 80% likely range. No
  training on the location: it's zero-shot.
- **Track record:** the same forecast is rerun as of exactly 7 days ago, using only weather-service
  forecasts that had already been issued by then, and scored against observed temperatures,
  side by side with the weather service.
- **Why it works:** in a year-long backtest in Montreal, this setup was about 15% more accurate than
  the weather service alone (1.46 °C vs 1.71 °C mean absolute error), mostly by correcting local
  bias in spring and summer and in the first 24 hours.

Data: [Open-Meteo](https://open-meteo.com/) forecast, previous-runs and geocoding APIs (free, no key).

## Run locally

```bash
cd webapp
uv run uvicorn main:app --reload     # first run installs dependencies and downloads the model (111 MB)
```

Open http://127.0.0.1:8000. Try `?lat=48.8566&lon=2.3522&name=Paris` to link straight to a place.

## Deploy to Hugging Face Spaces (free)

1. Create a Space at https://huggingface.co/new-space: pick **Docker**, then **Blank**, on the free CPU hardware.
2. Push this folder's contents to the Space's git repo:

   From the root of this repository:

   ```bash
   git clone https://huggingface.co/spaces/<your-username>/timestral-weather ../timestral-weather-space
   cp -r webapp/. ../timestral-weather-space/
   cd ../timestral-weather-space && git add . && git commit -m "Timestral Weather" && git push
   ```

   (When git asks for a password, use a Hugging Face access token with write permission.)
3. The first build takes about 5–10 minutes. The app is then live at
   `https://<your-username>-timestral-weather.hf.space`.

Free Spaces go to sleep after a period without visitors and wake on the next visit, which can
take a minute or so.
Any host that runs Docker with at least 2 GB of RAM also works (Google Cloud Run, Fly.io, Railway).

## Files

| File | What it does |
|---|---|
| `forecasting.py` | Fetches data, runs Timestral for the next 7 days and for last week's rerun, scores it |
| `main.py` | FastAPI app: `/api/forecast`, `/api/geocode`, `/api/health`, and serves the page |
| `static/` | The page: `index.html`, `style.css`, `app.js` (no build step, no frameworks) |
| `Dockerfile` | CPU-only image with the model weights baked in |

## Notes

- Observed temperatures are Open-Meteo's hourly analysis for the coordinates, not a single
  weather station's readings.
- Open-Meteo's free API is for non-commercial use (up to 10,000 calls a day). Each new
  forecast uses 3 calls, and results are cached per location for the hour.
- Set `TIMESTRAL_MODEL=mistralai/Timestral-209M` to use the largest model (837 MB; in the
  Montreal backtest it was no more accurate than 28M, and about 6× slower).
- Timestral code is Apache 2.0; the model weights are under Mistral's Modified MIT license.
