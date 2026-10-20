"""Timestral Weather: a 7-day temperature forecast from a pretrained time-series model.

    uvicorn main:app --reload        # local, then open http://127.0.0.1:8000
"""

import threading
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

import forecasting


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=forecasting.get_model, daemon=True).start()  # warm the model in the background
    yield


app = FastAPI(title="Timestral Weather", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"ok": True, "model": forecasting.model_name(), "loaded": forecasting._model is not None}


@app.get("/api/geocode")
def geocode(q: str = Query(min_length=2, max_length=80)):
    try:
        return {"results": forecasting.geocode(q)}
    except httpx.HTTPError:
        raise HTTPException(502, "Place search is unavailable right now. Try again in a minute.")


@app.get("/api/forecast")
def forecast(lat: float = Query(ge=-90, le=90), lon: float = Query(ge=-180, le=180)):
    try:
        return forecasting.forecast(lat, lon)
    except httpx.HTTPError:
        raise HTTPException(502, "Weather data is unavailable right now. Try again in a minute.")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
