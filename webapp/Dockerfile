FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

# Bake the model weights into the image so the first visitor doesn't wait for a download
ARG TIMESTRAL_MODEL=mistralai/Timestral-28M
ENV TIMESTRAL_MODEL=${TIMESTRAL_MODEL}
RUN python -c "import os; from timestral import TimestralForecaster; TimestralForecaster.from_pretrained(os.environ['TIMESTRAL_MODEL'])"

COPY --chown=user . .
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
