FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY . .

ENV PORT=8080

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} app:app"]
