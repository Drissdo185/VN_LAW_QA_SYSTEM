FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY /src .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY .env .

COPY . .

ENV PORT=8501

EXPOSE 8501

WORKDIR /app/src
CMD ["streamlit", "run", "main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]