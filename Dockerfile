FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./rag/ /app/

# Setup environment variables (these will be overridden by k8s secrets)
ENV OPENAI_API_KEY=""
ENV GOOGLE_API_KEY=""
ENV GOOGLE_CSE_ID=""
ENV WEAVIATE_TRAFFIC_URL=""
ENV WEAVIATE_TRAFFIC_KEY=""

# Create log directory
RUN mkdir -p /app/log

# Expose the port Streamlit runs on
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]