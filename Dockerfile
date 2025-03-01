FROM python:3.10

WORKDIR /app


# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy application code
COPY .env .
COPY ./rag/ .


# Expose the port Streamlit runs on
EXPOSE 8501


# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]