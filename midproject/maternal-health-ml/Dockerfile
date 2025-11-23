FROM python:3.11-slim

WORKDIR /app

# Copy files first
COPY requirements.txt .
COPY app.py .
COPY model.pkl .
COPY label_encoder.pkl .

# Install Python packages only
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]