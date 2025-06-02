# Use a lightweight Python 3.9 slim base image
FROM python:3.9-slim

# Set environment variables
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install system dependencies needed for RDKit drawing (and general)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code, models and dataset
COPY . .

# Expose port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "fluroml-app_2025.py", "--server.port=8501", "--server.address=0.0.0.0"]
