# News Classifier - Docker Image
# Python 3.10 with ML/NLP dependencies

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data (needed for some operations)
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/chromadb models

# Expose ports
# Streamlit default port
EXPOSE 8501
# Jupyter notebook port
EXPOSE 8888

# Default command - run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
