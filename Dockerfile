# Start with a base Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any, e.g., for bitsandbytes or other libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
# Consider pre-downloading models here if they are large and static,
# or use a volume for Hugging Face cache.
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_lg

# Copy the rest of the application code into the container
COPY *.py ./
COPY frontend ./frontend

# Expose the port the app runs on
EXPOSE 8000

# Set environment variable for Hugging Face cache (optional, good practice)
# This helps persist downloads across container rebuilds if you mount a volume here.
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache/transformers
ENV HF_DATASETS_CACHE=/app/huggingface_cache/datasets
ENV SENTENCE_TRANSFORMERS_HOME=/app/huggingface_cache/sentence_transformers
RUN mkdir -p $HF_HOME/transformers $HF_HOME/datasets $HF_HOME/sentence_transformers

# Command to run the FastAPI application
# Ensure main.py is in the root of the /app directory
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
