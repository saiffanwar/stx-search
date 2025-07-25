# Use official Python 3.10.10 image
FROM python:3.10.10-slim

# Set the working directory in the container
WORKDIR /app/explainer

# Install build tools (optional, but helps with some pip installs)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your application code
COPY explainer/ /app/explainer/

# Set the default command
CMD ["python", "explainer_experiments_runner.py"]

