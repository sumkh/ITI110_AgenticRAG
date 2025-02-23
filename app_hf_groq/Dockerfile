# Use Python 3.11 slim image
FROM python:3.11-slim

# Ensure we run as root (the default) so we can set permissions
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create and set permissions for cache directories
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache \
    && mkdir -p /app/.config/matplotlib && chmod -R 777 /app/.config/matplotlib

# Set environment variables for cache directories
ENV HF_HOME=/app/.cache
ENV XDG_CACHE_HOME=/app/.cache
ENV MPLCONFIGDIR=/app/.config/matplotlib
ENV USER_AGENT="my-gradio-app"

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port for Gradio (Spaces expects the app on port 7860)
EXPOSE 7860

CMD ["python", "app.py"]
