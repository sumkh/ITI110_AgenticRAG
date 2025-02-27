# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose the port on which Gradio will run (default is 7860)
EXPOSE 7860

# Define environment variable for the vLLM endpoint (adjust as needed)
# This could also be passed in at runtime via docker run -e VLLM_ENDPOINT=...
ENV VLLM_ENDPOINT=http://vllm-instance:8000

# Run the application
CMD ["python", "app.py"]
