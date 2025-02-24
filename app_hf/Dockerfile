# Use the official vLLM Docker image as the base image
FROM vllm/vllm-openai:latest

# Set working directory
WORKDIR /app

# Copy application files into the container
COPY . /app

# Install additional Python packages if needed
RUN pip install --no-cache-dir -r requirements.txt

# Create a writable cache directory and set the HF_HOME environment variable
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache
ENV HF_HOME=/app/.cache

# Expose the port used by Gradio (default: 7860)
EXPOSE 7860

# Clear the base image's entrypoint (if needed)
ENTRYPOINT []

# Make the startup script executable
RUN chmod +x start.sh

# Run the startup script
CMD ["bash", "start.sh"]