# Use the official vLLM Docker image as the base image
FROM vllm/vllm-openai:latest

# Use the root user to ensure write permissions (if needed)
USER root

# Set the working directory
WORKDIR /app

# Install system dependencies if needed (e.g., wget)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Create and set permissions for cache directories (for Hugging Face, matplotlib, etc.)
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache \
    && mkdir -p /app/.config/matplotlib && chmod -R 777 /app/.config/matplotlib

# Set environment variables for cache directories and user agent
ENV HF_HOME=/app/.cache
ENV XDG_CACHE_HOME=/app/.cache
ENV MPLCONFIGDIR=/app/.config/matplotlib
ENV USER_AGENT="my-gradio-app"

# Copy the requirements file and install additional Python dependencies (e.g., gradio)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code (including app.py and any other needed files)
COPY . .

# Expose the port for Gradio (Spaces expects the app on port 7860)
EXPOSE 7860

# Override the base image's entrypoint so our CMD is executed directly.
ENTRYPOINT []

# Create a writable log file (alternatively, you could redirect logs to /tmp)
RUN touch /app/vllm.log && chmod 666 /app/vllm.log

# Set the CMD to launch the vLLM server (for your new model) in the background and then start the Gradio app.
CMD ["bash", "-c", "vllm.entrypoints.openai.api_server --model unsloth/llama-3-8b-Instruct-bnb-4bit --enable-auto-tool-choice --tool-call-parser llama3_json --chat-template examples/tool_chat_template_llama3.1_json.jinja --quantization bitsandbytes --load-format bitsandbytes --dtype half --max-model-len 8192 > /app/vllm.log 2>&1 & python3 app.py"]
