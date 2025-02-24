#!/bin/bash
# Set a writable cache directory for Hugging Face Hub
export HF_HOME=/app/.cache
export XDG_CONFIG_HOME=/app/.config
mkdir -p /app/.cache

# Optionally set a USER_AGENT to identify your requests
export USER_AGENT="vllm_huggingface_space"

# Launch the vLLM server with the model tag as a positional argument
vllm serve unsloth/llama-3-8b-Instruct-bnb-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  --chat-template examples/tool_chat_template_llama3.1_json.jinja \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype half \
  --enforce-eager \
  --max-model-len 8192 &

# Wait to ensure the vLLM server is fully started (adjust if needed)
sleep 10

# Start the Gradio application using python3
python3 app.py
