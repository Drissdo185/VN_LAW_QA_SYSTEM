./bin/ollama serve

wait 5

echo "Pull model qwen"
ollama pull qwen2.5:0.5b-instruct

