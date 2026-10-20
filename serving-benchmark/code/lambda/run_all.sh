set -e
sudo apt-get update && sudo apt-get install -y python3-venv
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
pip install vllm requests
cd ~/llm-inference-systems/serving-benchmark
python -c "import torch, vllm; print(torch.__version__, torch.cuda.is_available(), vllm.__version__)"

echo "=== BASELINE ==="
python code/run_sweep.py --system minimal_b1

echo "=== VLLM ==="
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8000 > ~/vllm_server.log 2>&1 &
VLLM_PID=$!
for i in $(seq 120); do curl -s localhost:8000/health >/dev/null && break; sleep 5; done
python code/run_sweep.py --system vllm_paged --url http://localhost:8000
kill $VLLM_PID
echo "=== ALL DONE ==="
