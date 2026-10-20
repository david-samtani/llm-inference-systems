set -e
source ~/venv/bin/activate
pip install mooncake-transfer-engine nvidia-cuda-runtime-cu12
export LD_LIBRARY_PATH=$(find ~/venv -name "libcudart.so.12*" -printf "%h\n" | head -1):$LD_LIBRARY_PATH
cd ~/llm-inference-systems/serving-benchmark
mooncake_master --port 50051 > ~/mooncake_master.log 2>&1 &
MASTER_PID=$!
sleep 5
MOONCAKE_CONFIG_PATH=$HOME/mooncake_config.json vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --port 8000 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}' \
  > ~/vllm_mooncake.log 2>&1 &
VLLM_PID=$!
for i in $(seq 120); do curl -s localhost:8000/health >/dev/null && break; sleep 5; done
curl -sf localhost:8000/health || { echo "vLLM FAILED:"; tail -40 ~/vllm_mooncake.log; exit 1; }
echo "=== MOONCAKE COLD ==="
python code/run_sweep.py --system mooncake --url http://localhost:8000
echo "=== MOONCAKE WARM ==="
python code/run_sweep.py --system mooncake_warm --url http://localhost:8000 --warm
kill $VLLM_PID $MASTER_PID
echo "=== MOONCAKE DONE ==="
