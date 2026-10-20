set -e
source ~/venv/bin/activate
pip install pyarrow
cd ~/llm-inference-systems/serving-benchmark

echo "=== BASELINE WILDCHAT ==="
python code/run_sweep.py --system minimal_b1 --wildchat

echo "=== VLLM WILDCHAT ==="
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8000 > ~/vllm_wildchat.log 2>&1 &
VLLM_PID=$!
for i in $(seq 120); do curl -s localhost:8000/health >/dev/null && break; sleep 5; done
curl -sf localhost:8000/health || { echo "vLLM FAILED:"; tail -40 ~/vllm_wildchat.log; exit 1; }
python code/run_sweep.py --system vllm_paged --url http://localhost:8000 --wildchat
kill $VLLM_PID
sleep 10

echo "=== MOONCAKE WILDCHAT ==="
export LD_LIBRARY_PATH=$(find ~/venv -name "libcudart.so.12*" -printf "%h\n" | head -1):$LD_LIBRARY_PATH
pkill mooncake_master || true
mooncake_master --port 50051 > ~/mooncake_master.log 2>&1 &
MASTER_PID=$!
sleep 5
MOONCAKE_CONFIG_PATH=$HOME/mooncake_config.json vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --port 8000 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}' \
  > ~/vllm_mooncake_wildchat.log 2>&1 &
VLLM_PID=$!
for i in $(seq 120); do curl -s localhost:8000/health >/dev/null && break; sleep 5; done
curl -sf localhost:8000/health || { echo "vLLM FAILED:"; tail -40 ~/vllm_mooncake_wildchat.log; exit 1; }
python code/run_sweep.py --system mooncake --url http://localhost:8000 --wildchat
python code/run_sweep.py --system mooncake_warm --url http://localhost:8000 --wildchat --warm
kill $VLLM_PID $MASTER_PID
echo "=== WILDCHAT DONE ==="
