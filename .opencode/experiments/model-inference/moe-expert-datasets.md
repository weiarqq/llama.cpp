# MoE Expert Dataset Collection

Use `llama-server`, not `llama-bench`, because `llama-bench` can generate random token prompts that do not represent real routing behavior.

## Dataset Presets

The collection script has these Hugging Face dataset presets:

- `wikitext`: English prose / wiki text, completion-style workload.
- `mmlu`: English knowledge questions, chat-style workload.
- `gsm8k`: math word problems, chat-style workload.
- `humaneval`: Python code completion prompts, completion-style workload.
- `alpaca`: instruction-following prompts, chat-style workload.

You can also pass local JSONL files from OpenCLAW production traces or curated prompts with `--jsonl`.

## Recommended OpenCLAW Run

Start from an existing `llama-server` deployment or let the script launch one. The server must run with `LLAMA_MOE_EXPERT_STATS=1`, and stderr/stdout must be saved so the `LLAMA_MOE_EXPERT_STATS ` lines are preserved.

Example with script-managed server:

```sh
python3 scripts/moe-expert-collect.py \
  --server-cmd './build-nollama/bin/llama-server -m qwen-model/Qwen3.5-35B-A3B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -c 8192 -np 1' \
  --base-url http://127.0.0.1:8080 \
  --dataset wikitext \
  --dataset mmlu \
  --dataset gsm8k \
  --dataset humaneval \
  --dataset alpaca \
  --limit 128 \
  --max-tokens 128 \
  --expert-log .opencode/experiments/model-inference/artifacts/moe-server.log \
  --expert-jsonl .opencode/experiments/model-inference/artifacts/moe-experts.jsonl \
  --responses-jsonl .opencode/experiments/model-inference/artifacts/moe-responses.jsonl
```

Example with an already running OpenCLAW service:

```sh
LLAMA_MOE_EXPERT_STATS=1 ./build-nollama/bin/llama-server -m qwen-model/Qwen3.5-35B-A3B-Q4_K_M.gguf --host 0.0.0.0 --port 8080 2> .opencode/experiments/model-inference/artifacts/moe-server.log

python3 scripts/moe-expert-collect.py \
  --base-url http://127.0.0.1:8080 \
  --dataset wikitext \
  --dataset mmlu \
  --dataset gsm8k \
  --limit 128 \
  --responses-jsonl .opencode/experiments/model-inference/artifacts/moe-responses.jsonl
```

Then extract and analyze:

```sh
python3 scripts/moe-expert-collect.py \
  --responses-jsonl /tmp/unused.jsonl \
  --expert-log .opencode/experiments/model-inference/artifacts/moe-server.log \
  --expert-jsonl .opencode/experiments/model-inference/artifacts/moe-experts.jsonl \
  --limit 0

python3 scripts/moe-expert-analyze.py \
  --input .opencode/experiments/model-inference/artifacts/moe-experts.jsonl \
  --summary .opencode/experiments/model-inference/artifacts/moe-expert-summary.json \
  --reorder-map .opencode/experiments/model-inference/artifacts/moe-frequency-reorder-map.json
```

## Outputs

- `moe-experts.jsonl`: one JSON object per layer/token expert selection.
- `moe-expert-summary.json`: per-layer expert frequency, entropy, top expert coverage, co-occurrence, and consecutive-token Jaccard overlap.
- `moe-frequency-reorder-map.json`: per-layer `new_to_old` and `old_to_new` expert id maps sorted by observed frequency.

## Notes

- Use `--temperature 0` for repeatable generated continuations; routing for prompt tokens still comes from real input text.
- Use `--jsonl` for OpenCLAW production prompts when available. This is usually better than public datasets for final reorder decisions.
- Keep `-np 1` for clean single-sequence continuity analysis. Increase parallelism only after validating how `seq_ids` map to OpenCLAW request slots.
- For SYCL runs, source oneAPI before starting the server: `source /opt/intel/oneapi/setvars.sh`.
