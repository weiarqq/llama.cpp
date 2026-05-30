# Model Inference Experiments

## Purpose

Persistent memory for model inference experiments in this project.

## Quick Links

- Records: `experiments.jsonl`
- Artifacts: `artifacts/`

## Experiment Records

Experiment records are stored as JSON Lines in `experiments.jsonl`. Append one JSON object per experiment. Do not rewrite or reorder existing records.

## Record Schema

Recommended fields:

- `id`: Stable experiment identifier.
- `timestamp`: ISO 8601 timestamp for when the experiment was run or recorded.
- `model`: Model name, path, or identifier.
- `runtime`: Runtime, backend, hardware, and build details relevant to inference.
- `prompt`: Prompt, dataset, or workload description.
- `parameters`: Inference parameters such as context size, temperature, top-p, seed, batch size, and thread count.
- `metrics`: Measured outputs such as latency, throughput, memory, quality, or correctness.
- `artifacts`: Relative paths under `artifacts/` for logs, outputs, traces, profiles, or screenshots.
- `notes`: Observations, anomalies, conclusions, and follow-up questions.

## Recent Experiments

See `experiments.jsonl` for the append-only experiment log.

## Artifacts

Store supporting files in `artifacts/`. Keep artifact paths relative to this directory when referencing them from experiment records.

## Workflow

1. Run the inference experiment.
2. Save logs, outputs, profiles, or other supporting files under `artifacts/`.
3. Append one JSON object to `experiments.jsonl` with links to any artifacts.
4. Preserve existing records exactly; add new records only at the end of the file.
