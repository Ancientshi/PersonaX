# Experiments

This directory contains the released evaluation client, EasyRec adapter, preprocessing notebook, metrics, and plotting helpers. The PersonaX method is in [`personax/`](../personax/).

## Setup

Use Python 3.10 or 3.11. From the repository root, activate your environment and install the experiment dependencies:

```bash
python -m pip install -r experiments/requirements.txt
```

The adapter needs `model.py` from [HKUDS/EasyRec](https://github.com/HKUDS/EasyRec), which is not bundled here. Keep the upstream source in a separate checkout and add its root to `PYTHONPATH` when starting the adapter:

```bash
PYTHONPATH=/absolute/path/to/EasyRec python experiments/easyrec/app.py
```

It serves `hkuds/easyrec-roberta-large` on port 8500 and loads the checkpoint and tokenizer when it starts. The first start may download these files. Set `EASYREC_CACHE_DIR` to use a different cache directory; the default is `model/` in this repository. Follow the EasyRec repository for its source and model setup.

In a second terminal, start PersonaX:

```bash
python -m personax.server
```

The server listens on `http://127.0.0.1:8001`. Set `PERSONAX_STORAGE_DIR` before starting it to choose a cache directory; the default is `storage/` in this repository. Existing personas are reused by `user_id`, without checking dataset, model, or sampling settings. Use a fresh storage directory and restart the server for each independent comparison.

## Evaluation client

The sampled CSV files remain in `Amazon/`. The client reads `Amazon/<dataset>/sampled_<subset>.csv`; the defaults are `CDs_and_Vinyl` and `200`. Required columns are `user_id`, `item_id`, `rating`, and `timestamp`, together with the available item attributes.

Set `OPENAI_API_KEY` in the client terminal, then run:

```bash
python -m experiments.client_agent \
  --dataset CDs_and_Vinyl --subset 200 \
  --method personax --persona_learning_type distill \
  --distance_threshold 0.7 --alpha 1.06 --ratio 0.6
```

This calls the embedding service and LLM API. Other methods are `recent`, `relevance`, and `random`; use `--k 5` for their online sampling. `distill` summarizes behavior, and `pairwise` uses reflection. The legacy `pointwise` path is retained, but its current prediction parser is inconsistent with its prompt and is not recommended for evaluation.

`--data_dir` and `--result_dir` override the input and output roots. Their defaults are repository-relative, so they do not depend on the terminal's working directory. `--server_url`, `--model_name`, and `--api_key` override the server and LLM settings. `--api_key` takes precedence over `OPENAI_API_KEY`. Add `--interactive` to pause after each displayed persona.

Results are appended to `result/<setting>/validation.jsonl`, with each user's NDCG, Hit Rate, and MRR at 1, 5, and 10, and candidate scores. Use a fresh result directory for a new run. The launcher runs the original recent, relevance, and PersonaX distillation settings and forwards additional client arguments:

```bash
bash experiments/run.sh --dataset Books --subset 480
```

Set `PYTHON` to select the launcher's Python executable.

## Evaluation protocol

The supplied client uploads each user's full timestamp-sorted history, including the final interaction that it subsequently treats as the target. It ranks that target against up to nine randomly sampled items outside the user's interaction history. The ranker embeds the persona and candidate descriptions with EasyRec and scores them by cosine similarity. The original candidate selection and metric calculations are retained. The client seeds Python and NumPy with 42, but its set-based negative pool can have a different ordering between processes, so the seed alone does not ensure identical candidates.

The [paper](https://aclanthology.org/2025.findings-acl.300/) evaluates downstream AgentCF and Agent4Rec and reserves the latest item for evaluation. This client is a simple embedding harness with a different target-in-history protocol; running it does not reproduce those paper experiments. For a held-out evaluation, construct the training history and downstream agent integration explicitly. The client also omits `all_items_catalog`, so persona reflection pairs use the server's existing fallback sampling from selected positive items.

## Minimal API example

With both services running, this example constructs cached personas from a small history, then retrieves one for a target item. It makes LLM calls during ingestion.

```python
import os
import requests

server = "http://127.0.0.1:8001"
history = [
    {"item": {"item_id": "book-1", "title": "History of astronomy"},
     "rating": 5, "timestamp": 1},
    {"item": {"item_id": "book-2", "title": "Introduction to physics"},
     "rating": 5, "timestamp": 2},
]
response = requests.post(f"{server}/ingest_history", json={
    "PersonaX": True,
    "user_id": "example-user",
    "interactions": history,
    "persona_learning_type": "distill",
    "model_name": "gpt-4.1-nano-2025-04-14",
    "api_key": os.environ["OPENAI_API_KEY"],
    "distance_threshold": 0.7, "alpha": 1.06, "ratio": 0.6,
}, timeout=180)
response.raise_for_status()

response = requests.post(f"{server}/online_profile", json={
    "user_id": "example-user",
    "target_item": {"item_id": "book-3", "title": "Exploring the solar system"},
    "method": "personax",
}, timeout=60)
response.raise_for_status()
print(response.json()["user_profile"])
```

The cached `personax` retrieval needs only `user_id`, `target_item`, and `method`. The online baseline methods also require `model_name`, `api_key`, and optionally `k`. Ingestion accepts an optional `all_items_catalog` list for sampling negative items.

## Preprocessing and helpers

`preprocess_long.ipynb` retains its original cells and outputs. Before running it, set the notebook kernel's working directory to `Amazon/` (for example, `%cd /absolute/path/to/PersonaX/Amazon`). It expects `meta_Books.json` and `ratings_Books.csv` there and writes `sampled_long_org.csv` and `sampled_long.csv` there. These raw inputs are not supplied by the notebook.

`rerank.py` exposes the existing `compute_score` helper for the EasyRec `/compute_scores` endpoint. Plotting functions are in `plotting.py`. These helpers do not launch evaluation runs on import.
