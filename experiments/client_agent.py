"""Evaluate cached or online personas with the original embedding ranker."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from experiments.metrics import hit, mrr, ndcg
from personax.utils import EasyRec, item_feature_to_str, sanitize_data, sanitize_value

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

seed = 42
random.seed(seed)
np.random.seed(seed)

class LocalRanker:
    def __init__(self):
        self.embedder = EasyRec()

    def rank(self, user: dict, items: list):
        persona = (user.get("user_persona") or "").strip()
        if persona:
            persona_vec = np.array(self.embedder.get_embedding([persona])[0], dtype=np.float32)

        item_vecs = np.array(self.embedder.get_embedding(
            [item_feature_to_str(sanitize_data(it)) for it in items]
        ), dtype=np.float32)
        if not persona:
            # A neutral persona gives all items the same score.
            persona_vec = np.zeros(item_vecs.shape[1], dtype=np.float32)

        # cosine similarity for consistency
        iv = item_vecs / (np.linalg.norm(item_vecs, axis=1, keepdims=True) + 1e-12)
        pv = persona_vec / (np.linalg.norm(persona_vec) + 1e-12)
        scores = (iv @ pv)  # [N,]
        index_list = np.argsort(-scores).tolist()
        return index_list, scores.tolist()



def build_interactions_frame(df_user: pd.DataFrame):
    """Convert one user's rows to the /ingest_history interaction schema."""
    interactions = []
    for _, row in df_user.sort_values(by='timestamp').iterrows():
        item = {k: sanitize_value(v) for k, v in row.drop(['rating', 'timestamp']).to_dict().items()}
        interactions.append({
            "item": item,
            "rating": float(row['rating']),
            "timestamp": float(row['timestamp'])
        })
    return interactions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=str, default='CDs_and_Vinyl')
    parser.add_argument('--subset', type=str, default='200')
    parser.add_argument('--data_dir', type=Path, default=REPOSITORY_ROOT / 'Amazon',
                        help='Directory containing the dataset subdirectories.')
    parser.add_argument('--result_dir', type=Path, default=REPOSITORY_ROOT / 'result',
                        help='Directory for per-setting validation files.')
    parser.add_argument('--interactive', action='store_true',
                        help='Pause after displaying each persona.')

    # PersonaX handles history ingestion and persona retrieval.
    parser.add_argument('--method', type=str, default='recent')
    parser.add_argument('--server_url', type=str, default='http://127.0.0.1:8001')
    parser.add_argument('--persona_learning_type', type=str, default='distill')
    parser.add_argument('--model_name', type=str, default='gpt-4.1-nano-2025-04-14')
    parser.add_argument('--api_key', type=str, default=os.environ.get('OPENAI_API_KEY', ''),
                        help='API key; defaults to OPENAI_API_KEY.')

    # Pass clustering and sampling settings through to the server.
    parser.add_argument('--distance_threshold', type=float, default=0.7)
    parser.add_argument('--alpha', type=float, default=1.06)
    parser.add_argument('--ratio', type=float, default=0.6)
    parser.add_argument('--k', type=int, default=5)

    args = parser.parse_args()
    args.PersonaX = True if args.method == 'personax' else False
    args.k = None if args.method == 'personax' else args.k

    dataset_path = args.data_dir / args.dataset / f'sampled_{args.subset}.csv'
    data = pd.read_csv(dataset_path)

    unique_users = data['user_id'].unique().tolist()
    unique_items = data['item_id'].unique().tolist()

    if args.method == 'personax':
        dirname = f'{args.method}_{args.persona_learning_type}_{args.dataset}_{args.subset}_{args.distance_threshold}_{args.alpha}_{args.ratio}'
    else:
        dirname = f'{args.method}_{args.persona_learning_type}_{args.dataset}_{args.subset}_{args.k}'
    outdir = args.result_dir / dirname
    outdir.mkdir(parents=True, exist_ok=True)
    val_path = outdir / 'validation.jsonl'

    ranker = LocalRanker()

    for user_id in unique_users:
        user_df = data[data['user_id'] == user_id].sort_values(by='timestamp')

        # Preserve the original protocol: ingest the full history, including the target.
        interactions = build_interactions_frame(user_df)
        ingest_req = {
            "PersonaX": args.PersonaX,
            "user_id": user_id,
            "interactions": interactions,
            "persona_learning_type": args.persona_learning_type,
            "model_name": args.model_name,
            "api_key": args.api_key,
            "distance_threshold": args.distance_threshold,
            "alpha": args.alpha,
            "ratio": args.ratio
        }
        try:
            _ = requests.post(f"{args.server_url}/ingest_history", json=ingest_req, timeout=180).json()
        except Exception as e:
            print(f"[ingest_history] {user_id} error: {e}")
            continue

        # Use the latest interaction as the target item.
        if len(user_df) == 0:
            continue
        target_row = user_df.iloc[-1]
        target_item = sanitize_data(target_row.drop(['rating','timestamp']).to_dict())

        # Retrieve a cached persona or construct one with an online baseline.
        try:
            prof = requests.post(
                f"{args.server_url}/online_profile",
                json={"user_id": user_id, "target_item": target_item, "method": args.method, "k": args.k,
                    "persona_learning_type": args.persona_learning_type, "model_name": args.model_name, "api_key": args.api_key},
                timeout=60
            ).json()
            chosen_profile = prof.get("user_profile", "Currently Unknown")
        except Exception as e:
            print(f"[online_profile] {user_id} error: {e}")
            chosen_profile = "Currently Unknown"

        print(f"User {user_id} Profile: {chosen_profile}")
        if args.interactive:
            input("Press Enter to continue...")
        # The candidate set is the target plus up to nine random unseen items.
        positive_items = user_df['item_id'].tolist()
        negative_candidates = set(unique_items) - set(positive_items)
        neg_samples = random.sample(list(negative_candidates), min(9, len(negative_candidates)))
        items = [target_item] + [
            sanitize_data(data[data['item_id'] == i].iloc[0].drop(['rating', 'timestamp']).to_dict())
            for i in neg_samples
        ]

        # Rank candidate embeddings by cosine similarity with the persona embedding.
        user_for_rank = {"user_id": user_id, "user_persona": chosen_profile}
        try:
            index_list, scores = ranker.rank(user_for_rank, items)
        except Exception as e:
            print(f"[local_rank] {user_id} error: {e}")
            continue

        truth = [[0]]
        pred = [index_list]
        topk = [1,5,10]
        ndcgs = ndcg(truth, pred, topk)
        hits = hit(truth, pred, topk)
        mrrs = mrr(truth, pred, topk)
        metrics = {
            'ndcg@1': ndcgs[0], 'ndcg@5': ndcgs[1], 'ndcg@10': ndcgs[2],
            'hit@1': hits[0], 'hit@5': hits[1], 'hit@10': hits[2],
            'mrr@1': mrrs[0], 'mrr@5': mrrs[1], 'mrr@10': mrrs[2]
        }
        print(f"{user_id}: {metrics}")

        with open(val_path, 'a') as f:
            f.write(json.dumps({
                'user_id': user_id,
                'metrics': metrics,
                'scores': scores
            })+'\n')


if __name__ == "__main__":
    main()
