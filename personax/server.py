"""HTTP service for offline persona construction and online retrieval."""
import os
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, request, jsonify
from .core import distill, train
from .utils import EasyRec, item_feature_to_str, sanitize_data, hierarchical_clustering
from .sampling import sampling

STORAGE_DIR = os.environ.get(
    "PERSONAX_STORAGE_DIR", str(Path(__file__).resolve().parents[1] / "storage")
)
os.makedirs(STORAGE_DIR, exist_ok=True)
PERSONA_PATH = os.path.join(STORAGE_DIR, "user_personas.json")

def _load_personas() -> Dict[str, Any]:
    if os.path.exists(PERSONA_PATH):
        with open(PERSONA_PATH, "r") as f:
            return json.load(f)
    return {}

def _save_personas(d: Dict[str, Any]) -> None:
    with open(PERSONA_PATH, "w") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def _cosine_sim_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return cosine similarities between matrix rows and one vector."""
    if A.ndim != 2:
        A = A.reshape(len(A), -1)
    b = b.reshape(-1)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b) + 1e-12)
    return np.dot(An, bn)

def _ensure_user_history(user: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Return stored positive items and their timestamps."""
    pos_items: List[Dict[str, Any]] = user.get('pos_items', []) or []
    pos_ts: List[float] = user.get('pos_ts', []) or []
    return pos_items, pos_ts

USER_DB: Dict[str, Any] = _load_personas()
Embedder = EasyRec()

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

def _embed_items(items: List[Dict[str, Any]]) -> np.ndarray:
    texts = [item_feature_to_str(sanitize_data(it)) for it in items]
    embs = Embedder.get_embedding(texts)
    return np.array(embs)

def _server_cluster_and_sample(pos_item_list: List[Dict[str, Any]],
                               distance_threshold: float,
                               alpha: float,
                               ratio: float) -> Tuple[List[int], Dict[int, List[int]], Dict[int, List[float]], Dict[int, List[int]]]:
    """Embed, cluster, and sample positive items; return original indices and centroids."""
    if len(pos_item_list) == 0:
        return [], {}, {}, {}
    if len(pos_item_list) == 1:
        emb = _embed_items([pos_item_list[0]])[0]
        return [0], {1: [0]}, {1: emb.tolist()}, {1: [0]}

    embs = _embed_items(pos_item_list)  # N x D
    class2index_list, _ = hierarchical_clustering(embs, distance_threshold=distance_threshold)

    cluster_list, class2centroid = [], {}
    for cid, idxs in class2index_list.items():
        pts = embs[idxs]
        cluster_list.append(pts)
        class2centroid[cid] = np.mean(pts, axis=0).tolist()

    selected_rel = sampling(cluster_list, alpha=alpha, ratio=ratio)
    selected_class2index_list: Dict[int, List[int]] = {}
    for (cid, original_idxs), rel_idxs in zip(class2index_list.items(), selected_rel):
        selected_class2index_list[cid] = [original_idxs[i] for i in rel_idxs]

    all_selected = []
    for lst in selected_class2index_list.values():
        all_selected += lst

    return all_selected, class2index_list, class2centroid, selected_class2index_list

def _choose_cluster_by_target(class2centroid, target_item):
    if not class2centroid:
        return -1
    target_emb = _embed_items([target_item])[0]
    labels = list(class2centroid.keys())
    cents = np.array([class2centroid[c] for c in labels])
    # cosine
    cents_n = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)
    tgt_n = target_emb / (np.linalg.norm(target_emb) + 1e-12)
    scores = np.dot(cents_n, tgt_n)
    return labels[int(np.argmax(scores))]

def _make_sequence_pairs(selected_pos_items: List[Dict[str, Any]],
                         all_items_catalog: Optional[List[Dict[str, Any]]] = None,
                         pos_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Pair selected positives with catalog negatives, or with other selected items as a fallback."""
    seq = []
    pos_set_ids = set(pos_ids) if pos_ids else set()
    cat = all_items_catalog or []

    for pos in selected_pos_items:

        if cat:

            candidates = [x for x in cat if (not pos_set_ids or x.get("item_id") not in pos_set_ids)]
            neg = random.choice(candidates) if candidates else random.choice(cat)
        else:

            pool = [x for x in selected_pos_items if x is not pos] or [pos]
            neg = random.choice(pool)
        seq.append({"pos_item": sanitize_data(pos), "neg_item": sanitize_data(neg)})
    return seq

@app.route('/persona_learning', methods=['POST'])
def personal_learning():
    """Generate a persona from a user and a sequence of pos_item/neg_item dictionaries."""
    data = request.json
    typ = data['type']
    user = data['user']
    sequence = data['sequence']

    model_name = data.get('model_name', None)
    api_key = data.get('api_key', None)
    if model_name is None:
        return jsonify({'error': 'model_name is required', 'status': 400})
    siliconflow = bool(model_name and 'gpt' not in model_name)
    if api_key is None:
        return jsonify({'error': 'api_key is required', 'status': 400})

    if typ == 'distill':
        user_persona, log = distill(user, sequence, typ, model_name, siliconflow, api_key)
    else:
        user_persona, log = train(user, sequence, typ, model_name, siliconflow, api_key)

    return jsonify({'user_persona': user_persona, 'log': log, 'status': 200})

@app.route('/ingest_history', methods=['POST'])
def ingest_history():
    """Store interactions and optionally build cached personas for each sampled cluster."""
    data = request.json
    PersonaX = data.get('PersonaX', False)
    user_id = data['user_id']
    interactions = data['interactions']
    learn_type = data.get('persona_learning_type', 'distill')
    model_name = data.get('model_name')
    api_key = data.get('api_key')

    if not model_name:
        return jsonify({'error': 'model_name is required', 'status': 400})
    if not api_key:
        return jsonify({'error': 'api_key is required', 'status': 400})
    siliconflow = bool(model_name and 'gpt' not in model_name)

    distance_threshold = float(data.get('distance_threshold', 0.5))
    alpha = float(data.get('alpha', 1.06))
    ratio = float(data.get('ratio', 0.6))
    catalog = data.get('all_items_catalog', None)

    pos_items: List[Dict[str, Any]] = []
    pos_ids: List[str] = []
    for it in sorted(interactions, key=lambda x: x.get('timestamp', 0)):
        if (it.get('rating', 0) or 0) >= 1:
            item = sanitize_data(it['item'])
            pos_items.append(item)
            if 'item_id' in item:
                pos_ids.append(str(item['item_id']))

    if len(pos_items) == 0:

        USER_DB[user_id] = {
            "selected_count": {},
            "cluster_profiles": {},
            "class2centroid": {},
            "meta": {
                "distance_threshold": distance_threshold, "alpha": alpha, "ratio": ratio,
                "clusters": 0
            },
            "pos_items": [],
            "pos_ts": [],
        }
        _save_personas(USER_DB)
        return jsonify({'user_id': user_id, 'clusters': 0, 'selected_per_cluster': {}})

    if not PersonaX:

        USER_DB[user_id] = {
            "pos_items": pos_items,
            "pos_ts": [float(it.get('timestamp', 0.0)) for it in interactions if (it.get('rating', 0) or 0) >= 1],
        }
        _save_personas(USER_DB)
        return jsonify({'user_id': user_id, 'clusters': 0, 'selected_per_cluster': {}})
    else:

        existing_user = USER_DB.get(user_id, {})
        if existing_user.get('cluster_profiles'):
            return jsonify({
                'user_id': user_id,
                'clusters': len(existing_user.get('cluster_profiles', {})),
                'selected_per_cluster': existing_user.get('selected_count', {})
            })
        else:

            all_selected, class2index_list, class2centroid, selected_class2index_list = _server_cluster_and_sample(
                pos_items, distance_threshold, alpha, ratio
            )

            cluster_info = {
                "total_interactions": len(interactions),
                "total_positive_items": len(pos_items),
                "total_selected_items": len(all_selected),
                "total_clusters": len(class2index_list),
                "average_cluster_size": np.mean([len(v) for v in class2index_list.values()]) if class2index_list else 0,
                "average_selected_per_cluster": np.mean([len(v) for v in selected_class2index_list.values()]) if selected_class2index_list else 0,
            }

            USER_DB[user_id] = {
                "cluster_info": cluster_info,
            }
            _save_personas(USER_DB)

            cluster_profiles: Dict[str, str] = {}
            selected_count: Dict[str, int] = {}

            for cid, idxs in selected_class2index_list.items():
                selected_pos_items = [pos_items[i] for i in idxs]
                sequence = _make_sequence_pairs(selected_pos_items, catalog, pos_ids)

                user_stub = {"user_id": user_id, "user_persona": "Currently Unknown"}
                if learn_type == 'distill':
                    persona, log = distill(user_stub, sequence, learn_type, model_name, siliconflow, api_key)
                else:
                    persona, log = train(user_stub, sequence, learn_type, model_name, siliconflow, api_key)

                cluster_profiles[str(cid)] = persona
                selected_count[str(cid)] = len(idxs)

                print('--- Cluster', cid, '---')
                print(cluster_profiles[str(cid)])
                print(selected_count[str(cid)])

            USER_DB[user_id] = {
                "selected_count": selected_count,
                "cluster_profiles": cluster_profiles,
                "class2centroid": {str(k): v for k, v in class2centroid.items()},
                "meta": {
                    "distance_threshold": distance_threshold, "alpha": alpha, "ratio": ratio,
                    "clusters": len(class2index_list)
                },
                "cluster_info": cluster_info,
                # 👇 persist raw history for online methods
                "pos_items": pos_items,
                "pos_ts": [float(it.get('timestamp', 0.0)) for it in interactions if (it.get('rating', 0) or 0) >= 1],
            }
            _save_personas(USER_DB)

            return jsonify({
                'user_id': user_id,
                'clusters': len(class2index_list),
                'selected_per_cluster': selected_count,
            })

@app.route('/online_profile', methods=['POST'])
def online_profile():
    """Retrieve a cached PersonaX snippet, or learn from recent, relevant, or random history."""
    data = request.json
    user_id = data['user_id']
    method = data.get('method', 'recent')
    k = data.get('k', 5)
    target_item = sanitize_data(data.get('target_item', {}))
    learn_type = data.get('persona_learning_type', 'distill')
    model_name = data.get('model_name', None)
    api_key = data.get('api_key', None)
    catalog = data.get('all_items_catalog', None)

    if method in ('random', 'recent', 'relevance'):
        if not model_name:
            return jsonify({'error': 'model_name is required for online persona_learning', 'status': 400}), 400
        if not api_key:
            return jsonify({'error': 'api_key is required for online persona_learning', 'status': 400}), 400

    siliconflow = bool(model_name and 'gpt' not in model_name)

    user = USER_DB.get(user_id)
    if not user:
        return jsonify({'error': f'user_id={user_id} not found. Please call /ingest_history first.', 'status': 404}), 404

    if method == 'personax':
        class2centroid = {int(k): v for k, v in user.get('class2centroid', {}).items()}
        if not class2centroid:
            return jsonify({'user_id': user_id, 'method': 'personax', 'cluster_label': -1,
                            'user_profile': user.get('global_profile', 'Currently Unknown')})
        best_cid = _choose_cluster_by_target(class2centroid, target_item)
        persona = user['cluster_profiles'].get(str(best_cid), user.get('global_profile', 'Currently Unknown'))
        return jsonify({'user_id': user_id, 'method': 'personax', 'cluster_label': best_cid, 'user_profile': persona})

    pos_items, pos_ts = _ensure_user_history(user)
    if not pos_items:

        return jsonify({'user_id': user_id, 'method': method, 'k': 0,
                        'user_profile': 'Currently Unknown', 'selected_count': 0})

    n = len(pos_items)
    k = max(1, min(k, n))

    selected_idxs: List[int] = []

    if method == 'random':
        selected_idxs = random.sample(range(n), k)

    elif method == 'recent':

        order = list(range(n))

        try:
            order = sorted(range(n), key=lambda i: float(pos_ts[i]))
        except Exception:
            pass
        selected_idxs = order[-k:]

    elif method == 'relevance':

        if not target_item:
            return jsonify({'error': 'target_item is required for method=relevance', 'status': 400}), 400

        pos_texts = [item_feature_to_str(sanitize_data(it)) for it in pos_items]
        pos_embs = np.array(Embedder.get_embedding(pos_texts))  # (n, d)
        tgt_emb = _embed_items([target_item])[0]                # (d,)
        sims = _cosine_sim_matrix(pos_embs, tgt_emb)            # (n,)

        # Top-k indices
        order = np.argsort(-sims)[:k]
        selected_idxs = order.tolist()   # keep similarity order
    else:
        return jsonify({'error': f"unknown method: {method}. Choose from 'random','recent','relevance','personax'.",
                        'status': 400}), 400

    selected_pos_items = [pos_items[i] for i in selected_idxs]

    pos_ids = []
    for it in selected_pos_items:
        if 'item_id' in it:
            pos_ids.append(str(it['item_id']))

    sequence = _make_sequence_pairs(selected_pos_items, catalog, pos_ids)

    user_stub = {"user_id": user_id, "user_persona": "Currently Unknown"}
    if learn_type == 'distill':
        persona, log = distill(user_stub, sequence, learn_type, model_name, siliconflow, api_key)
    else:
        persona, log = train(user_stub, sequence, learn_type, model_name, siliconflow, api_key)

    return jsonify({
        'user_profile': persona,
        'log': log
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8001, debug=False)
