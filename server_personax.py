# server_personax.py
import os
import json
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, request, jsonify
from core import distill, train   # ← 直接用你已有的核心实现
# 你项目里的工具
from utils import EasyRec, item_feature_to_str, sanitize_data, hierarchical_clustering
from sampling import sampling  # 你的按簇抽样函数

# ================== 持久化与全局对象 ==================
STORAGE_DIR = os.path.join(os.getcwd(), "storage")
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
    """余弦相似度：A (n x d) 每行与向量 b(d,) 的相似度 -> shape (n,)"""
    if A.ndim != 2: 
        A = A.reshape(len(A), -1)
    b = b.reshape(-1)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b) + 1e-12)
    return np.dot(An, bn)

def _ensure_user_history(user: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[float]]:
    """从 USER_DB 条目里拿到正样本及其时间戳；没有则返回空列表。"""
    pos_items: List[Dict[str, Any]] = user.get('pos_items', []) or []
    pos_ts: List[float] = user.get('pos_ts', []) or []
    return pos_items, pos_ts

USER_DB: Dict[str, Any] = _load_personas()
Embedder = EasyRec()

# ================== Flask App ==================
app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

# ------------------ 工具函数 ------------------
def _embed_items(items: List[Dict[str, Any]]) -> np.ndarray:
    texts = [item_feature_to_str(sanitize_data(it)) for it in items]
    embs = Embedder.get_embedding(texts)
    return np.array(embs)

def _server_cluster_and_sample(pos_item_list: List[Dict[str, Any]],
                               distance_threshold: float,
                               alpha: float,
                               ratio: float) -> Tuple[List[int], Dict[int, List[int]], Dict[int, List[float]], Dict[int, List[int]]]:
    """
    服务端：对正样本做 嵌入→层次聚类→按簇抽样
    返回：
      all_selected（全局被选正样本索引）,
      class2index_list（簇→原始索引列表）,
      class2centroid（簇→簇心向量）,
      selected_class2index_list（簇→本簇内被选的原始索引列表）
    """
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

    # 按簇抽样：sampling 接受每簇的 points（相对索引）
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
    """
    将被选中的正样本，配上负样本，组装为 persona_learning 的 sequence。
    规则：
      - 若提供了 all_items_catalog，则负样本从 “catalog - positives” 中随机采（更合理）。
      - 否则用“同域随机采非自身”的伪负样本兜底。
    输出的每个元素形如：
      { "pos_item": {...}, "neg_item": {...} }
    兼容你原来的 core 接口（会把 dict 原样传给 distill/train）。
    """
    seq = []
    pos_set_ids = set(pos_ids) if pos_ids else set()
    cat = all_items_catalog or []

    for pos in selected_pos_items:
        # 采负样本
        if cat:
            # 从 catalog 里选一个不在正集合里的
            candidates = [x for x in cat if (not pos_set_ids or x.get("item_id") not in pos_set_ids)]
            neg = random.choice(candidates) if candidates else random.choice(cat)
        else:
            # 退化兜底：从正样本里随机挑一个非自身
            pool = [x for x in selected_pos_items if x is not pos] or [pos]
            neg = random.choice(pool)
        seq.append({"pos_item": sanitize_data(pos), "neg_item": sanitize_data(neg)})
    return seq

# ------------------ 你现成的两个 API（原样并入） ------------------
@app.route('/persona_learning', methods=['POST'])
def personal_learning():
    """
    你的原始接口：直接调用 core.distill/train
    兼容两种 sequence 结构：
      1) [{"pos_item_id": "...","neg_item_id":"..."}]
      2) [{"pos_item": {...}, "neg_item": {...}}]
    """
    data = request.json
    typ = data['type']     # 'pointwise'/'pairwise' (Reflection) 或 'distill' (Summarization)
    user = data['user']
    sequence = data['sequence']

    model_name = data.get('model_name', None)
    api_key = data.get('api_key', None)
    if model_name is None:
        return jsonify({'error': 'model_name is required', 'status': 400})
    siliconflow = False if 'gpt' in model_name else True
    if api_key is None:
        return jsonify({'error': 'api_key is required', 'status': 400})

    # 直接把 sequence 透传给你的 core（若 core 只接受 id，可在 core 内部或这里做映射）
    if typ == 'distill':
        user_persona, log = distill(user, sequence, typ, model_name, siliconflow, api_key)
    else:
        user_persona, log = train(user, sequence, typ, model_name, siliconflow, api_key)

    return jsonify({'user_persona': user_persona, 'log': log, 'status': 200})





# ------------------ 新增：服务端“历史→聚类→抽样→学习” ------------------
@app.route('/ingest_history', methods=['POST'])
def ingest_history():
    """
    客户端上传完整历史交互，服务端完成：
      嵌入→层次聚类→每簇抽样→每簇 persona 学习（调用 core.distill/train）
    请求体字段：
      PersonaX: bool  # 是否使用PersonaX
      user_id: str
      interactions: [{item:{...}, rating: float, timestamp: float}, ...]   # 全部真实历史
      persona_learning_type: "distill" | "pointwise" | "pairwise"
      model_name, api_key: 透传给 core
      distance_threshold, alpha, ratio: 聚类/抽样参数（服务端使用）
      all_items_catalog: (可选) 全量物品库 [{...}, ...]，用于抽负样本（更合理）
    """
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
    siliconflow = False if 'gpt' in model_name else True

    distance_threshold = float(data.get('distance_threshold', 0.5))
    alpha = float(data.get('alpha', 1.06))
    ratio = float(data.get('ratio', 0.6))
    catalog = data.get('all_items_catalog', None)  # 用于更合理的负采样（可选）

    
    # 1) 取正样本 item 列表（rating >= 1）
    pos_items: List[Dict[str, Any]] = []
    pos_ids: List[str] = []
    for it in sorted(interactions, key=lambda x: x.get('timestamp', 0)):
        if (it.get('rating', 0) or 0) >= 1:
            item = sanitize_data(it['item'])
            pos_items.append(item)
            if 'item_id' in item:
                pos_ids.append(str(item['item_id']))

    if len(pos_items) == 0:
        # 建空档案
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
        # 不使用 PersonaX，仅存储正样本历史
        USER_DB[user_id] = {
            "pos_items": pos_items,
            "pos_ts": [float(it.get('timestamp', 0.0)) for it in interactions if (it.get('rating', 0) or 0) >= 1],
        }
        _save_personas(USER_DB)
        return jsonify({'user_id': user_id, 'clusters': 0, 'selected_per_cluster': {}})
    else:
        #查看一下user_id有没有cluster_profiles，不为空，就不需要重新计算
        existing_user = USER_DB.get(user_id, {})
        if existing_user.get('cluster_profiles'):
            return jsonify({
                'user_id': user_id,
                'clusters': len(existing_user.get('cluster_profiles', {})),
                'selected_per_cluster': existing_user.get('selected_count', {})
            })
        else:
            # 2) 服务端聚类 + 抽样
            all_selected, class2index_list, class2centroid, selected_class2index_list = _server_cluster_and_sample(
                pos_items, distance_threshold, alpha, ratio
            )
            
            #存储一下聚类的情况
            cluster_info = {
                "total_interactions": len(interactions),
                "total_positive_items": len(pos_items),
                "total_selected_items": len(all_selected),
                "total_clusters": len(class2index_list),
                "average_cluster_size": np.mean([len(v) for v in class2index_list.values()]) if class2index_list else 0,
                "average_selected_per_cluster": np.mean([len(v) for v in selected_class2index_list.values()]) if selected_class2index_list else 0,
            }
            #先存储一下
            USER_DB[user_id] = {
                "cluster_info": cluster_info,
            }
            _save_personas(USER_DB)
            

            # 3) 为每个簇构造 sequence（pos/neg 对）并调用 core.distill/train 得到簇 persona
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
                print(cluster_profiles[str(cid)])  # 调试时可见每簇画像
                print(selected_count[str(cid)])
                aa=input('--- Press Enter to continue ---')

            # 5) 存储
            USER_DB[user_id] = {
                "selected_count": selected_count,
                "cluster_profiles": cluster_profiles,
                "class2centroid": {str(k): v for k, v in class2centroid.items()},
                "meta": {
                    "distance_threshold": distance_threshold, "alpha": alpha, "ratio": ratio,
                    "clusters": len(class2index_list)
                },
                "cluster_info": cluster_info,  # 可选，存储一些聚类统计信息
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
        

# ------------------ 新增：线上按目标 item 选簇并返回 persona ------------------
@app.route('/online_profile', methods=['POST'])
def online_profile():
    """
    在线画像（与离线聚类无关）：
    输入：
      {
        "user_id": "...",
        "target_item": {...},                # relevance 需要；random/recent 可选
        "method": "random" | "recent" | "relevance" | "personax",
        "k": 5,                              # 采样条数，默认 5
        "persona_learning_type": "distill" | "pointwise" | "pairwise",
        "model_name": "gpt-4o-mini",         # 透传到 core
        "api_key": "...",                    # 透传到 core
        "all_items_catalog": [ {...}, ... ]  # 可选，用于负采样更合理
      }

    逻辑：
      1) 从 USER_DB 中取该 user 的历史正样本（pos_items/pos_ts）
      2) 按 method 抽取 k 条正样本
         - random: 全历史随机取 k
         - recent: 时间上最近的 k（默认按 pos_ts 升序，取最后 k 条）
         - relevance: 与 target_item 语义最近的 k（EasyRec 嵌入 + 余弦相似度 Top-k）
         - personax: 保持你原有的 personax（用簇心/全局）——为兼容，仍然保留
      3) 以这些正样本，构造 (pos, neg) 的 sequence（优先从 catalog 里采负样本）
      4) 调用 core.distill/train，得到在线 persona，并返回
    """
    data = request.json
    user_id = data['user_id']
    method = data.get('method', 'recent')  # 默认给个“recent”
    k = data.get('k', 5)
    target_item = sanitize_data(data.get('target_item', {}))
    learn_type = data.get('persona_learning_type', 'distill')
    model_name = data.get('model_name', None)
    api_key = data.get('api_key', None)
    catalog = data.get('all_items_catalog', None)

    # 参数校验（仅对会调用 persona_learning 的三种在线方法要求）
    if method in ('random', 'recent', 'relevance'):
        if not model_name:
            return jsonify({'error': 'model_name is required for online persona_learning', 'status': 400}), 400
        if not api_key:
            return jsonify({'error': 'api_key is required for online persona_learning', 'status': 400}), 400

    siliconflow = False if 'gpt' in model_name else True

    user = USER_DB.get(user_id)
    if not user:
        return jsonify({'error': f'user_id={user_id} not found. Please call /ingest_history first.', 'status': 404}), 404

    # personax 保留你的旧逻辑（与离线聚类绑定）
    if method == 'personax':
        class2centroid = {int(k): v for k, v in user.get('class2centroid', {}).items()}
        if not class2centroid:
            return jsonify({'user_id': user_id, 'method': 'personax', 'cluster_label': -1,
                            'user_profile': user.get('global_profile', 'Currently Unknown')})
        best_cid = _choose_cluster_by_target(class2centroid, target_item)  # 点积相似度
        persona = user['cluster_profiles'].get(str(best_cid), user.get('global_profile', 'Currently Unknown'))
        return jsonify({'user_id': user_id, 'method': 'personax', 'cluster_label': best_cid, 'user_profile': persona})

    # 其余三种方法：完全在线采样 + 在线 persona_learning
    pos_items, pos_ts = _ensure_user_history(user)
    if not pos_items:
        # 无历史；直接返回空档案
        return jsonify({'user_id': user_id, 'method': method, 'k': 0,
                        'user_profile': 'Currently Unknown', 'selected_count': 0})

    n = len(pos_items)
    k = max(1, min(k, n))  # 边界保护

    # ---- 采样 ----
    selected_idxs: List[int] = []

    if method == 'random':
        selected_idxs = random.sample(range(n), k)

    elif method == 'recent':
        # 假定 pos_ts 与 pos_items 一一对应；按时间升序 -> 取最后 k 条
        order = list(range(n))
        # 有时间戳就按时间；若时间戳缺失，用原顺序兜底
        try:
            order = sorted(range(n), key=lambda i: float(pos_ts[i]))
        except Exception:
            pass
        selected_idxs = order[-k:]

    elif method == 'relevance':
        # 与 target_item 最近的 k 次交互（语义空间）
        if not target_item:
            return jsonify({'error': 'target_item is required for method=relevance', 'status': 400}), 400
        # 嵌入
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

    # 稳定排序（按原时间顺序输出会更可读；你也可以按相似度降序输出）
    selected_pos_items = [pos_items[i] for i in selected_idxs]

    # 为负采样准备正样本 ID 集合
    pos_ids = []
    for it in selected_pos_items:
        if 'item_id' in it:
            pos_ids.append(str(it['item_id']))

    # 构造 sequence（复用你的工具，优先从 catalog 里采负样本）
    sequence = _make_sequence_pairs(selected_pos_items, catalog, pos_ids)

    # 在线 persona_learning（调用你已有 core 接口）
    user_stub = {"user_id": user_id, "user_persona": "Currently Unknown"}
    if learn_type == 'distill':
        persona, log = distill(user_stub, sequence, learn_type, model_name, siliconflow, api_key)
    else:
        persona, log = train(user_stub, sequence, learn_type, model_name, siliconflow, api_key)

    return jsonify({
        'user_profile': persona,
        'log': log
    })

    
    

# ------------------ 健康检查 ------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

# ------------------ 入口 ------------------
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8001, debug=False)
