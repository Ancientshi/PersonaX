# client_agent.py
import os, json, random, argparse
import numpy as np
import pandas as pd
import requests

# 从 utils 里再引入 EasyRec / item_feature_to_str 用于本地排序
from utils import (
    sanitize_value, sanitize_data, ndcg, hit, mrr,
    EasyRec, item_feature_to_str
)

seed = 42
random.seed(seed)
np.random.seed(seed)

# ========= 新增：本地排序器 =========
class LocalRanker:
    def __init__(self):
        self.embedder = EasyRec()

    def rank(self, user: dict, items: list):
        persona = (user.get("user_persona") or "").strip()
        if not persona:
            # fallback: neutral vector (zeros) -> equal scores -> stable by index
            persona_vec = np.zeros((self.embedder.dim,), dtype=np.float32)
        else:
            persona_vec = np.array(self.embedder.get_embedding([persona])[0], dtype=np.float32)

        item_vecs = np.array(self.embedder.get_embedding(
            [item_feature_to_str(sanitize_data(it)) for it in items]
        ), dtype=np.float32)

        # cosine similarity for consistency
        iv = item_vecs / (np.linalg.norm(item_vecs, axis=1, keepdims=True) + 1e-12)
        pv = persona_vec / (np.linalg.norm(persona_vec) + 1e-12)
        scores = (iv @ pv)  # [N,]
        index_list = np.argsort(-scores).tolist()
        return index_list, scores.tolist()



def build_interactions_frame(df_user: pd.DataFrame):
    """把单用户数据行转换为服务端 /ingest_history 所需的 interactions 列表。"""
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CDs_and_Vinyl')
    parser.add_argument('--subset', type=str, default='200')

    # 服务端（仅用于 /ingest_history 和 /online_profile）
    #method
    parser.add_argument('--method', type=str, default='recent') # 可选 personax, recent, relevance, random 三种方法
    parser.add_argument('--server_url', type=str, default='http://127.0.0.1:8001')
    parser.add_argument('--persona_learning_type', type=str, default='distill')
    parser.add_argument('--model_name', type=str, default='gpt-4.1-nano-2025-04-14')
    parser.add_argument('--api_key', type=str, default='')

    # 这些是“应由服务端负责”的抽样/聚类参数——客户端仅**透传**到 /ingest_history
    parser.add_argument('--distance_threshold', type=float, default=0.7)
    parser.add_argument('--alpha', type=float, default=1.06)
    parser.add_argument('--ratio', type=float, default=0.6)
    parser.add_argument('--k', type=int, default=5)  # 用于 non-personax 方法的在线画像生成

    args = parser.parse_args()
    args.PersonaX = True if args.method == 'personax' else False
    args.k = None if args.method == 'personax' else args.k

    # 读数据
    dataset_path = f'Amazon/{args.dataset}/sampled_{args.subset}.csv'
    data = pd.read_csv(dataset_path)

    unique_users = data['user_id'].unique().tolist()
    unique_items = data['item_id'].unique().tolist()

    # 评测输出目录
    if args.method == 'personax':
        dirname = f'{args.method}_{args.persona_learning_type}_{args.dataset}_{args.subset}_{args.distance_threshold}_{args.alpha}_{args.ratio}'
    else:
        dirname = f'{args.method}_{args.persona_learning_type}_{args.dataset}_{args.subset}_{args.k}'
    outdir = os.path.join('result', dirname)
    os.makedirs(outdir, exist_ok=True)
    val_path = os.path.join(outdir, 'validation.jsonl')

    # 初始化本地排序器
    ranker = LocalRanker()

    for user_id in unique_users:
        user_df = data[data['user_id'] == user_id].sort_values(by='timestamp')

        # 1) 离线：把完整历史交互发给 PersonaX, 让服务端PersonaX负责建模用户画像
        interactions = build_interactions_frame(user_df)
        ingest_req = {
            "PersonaX": args.PersonaX, # 是否使用PersonaX
            "user_id": user_id,
            "interactions": interactions,
            "persona_learning_type": args.persona_learning_type, # 有distill, reflection 两种
            "model_name": args.model_name, #使用的 LLM 模型
            "api_key": args.api_key, # API Key
            "distance_threshold": args.distance_threshold,  # 层次聚类的距离阈值
            "alpha": args.alpha,  # 用于平衡prototype和diversity的系数，越接近于1越重视prototype
            "ratio": args.ratio   # 数据选择的比例
        }
        try:
            _ = requests.post(f"{args.server_url}/ingest_history", json=ingest_req, timeout=180).json()
        except Exception as e:
            print(f"[ingest_history] {user_id} error: {e}")
            continue

        # 2) 线上：取最后一个为“目标 item”
        if len(user_df) == 0:
            continue
        target_row = user_df.iloc[-1]
        target_item = sanitize_data(target_row.drop(['rating','timestamp']).to_dict())

        # 3) 线上：拿到“与目标 item 相关的 User Profile”（仍从服务端获取画像文本）
        try:
            '''如果是使用PersonaX, 参数k,persona_learning_type, model_name, api_key是不需要的；
            如果是使用其他方法, 需要传入参数k, persona_learning_type, model_name, api_key，进行在线画像生成'''
            prof = requests.post(
                f"{args.server_url}/online_profile",
                # 可选personax, recent, relevance, random三种方法
                json={"user_id": user_id, "target_item": target_item, "method": args.method, "k": args.k,
                    "persona_learning_type": args.persona_learning_type, "model_name": args.model_name, "api_key": args.api_key},
                timeout=60
            ).json()
            chosen_profile = prof.get("user_profile", "Currently Unknown")
        except Exception as e:
            print(f"[online_profile] {user_id} error: {e}")
            chosen_profile = "Currently Unknown"

        print(f"User {user_id} Profile: {chosen_profile}")
        aa=input("Press Enter to continue...")
        # 4) 构造评测候选（正样本 + 9 个随机负样本）
        positive_items = user_df['item_id'].tolist()
        negative_candidates = set(unique_items) - set(positive_items)
        neg_samples = random.sample(list(negative_candidates), min(9, len(negative_candidates)))
        items = [target_item] + [
            sanitize_data(data[data['item_id'] == i].iloc[0].drop(['rating', 'timestamp']).to_dict())
            for i in neg_samples
        ]

        # 5) **本地排序（替代服务端 /rank_local）**
        user_for_rank = {"user_id": user_id, "user_persona": chosen_profile}
        try:
            index_list, scores = ranker.rank(user_for_rank, items)
        except Exception as e:
            print(f"[local_rank] {user_id} error: {e}")
            continue

        # 6) 评测
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
                'scores': scores  # 如需复核可保留
            })+'\n')


if __name__ == "__main__":
    main()
