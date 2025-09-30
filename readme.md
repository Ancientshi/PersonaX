



# PersonaX

**Architecture at a Glance**
```
                    Offline (per user)                          Online (per request)
┌───────────────────────────────────────────┐          ┌───────────────────────────────────────────┐
│                PersonaX Server            │          │           Recommendation Agent            │
│  (Flask, embeds + cluster + persona LLM)  │          │     (LocalRanker + candidate scoring)     │
│                                           │          │                                           │
│ 1. /ingest_history                        │          │ 5. Build candidate set (1 positive + neg) │
│    • Embed past positives (EasyRec)       │◄───┐     │ 6. Call /online_profile                   │
│    • Hierarchical clustering              │    │     │    • recent | relevance | random | personax│
│    • Per-cluster sampling (α, ratio)      │    │     │    • PersonaX distill/train (LLM)         │
│    • Per-cluster persona via LLM          │    │     │ 7. Get persona text                       │
│    • Persist cluster profiles & centroids │    │     │ 8. Rank candidates using persona (cosine) │
│                                           │    │     │ 9. Output top-K + metrics                 │
└───────────────────────────────────────────┘    │     └───────────────────────────────────────────┘
                   ▲                             │
                   │ 2. Store user history       │
User CSV ───────► client_agent.py ───────────────┘     Result: recommendation list (+ ndcg/hit/mrr)
```

**PersonaX ⇄ Recommendation Agent (what flows where)**

PersonaX turns raw interaction history into compact persona snippets (per-cluster offline; or sampled online).

The agent converts the persona text into an embedding and re-ranks candidate items with a simple cosine scorer (can be replaced by any downstream model).

**PersonaX API**
server_personax.py (Flask)
- /ingest_history: offline persona learning pipeline (embed → cluster → sample → LLM distill/reflection).
- /online_profile: online persona generation (traditional basic methods: recent/relevance/random) or retrieve cached profiles (built from clustered and sampled core behaviors when /ingest_history).

## Dataset
Dataset is provided in `Amazon` folder with the preprocessing script `preprocess_long.ipynb` for the Book480 subset (Similar procedure applies to the CDs dataset.). It includes Book480, CDs10, 50, and 200. These samples are drawn from the original Amazon dataset. The preprocessing script for  is available in preprocess_long.ipynb.

## How to run
1. https://github.com/HKUDS/EasyRec, put the code into EasyRec dir.
2. python app.py, launch the flask server for EasyRec embedding model.
3. python server_personax.py, lauch the flask server for PersonaX framework, used for modeling user persona.
4. bash run.sh


## Cite
If our work inspires your research, we would greatly appreciate your citation.
```
@inproceedings{shi-etal-2025-personax,
    title = "{P}ersona{X}: A Recommendation Agent-Oriented User Modeling Framework for Long Behavior Sequence",
    author = "Shi, Yunxiao  and
      Xu, Wujiang  and
      Zeqi, Zhang  and
      Zi, Xing  and
      Wu, Qiang  and
      Xu, Min",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.300/",
    doi = "10.18653/v1/2025.findings-acl.300",
    pages = "5764--5787",
    ISBN = "979-8-89176-256-5",
}
```

## Contact
Yunxiao.Shi@student.uts.edu.au