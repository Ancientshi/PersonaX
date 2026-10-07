# PersonaX

Code for [PersonaX: A Recommendation Agent-Oriented User Modeling Framework for Long Behavior Sequence](https://aclanthology.org/2025.findings-acl.300/) (Findings of ACL 2025).

PersonaX clusters a user's interaction history, selects representative behaviors from each cluster, and generates persona snippets offline. At inference time, it retrieves the snippet most relevant to a target item. The released sampling, profiling prompts, and retrieval logic are kept in `personax/`.

## Layout

- `personax/`: clustering, sampling, persona learning, prompts, and the HTTP service.
- `experiments/`: recommendation client, ranking and metric helpers, launch script, EasyRec adapter, and preprocessing notebook.
- `Amazon/`: existing Books and CDs & Vinyl data subsets.

## Installation

Use Python 3.10 or 3.11 in a virtual environment:

```bash
git clone https://github.com/Ancientshi/PersonaX.git
cd PersonaX
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The core sampling code operates on supplied embeddings. The HTTP profiling service additionally needs an embedding service and a language-model API key; see [experiment setup](experiments/README.md).

## Minimal usage

Select behaviors from clusters of embeddings without a model or API call:

```python
import numpy as np
from personax.sampling import sampling

clusters = [
    np.array([[0.0, 0.0], [0.2, 0.1], [0.4, 0.0]]),
    np.array([[1.0, 1.0], [1.1, 1.2], [1.3, 1.0]]),
]
selected_indices = sampling(clusters, alpha=1.06, ratio=0.6)
# Each list contains indices relative to its input cluster.
```

The [sampling parameter demo](experiments/sampling_demo/README.md) shows how `alpha`, sample count, and input scale affect selection on fixed synthetic points. Download the [self-contained HTML](experiments/sampling_demo/select-samples-explorer.html) and open it in a browser, or view the [comparison figure](experiments/sampling_demo/select-samples-comparison.png).

For the full workflow, start `python -m personax.server`, upload history to `/ingest_history`, then retrieve a cached snippet through `/online_profile` with `method="personax"`. Request fields and experiment commands are in [experiments/README.md](experiments/README.md).

## Experiments

The provided client compares PersonaX with recent, relevance, and random sampling, using an EasyRec cosine ranker. Its protocol is described in [experiments/README.md](experiments/README.md). It is not a complete reproduction of the paper's AgentCF and Agent4Rec experiments.

## Citation

```bibtex
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

## License and contact

Original PersonaX code and documentation are released under the [MIT License](LICENSE). Data in `Amazon/`, data included in notebook outputs, the EasyRec adapter (`experiments/easyrec/app.py`), and other third-party code, dependencies, and model weights are excluded. They remain subject to their original terms; see [NOTICE](NOTICE) for the EasyRec source attribution.

Contact: Yunxiao.Shi@student.uts.edu.au
