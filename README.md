# Precomputed BEIR Embeddings: BGE-Large & E5-Large

![Alternative Text for Image](./images/color_logo_transparent_cropped.png)


<p align="center">
    <a href="https://huggingface.co">
        <img alt="Hugging Face" src="https://img.shields.io/badge/Hosted%20on-Hugging%20Face-yellow?logo=huggingface&style=flat">
    </a>
    <a href="https://github.com/beir-cellar/beir">
        <img alt="BEIR Benchmark" src="https://img.shields.io/badge/Benchmark-BEIR-blue?style=flat">
    </a>
    <a href="https://github.com/beir-cellar/beir/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green?style=flat">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="#what-is-it">Overview</a> |
        <a href="#quick-download--usage">Quick Usage</a> |
        <a href="#available-embedding-corpora">Datasets</a> |
        <a href="#citing">Citing</a>
    </p>
</h4>

---

## 📦 What is it?

This repository provides **fully precomputed document embedding corpora** for **27 retrieval collections** from the BEIR benchmark (15 standalone collections and the 12 sub-datasets of CQADupStack). 

Generating these embeddings from scratch requires significant compute time and infrastructure. To help researchers, developers, and practitioners bypass this overhead, we are releasing the complete collection-specific representations generated using state-of-the-art dense models:
- **BGE-Large** (`BAAI/bge-large-en-v1.5`)
- **E5-Large** (`intfloat/e5-large-v2`)

These embeddings are tailored for tasks like constructing retrieval neighborhoods, evaluating representation spaces, late-interaction indexing, and semantic cohesion analyses.

---

## 📦 Quick Download & Usage

You can easily download and load the precomputed vectors into your Python pipeline using the `huggingface_hub` and `numpy` libraries.

```python
from huggingface_hub import hf_hub_download
import numpy as np
import json

# Replace with your actual Hugging Face repository ID
repo_id = "your-username/beir-precomputed-embeddings"
dataset_name = "scifact"
model_name = "bge-large"  # or "e5-large"

# Download ID mapping and numpy embedding matrix
ids_path = hf_hub_download(repo_id=repo_id, filename=f"{model_name}/{dataset_name}/corpus_ids.json")
emb_path = hf_hub_download(repo_id=repo_id, filename=f"{model_name}/{dataset_name}/corpus_embeddings.npy")

# Load data
with open(ids_path, "r") as f:
    corpus_ids = json.load(f)

corpus_embeddings = np.load(emb_path)

print(f"Successfully loaded {len(corpus_ids)} documents.")
print(f"Embedding matrix shape: {corpus_embeddings.shape}")
```

📦 Available Embedding Corpora

The table below outlines the 27 BEIR collections for which both BGE-Large and E5-Large document embedding matrices are provided.


| Collection Name | BEIR Identifier | Type | Document Count | BGE-Large Link | E5-Large Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MS MARCO | `msmarco` | Standalone | 8.84M | [Download](url-to-bge) | [Download](url-to-e5) |
| TREC-COVID | `trec-covid` | Standalone | 171K | [Download](url-to-bge) | [Download](url-to-e5) |
| NFCorpus | `nfcorpus` | Standalone | 3.6K | [Download](url-to-bge) | [Download](url-to-e5) |
| Natural Questions | `nq` | Standalone | 2.68M | [Download](url-to-bge) | [Download](url-to-e5) |
| HotpotQA | `hotpotqa` | Standalone | 5.23M | [Download](url-to-bge) | [Download](url-to-e5) |
| FiQA-2018 | `fiqa` | Standalone | 57K | [Download](url-to-bge) | [Download](url-to-e5) |
| TREC-NEWS | `trec-news` | Standalone (NIST) | 595K | [Download](url-to-bge) | [Download](url-to-e5) |
| Robust04 | `robust04` | Standalone (NIST) | 528K | [Download](url-to-bge) | [Download](url-to-e5) |
| ArguAna | `arguana` | Standalone | 8.67K | [Download](url-to-bge) | [Download](url-to-e5) |
| Touche-2020 | `webis-touche2020` | Standalone | 382K | [Download](url-to-bge) | [Download](url-to-e5) |
| Quora | `quora` | Standalone | 523K | [Download](url-to-bge) | [Download](url-to-e5) |
| DBpedia-Entity | `dbpedia-entity` | Standalone | 4.63M | [Download](url-to-bge) | [Download](url-to-e5) |
| SCIDOCS | `scidocs` | Standalone | 25K | [Download](url-to-bge) | [Download](url-to-e5) |
| FEVER | `fever` | Standalone | 5.42M | [Download](url-to-bge) | [Download](url-to-e5) |
| SciFact | `scifact` | Standalone | 5K | [Download](url-to-bge) | [Download](url-to-e5) |
| CQADupStack (12 subsets) | `cqadupstack` | Subsets (Android, English, etc.) | 457K total | [Download](url-to-bge) | [Download](url-to-e5) |


📦 Citing

If you use these precomputed embeddings or reference our methodology, please cite the official BEIR benchmark paper:

```
@inproceedings{
    thakur2021beir,
    title={{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
    author={Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
    booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
    year={2021},
    url={[https://openreview.net/forum?id=wCu6T5xFjeJ](https://openreview.net/forum?id=wCu6T5xFjeJ)}
}

```

### Next Steps to Launch:
1. Create a repository on Hugging Face (e.g., `your-username/beir-precomputed-embeddings`) or GitHub.
2. Upload your `corpus_ids.json` and `corpus_embeddings.npy` files into folders matching the structure (`bge-large/<dataset>/` and `e5-large/<dataset>/`).
3. Replace `"your-username/beir-precomputed-embeddings"` in the Python snippet with your actual repository ID!


### Collaboration
The BEIR Benchmark has been made possible due to a collaborative effort of the following universities and organizations:

![Alternative Text for Image](./images/National-Institute-of-Standards-and-Technology-nist.jpg)
![Alternative Text for Image](./images/Buffalo.jpg)

### Contributors

Thanks go to all these wonderful collaborations for their contribution towards the BEIR benchmark:

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/2ahmedabdullah">
        <img src="./images/abdul.jpeg" width="120px" alt="Abdul"/>
        <br />
        <sub><b>Abdul Ahmed</b></sub>
      </a>
    </td>

    <td align="center">
      <a href="https://github.com/YOUR_GITHUB_USERNAME">
        <img src="./images/madhukar.jpeg" width="120px" alt="Madhukar"/>
        <br />
        <sub><b>Madhukar Dwivedi</b></sub>
      </a>
    </td>
  </tr>
</table>