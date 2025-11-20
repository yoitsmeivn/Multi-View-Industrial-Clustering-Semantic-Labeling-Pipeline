# Multi‑View Spectral Clustering & LLM Taxonomy Pipeline
### Advanced Industrial Taxonomy Builder (Broad → Sub → Niche)

This project implements a **full multi‑view spectral clustering system**, combined with **LLM‑powered label generation**, for building clean industrial taxonomies from messy company‑level data.

It goes far beyond standard sklearn spectral clustering — this pipeline includes:

- Multi‑view embeddings (Qwen3‑Embedding‑0.6B)
- PCA compression
- Adaptive k‑NN graphs per view
- Heat kernel affinities
- Shared‑Nearest‑Neighbor (SNN) graph fusion
- Laplacian construction per view
- **True spectral embedding** using eigen‑decomposition
- KMeans on spectral space
- Hierarchical clustering (Broad → Sub → Niche)
- LLM labeling (GPT‑4o‑mini)
- Excel export with reproducible cache

---

# 🚀 Features

## 1. **Multi‑View Embeddings**
The model constructs multiple semantic “views” of each company:

- **v1:** AI summary + services  
- **v2:** roles + business model + product categories  
- **v3:** market focus + segments + certifications  
- **v4:** company name + industry + domain  
- **v5:** full metadata fusion  
- **vloc:** location embedding  

Embeddings use:

```
Qwen/Qwen3-Embedding-0.6B
```

with automatic fallback if GPU memory is low.

---

## 2. **Advanced Spectral Graph Construction**
This project **implements spectral clustering from scratch**, including:

### ✔ Adaptive k‑NN graphs  
Different neighborhood sizes per cluster level.

### ✔ Heat kernel affinity  

### ✔ Mutual k‑NN graph  

### ✔ SNN similarity graph  

### ✔ View‑weighted Laplacian fusion  
Each view contributes fractionally to the Laplacian.

### ✔ Multi‑view Laplacian  
The final Laplacian is:

```
L = Σ (view_weight_i × L_i)
```

---

## 3. **True Spectral Embedding**
Spectral embeddings are computed using eigen‑decomposition:

```
eigsh(L, k, which="SM")
```

This gives low‑dimensional embeddings that preserve cluster structure.

This is **real spectral clustering**, not k-means on embeddings or sklearn wrappers.

---

## 4. **Hierarchical Clustering Levels**
### **Broad Level**
High‑level industrial sectors.

### **Sub Level**
Mid‑granularity industry clusters.

### **Niche Level**
Fine‑grained professional categories.

All levels include:

- silhouette‑based k‑selection  
- imbalance penalties  
- centroid‑refinement  
- duplicate‑cluster merging  
- small‑cluster merging  

---

# 🧠 LLM Labeling
Cluster labels are generated using:

```
gpt-4o-mini
```

with strict rules:

- Broad → umbrella sector labels  
- Sub/Niche → “Head + Role Suffix”  
- Optional suffix handling:
  - Manufacturers  
  - Integrators  
  - Consultants  
  - Labs  
  - Distributors  
  - Service Providers  
  - Platforms  
  - etc.

A caching system ensures reproducibility across runs.

---

# 📦 Input
CSV file such as:

```
/content/drive/MyDrive/scraped_outputs/crawlaisummary.csv
```

Required fields (if available):

- company_name  
- ai_summary  
- services  
- roles  
- industry  
- business_model  
- certifications  
- keywords  
- location  

---

# 📤 Output
Excel file:

```
mv_semantic_labels.xlsx
```

With sheets:

- **Clusters**  
- **label_cache**  

Also includes:

- `cluster_label_broad`  
- `cluster_label_sub`  
- `cluster_label_niche`  
- broad_dim*, sub_dim*, niche_dim*  
- location dims  
- fused view dims  

---

# 📊 Export Example
```
Top Broad Clusters
Top Sub Clusters
Top Niche Clusters
```

---

# 🔧 Requirements
- Python 3.10+
- PyTorch with CUDA (optional)
- HuggingFace Transformers
- SciPy / scikit‑learn
- Pandas
- OpenAI Python SDK

---

# ▶️ Running
Place the script inside Google Colab and run:

```python
!pip install torch transformers openai scikit-learn pandas scipy
```

Ensure your input CSV is in the correct Drive path.

---

# 📄 License
For personal and research use.  
Commercial licensing available upon request.

---

# 🙌 Credits
Built for large‑scale industrial dataset organization, clustering, and semantic enrichment.
