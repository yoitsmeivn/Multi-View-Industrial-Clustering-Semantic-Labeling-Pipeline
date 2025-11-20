from google.colab import drive
try:
    drive.flush_and_unmount()
except Exception:
    pass
drive.mount('/content/drive', force_remount=True)

import os, re, ast, math, json, time, random, string, warnings, gc, contextlib
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.sparse import csr_matrix, diags, eye as speye
from scipy.sparse.linalg import eigsh

import torch
from transformers import AutoTokenizer, AutoModel
warnings.filterwarnings("ignore")


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ------------------ IO ------------------
INFILE  = "/content/drive/MyDrive/scraped_outputs/crawlaisummary.csv"
OUTFILE = "/content/drive/MyDrive/scraped_outputs/mv_semantic_labels.xlsx"


random.seed(42); np.random.seed(42); torch.manual_seed(42)
CUDA = torch.cuda.is_available()
if CUDA:
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# Graph scales
BASE_K_BROAD = (6, 8, 10)
KRANGE_BROAD = (6, 14)
BASE_K_SUB   = (8,  12, 18)
BASE_K_NICHE = (6,  10, 14)

# Search k
KRANGE_SUB   = (6,  24)
KRANGE_NICHE = (4,  16)

# PCA
PCA_DIMS = {"v1":128, "v2":96, "v3":64, "v4":48, "v5":64, "vloc":32}

# View  sematnics here
VIEW_WEIGHTS = {"v1":0.55, "v2":0.16, "v3":0.10, "v5":0.08, "v4":0.06, "vloc":0.05}

# Embeddings
EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMB_MAXLEN = 384  # auto-backs off to 192 on OOM
EMB_BS = 192 if CUDA else 32

# LLM labeling
LLM_MODEL       = "gpt-4o-mini"
LLM_TEMPERATURE = 0.10
LLM_MAXTOK      = 48
LLM_BUDGET      = 1200
ALWAYS_CALL_LLM = True
RESPECT_CACHE   = True
CACHE_VERSION   = "mvsem-v10-labels-2025-09-26"

MAX_LABEL_WORDS = 5

# Base
NEEDED = [
    "company_name","website_url","ai_summary","industry","services","market_focus",
    "roles","customer_types","product_categories","customers_segments","business_model",
    "certifications","keywords","location"
]

# ------------------ LOAD ------------------
raw = pd.read_csv(INFILE)
cols = [c for c in NEEDED if c in raw.columns]
df = raw[cols].fillna("").astype(str).copy()
print(f"Loaded {len(df):,} rows; using {len(cols)} columns")

# ------------------ ------------------
PUNCT = str.maketrans({c:" " for c in string.punctuation})

STOPWORDS = set("""
a an and or the of for to with by in on at from into over under between among within across via per
as is are be been being about plus include includes including provide provides providing offered offering
solution solutions product products system systems equipment company ltd limited plc group global international
""".split())

GENERIC_BAN = {"general","misc","others","services","solutions","systems","products","company","group"}

ACRONYMS = {"BMS","PLC","SCADA","HVAC","NDT","PV","UAV","API","QA","QC","FM200","CO2","VMS","PSIM","HMI","DCS","RTU","VFD","LV","MV","RFID","CCTV","IP","RO","STP","ETP","AHU","VRF","VRV"}

def series_or_blank(df, col):
    return df[col].astype(str) if col in df.columns else pd.Series([""]*len(df), index=df.index, dtype=str)

def to_list(x):
    if isinstance(x, list): return [str(s).strip() for s in x if str(s).strip()]
    s = str(x).strip()
    if not s: return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list): return [str(t).strip() for t in v if str(t).strip()]
    except Exception:
        pass
    return [t.strip() for t in re.split(r"[;,|]|\t", s) if t.strip()]

def uniq(seq):
    seen=set(); out=[]
    for s in seq:
        s=str(s).strip()
        if not s or s in seen: continue
        seen.add(s); out.append(s)
    return out

def title_case_keep_acronyms(s):
    words = re.split(r"(\s+|&|/|-)", s.strip())
    out=[]
    for w in words:
        if not w or re.fullmatch(r"\s+|&|/|-", w):
            out.append(w); continue
        if w.upper() in ACRONYMS:
            out.append(w.upper())
        else:
            out.append(w.capitalize())
    return "".join(out).strip()

# ------------------ EMBEDDER ---------------
class HFMeanPooler:
    def __init__(self, model_name=EMB_MODEL, max_len=EMB_MAXLEN, prefer_math_attn=False):
        self.device = "cuda" if CUDA else "cpu"
        self.max_len = max_len
        self.prefer_math_attn = prefer_math_attn
        print(f"Loading embedder: {model_name} on {self.device} (max_len={self.max_len})")

        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        try:
            from transformers import BitsAndBytesConfig
            if CUDA:
                quant = BitsAndBytesConfig(load_in_4bit=True)
                self.model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    quantization_config=quant,
                    device_map="auto"
                )
            else:
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        except Exception:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

        self.attn_ctx = (
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
            if (self.device == "cuda" and self.prefer_math_attn) else contextlib.nullcontext()
        )

    @torch.no_grad()
    def encode(self, texts, bs=None, max_len=None):
        if max_len is None:
            max_len = self.max_len
        if bs is None:
            bs = 96 if (self.device == "cuda") else 16

        vecs = []

        def _try_encode(batch, bs_local, max_len_local):
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=max_len_local, return_tensors="pt")
            enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
            with self.attn_ctx:
                out = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).type_as(out)
            x = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            x = torch.nn.functional.normalize(x, p=2, dim=1)
            return x

        i = 0
        while i < len(texts):
            batch = texts[i:i+bs]
            try:
                x = _try_encode(batch, bs, max_len)
                vecs.append(x.cpu().numpy())
                i += bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()
                if bs > 12:
                    bs = max(12, bs // 2)
                    print(f"[OOM] Reducing batch size → {bs}")
                    continue
                if max_len > 192:
                    max_len = max(192, max_len // 2)
                    print(f"[OOM] Reducing max length → {max_len}")
                    continue
                if self.device == "cuda":
                    print("[OOM] Switching embeddings to CPU for remaining chunks")
                    self.device = "cpu"
                    self.model = self.model.to("cpu")
                    self.attn_ctx = contextlib.nullcontext()
                    bs = 8
                    continue
                raise
        arr = np.vstack(vecs).astype(np.float32)
        del vecs; gc.collect()
        return arr

def pca_reduce(X, dim):
    if X.shape[1] <= dim:
        return normalize(X, axis=1)
    P = PCA(n_components=dim, random_state=42)
    Y = P.fit_transform(X)
    return normalize(Y, axis=1)

# ------------------ views ------------
def build_views(df, emb: HFMeanPooler):
    views = {}

    def _embed_and_reduce(texts, dim, label):
        X = emb.encode(texts)
        Y = pca_reduce(X, dim)
        del X; gc.collect()
        if CUDA: torch.cuda.empty_cache()
        print(f"[view:{label}] shape={Y.shape}")
        return Y

    # v1: ai_summary + services
    ai   = series_or_blank(df, "ai_summary")
    svc  = series_or_blank(df, "services")
    v1_text = (ai + " || Services: " + svc).tolist()
    views["v1"] = _embed_and_reduce(v1_text, PCA_DIMS["v1"], "v1"); del v1_text

    # v2: roles + business_model + product_categories + keywords
    roles = series_or_blank(df,"roles")
    bmod  = series_or_blank(df,"business_model")
    pcats = series_or_blank(df,"product_categories")
    kw    = series_or_blank(df,"keywords")
    v2_text = ("Roles: " + roles + " || Business Model: " + bmod +
               " || Product Categories: " + pcats + " || Keywords: " + kw).tolist()
    views["v2"] = _embed_and_reduce(v2_text, PCA_DIMS["v2"], "v2"); del v2_text

    # v3: market_focus + customers_segments + certifications
    mfocus = series_or_blank(df,"market_focus")
    cseg   = series_or_blank(df,"customers_segments")
    certs  = series_or_blank(df,"certifications")
    v3_text = ("Market Focus: " + mfocus + " || Customer Segments: " + cseg +
               " || Certifications: " + certs).tolist()
    views["v3"] = _embed_and_reduce(v3_text, PCA_DIMS["v3"], "v3"); del v3_text

    # v4: name + industry + url domain
    cname = series_or_blank(df,"company_name")
    ind   = series_or_blank(df,"industry")
    wurl  = series_or_blank(df,"website_url").str.replace(r"https?://", "", regex=True)
    v4_text = ("Company: " + cname + " || Industry: " + ind + " || Domain: " + wurl).tolist()
    views["v4"] = _embed_and_reduce(v4_text, PCA_DIMS["v4"], "v4"); del v4_text

    # v5: mega fusion
    mkt   = series_or_blank(df,"market_focus")
    cust  = series_or_blank(df,"customers_segments")
    mega = (
        "Company: " + cname + " | Industry: " + ind + " | Roles: " + roles +
        " | Services: " + svc + " | Market Focus: " + mkt +
        " | Customer Segments: " + cust + " | Business Model: " + bmod +
        " | Keywords: " + kw + " | Summary: " + ai
    ).tolist()
    views["v5"] = _embed_and_reduce(mega, PCA_DIMS["v5"], "v5"); del mega

    # vloc
    vloc_text = series_or_blank(df, "location").tolist()
    views["vloc"] = _embed_and_reduce(vloc_text, PCA_DIMS["vloc"], "vloc"); del vloc_text

    if CUDA: torch.cuda.empty_cache()
    gc.collect()
    return views

# ------------------ spectral---------
def clamp_k(k, n): return max(1, min(int(k), n-1))
def clamp_kset(kset, n): return tuple(sorted({clamp_k(k, n) for k in kset}))

def adaptive_kset(n, base):
    scale = max(0.7, min(1.6, np.sqrt(max(n,1))/25.0))  # ~25 → 1.0
    ks = sorted({max(2, int(round(k*scale))) for k in base})
    return tuple(min(k, max(2, n-1)) for k in ks)

def kneighbors(X, k, metric='cosine'):
    n = X.shape[0]
    if n <= 1:
        return np.zeros((n,0)), np.zeros((n,0), dtype=int)
    k = clamp_k(k, n)
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X)
    dist, idx = nn.kneighbors(X)
    return dist[:,1:], idx[:,1:]  # drop self

def build_heat_affinity(X, k, sigma_k=7):
    n = X.shape[0]
    if n <= 1: return csr_matrix((n,n))
    dist, idx = kneighbors(X, k)
    sig_index = max(0, min(sigma_k - 1, dist.shape[1] - 1))
    sig = dist[:, sig_index] + 1e-12
    rows, cols, data = [], [], []
    for i in range(n):
        for jpos, j in enumerate(idx[i]):
            w = math.exp(- (dist[i, jpos] ** 2) / (sig[i] * sig[j]))
            rows.append(i); cols.append(j); data.append(w)
    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    return W

def mutualize(W): return W.minimum(W.T)

def build_snn(X, k):
    n = X.shape[0]
    if n <= 1: return csr_matrix((n,n))
    _, idx = kneighbors(X, k)
    neigh = [set(ix) for ix in idx]
    rows, cols, data = [], [], []
    for i in range(n):
        Ni = neigh[i]
        for j in idx[i]:
            if i==j: continue
            Nj = neigh[j]
            sim = len(Ni & Nj) / float(k)
            if sim>0:
                rows.append(i); cols.append(j); data.append(sim)
    W = csr_matrix((data, (rows, cols)), shape=(n,n))
    return W.maximum(W.T)

def view_laplacian_multi(X, kset):
    n = X.shape[0]
    if n <= 2: return speye(n)
    kset = clamp_kset(kset, n)
    Lsum = None
    for k in kset:
        W_heat = build_heat_affinity(X, k)
        Wm     = mutualize(W_heat)
        Wsnn   = build_snn(X, k)
        W = (0.65 * Wm) + (0.35 * Wsnn)
        deg = np.array(W.sum(1)).ravel()
        Dm12 = diags(1.0/np.sqrt(np.maximum(deg,1e-12)))
        L = speye(n) - (Dm12 @ W @ Dm12)
        Lsum = L if Lsum is None else (Lsum + L)
    return Lsum / float(len(kset))

def combined_laplacian(views, weights, kset):
    n = next(iter(views.values())).shape[0]
    Lsum = csr_matrix((n,n))
    for key, Xv in views.items():
        Lsum = Lsum + (weights[key] * view_laplacian_multi(Xv, kset))
    return Lsum

def spectral_embed(L, kmax):
    kmax = max(2, min(kmax, L.shape[0]-1, 48))
    vals, vecs = eigsh(L, k=kmax, which='SM')
    order = np.argsort(vals)
    Z = normalize(vecs[:,order], axis=1)
    return vals[order], Z

def choose_k(Z, kmin, kmax, max_big_frac=0.35, sample_cap=7000):
    n, d = Z.shape
    d_eff = max(2, d)
    kmin = max(2, min(kmin, d_eff))
    kmax = max(kmin, min(kmax, d_eff))
    idx = np.arange(n)
    if n > sample_cap:
        idx = np.random.choice(n, size=sample_cap, replace=False)

    best_k, best_score = None, -1e9
    for k in range(kmin, kmax + 1):
        if k >= len(idx): break
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            lab_small = km.fit_predict(Z[idx, :k])
            try:
                sil = silhouette_score(Z[idx, :k], lab_small, metric="cosine")
            except Exception:
                sil = -0.5
            _, counts = np.unique(lab_small, return_counts=True)
            big_frac = counts.max()/counts.sum()
            imb = counts.std() / (counts.mean() + 1e-9)
            score = sil + 0.13*math.log(k) - 0.07*imb - (0.35 if big_frac>max_big_frac else 0.0)
            if score > best_score:
                best_k, best_score = k, score
        except Exception:
            continue
    return int(best_k if best_k else min(max(2, d_eff), max(2, n-1), 3))

def refine_by_centroid(X, labels, margin=0.030, iters=1):
    lab = labels.copy()
    for _ in range(iters):
        uniq = np.unique(lab)
        C = np.vstack([X[lab==u].mean(0) for u in uniq])
        C = normalize(C, axis=1)
        sims = X @ C.T
        mp = {u:i for i,u in enumerate(uniq)}
        curr = np.array([mp[u] for u in lab])
        best = sims.max(1); best_lab = uniq[sims.argmax(1)]
        curr_sim = sims[np.arange(X.shape[0]), curr]
        move = (best - curr_sim) > margin
        lab[move] = best_lab[move]
    return lab

def concat_views(views, keys=("v1","v2","v3","v5","v4","vloc")):
    X = np.hstack([views[k] for k in keys if k in views])
    return normalize(X, axis=1)

def merge_small(labels, X, min_size=10):
    lab = labels.copy()
    uniq, cnt = np.unique(lab, return_counts=True)
    small = [u for u,c in zip(uniq,cnt) if c<min_size]
    if not small: return lab
    C = {}
    for u in uniq:
        C[u] = normalize(X[lab==u].mean(0, keepdims=True), axis=1)[0]
    for s in small:
        best, tgt = -1.0, None
        for u in uniq:
            if u==s or u in small: continue
            sim = float(C[s] @ C[u])
            if sim>best: best, tgt = sim, u
        if tgt is not None: lab[lab==s] = tgt
    uniq2 = sorted(np.unique(lab)); mp = {u:i for i,u in enumerate(uniq2)}
    return np.array([mp[u] for u in lab])

def merge_duplicates(labels, X, thr=0.998):
    lab = labels.copy()
    changed = True
    while changed:
        changed = False
        uniq = sorted(np.unique(lab))
        if len(uniq)<=1: break
        C = normalize(np.vstack([X[lab==u].mean(0) for u in uniq]), axis=1)
        best, pair = 0, None
        for i in range(len(uniq)-1):
            sims = C[i] @ C[i+1:].T
            if sims.size == 0: continue
            j_rel = int(np.argmax(sims))
            sim   = float(np.max(sims))
            if sim>=thr and sim>best:
                best, pair = sim, (uniq[i], uniq[i+1+j_rel])
        if pair:
            a,b = pair
            if (lab==a).sum() >= (lab==b).sum():
                lab[lab==b]=a
            else:
                lab[lab==a]=b
            changed=True
    uniq2 = sorted(np.unique(lab)); mp = {u:i for i,u in enumerate(uniq2)}
    return np.array([mp[u] for u in lab])

def merge_small_level(labels, X, level, n):
    if "Broad" in str(level):
        ms = max(12, int(0.003 * n)); thr = 0.998
    elif "Sub" in str(level):
        ms = max(8, int(0.002 * n));  thr = 0.999
    else:
        ms = max(6, int(0.002 * n));  thr = 0.9992
    lab = merge_small(labels, X, min_size=ms)
    lab = merge_duplicates(lab, X, thr=thr)
    return lab

def run_level(views, level, kset, krange):
    n = next(iter(views.values())).shape[0]
    kset = clamp_kset(kset, n)
    if n <= 2:
        return np.zeros(n, dtype=int), None

    L = combined_laplacian(views, VIEW_WEIGHTS, kset)
    _, Z = spectral_embed(L, kmax=min(krange[1], 32))

    k = choose_k(Z, kmin=krange[0], kmax=krange[1],
                 max_big_frac=0.35 if "Broad" in str(level) else (0.42 if "Sub" in str(level) else 0.50))
    nZ = Z.shape[0]
    k = max(2, min(k, Z.shape[1], nZ - 1))
    Zk = Z[:, :k]
    lab = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(Zk)

    # refine: v1 then concat
    lab = refine_by_centroid(views["v1"], lab, margin=0.025, iters=1)
    Xcat = concat_views(views)
    lab = refine_by_centroid(Xcat, lab, margin=0.015, iters=1)

    # merge undersized & near-duplicates
    lab = merge_small_level(lab, views["v1"], level, nZ)

    print(f"{level}: K={len(np.unique(lab))}, n={nZ}")
    return lab, Zk

# ------------------ Build ------------------
embedder = HFMeanPooler()
VIEWS = build_views(df, embedder)

# ------------------ BROAD -> SUB -> NICHE ------------------
labels_broad, Zb = run_level(VIEWS, "Broad", adaptive_kset(len(df), BASE_K_BROAD), KRANGE_BROAD)
df["cluster_broad"] = labels_broad
B = Zb.shape[1]
for d in range(B):
    df[f"broad_dim{d+1}"] = Zb[:, d]

labels_sub = np.full(len(df), -1, int)
sub_parent = {}
sub_counter = 0

sub_chunks = []
sub_maxdim = 0

for b in sorted(np.unique(labels_broad)):
    idx = np.where(labels_broad == b)[0]
    if len(idx) < 5:
        labels_sub[idx] = sub_counter
        sub_parent[sub_counter] = b
        sub_counter += 1
        continue
    views_sub = {k: VIEWS[k][idx] for k in VIEWS}
    kset_sub = adaptive_kset(len(idx), BASE_K_SUB)
    lab_s, Zs = run_level(views_sub, f"Sub[{b}]", kset_sub, KRANGE_SUB)

    for u in np.unique(lab_s):
        uidx = idx[lab_s == u]
        labels_sub[uidx] = sub_counter
        sub_parent[sub_counter] = b
        sub_counter += 1

    if Zs is not None and Zs.size:
        sub_chunks.append((idx, Zs))
        sub_maxdim = max(sub_maxdim, Zs.shape[1])

df["cluster_sub"] = labels_sub

if sub_maxdim > 0:
    Zsub_global = np.zeros((len(df), sub_maxdim), dtype=np.float32)
    for idx, Zs in sub_chunks:
        Zsub_global[idx, :Zs.shape[1]] = Zs
    for d in range(sub_maxdim):
        df[f"sub_dim{d+1}"] = Zsub_global[:, d]

labels_niche = np.full(len(df), -1, int)
niche_parent = {}
niche_counter = 0

niche_chunks = []
niche_maxdim = 0

for s in sorted(np.unique(labels_sub)):
    idx = np.where(labels_sub == s)[0]
    if len(idx) < 4:
        labels_niche[idx] = niche_counter
        niche_parent[niche_counter] = s
        niche_counter += 1
        continue
    views_n = {k: VIEWS[k][idx] for k in VIEWS}
    kset_n = adaptive_kset(len(idx), BASE_K_NICHE)
    lab_n, Zn = run_level(views_n, f"Niche[{s}]", kset_n, KRANGE_NICHE)

    uniq_local = sorted(np.unique(lab_n))
    mp = {u: (niche_counter + i) for i, u in enumerate(uniq_local)}
    labels_niche[idx] = np.array([mp[u] for u in lab_n])
    for u in uniq_local:
        niche_parent[mp[u]] = s
    niche_counter += len(uniq_local)

    if Zn is not None and Zn.size:
        niche_chunks.append((idx, Zn))
        niche_maxdim = max(niche_maxdim, Zn.shape[1])

df["cluster_niche"] = labels_niche

if niche_maxdim > 0:
    Zniche_global = np.zeros((len(df), niche_maxdim), dtype=np.float32)
    for idx, Zn in niche_chunks:
        Zniche_global[idx, :Zn.shape[1]] = Zn
    for d in range(niche_maxdim):
        df[f"niche_dim{d+1}"] = Zniche_global[:, d]

# ------------------ LABELING ------------------
from openai import OpenAI
client = None
llm_calls = 0
try:
    os.environ["OPENAI_API_KEY"] =  "" #insert OPENAI_API_KEY | WARNING! THIS IS DANGEROUS! HIDE THIS KEY IN secrets.env
        client = OpenAI()
    else:
        print("[WARN] OPENAI_API_KEY not set; using deterministic fallback only")
except Exception as e:
    print(f"[WARN] OpenAI not initialized: {e}")

# ------------------ UMBRELLA --------------

BROAD_UMBRELLAS = [
    "Testing & Inspection",
    "Fire & Life Safety",
    "Security & Surveillance",
    "Industrial Automation",
    "HVAC & Building Services",
    "Water & Environmental",
    "Power & Energy",
    "Marine & Offshore",
    "Defence & Security",
    "Real Estate & Property",
    "Publishing & Media",
    "Automotive Retail"
]


BROAD_MAP = {
    r"\b(calibration|metrology|ndt|non destructive|inspection|testing|laborator(y|ies))\b": "Testing & Inspection",
    r"\b(fire alarm|fire detection|fire suppression|sprinkler|fm200|novec|inergen|hydrant|smoke control)\b": "Fire & Life Safety",
    r"\b(cctv|video surveillance|access control|intrusion|perimeter|vms|psim|genetec|avigilon|milestone)\b": "Security & Surveillance",
    r"\b(plc|scada|rtu|dcs|hmi|automation|vfd|servo|robot|cobot|bms)\b": "Industrial Automation",
    r"\b(hvac|ventilation|ahu|vrf|vrv|chiller|boiler|heat pump|refrigeration)\b": "HVAC & Building Services",
    r"\b(wastewater|sewage|stp|etp|water treatment|reverse osmosis|ro|desalination|ultrafiltration)\b": "Water & Environmental",
    r"\b(power systems|substation|switchgear|mv|lv|power quality|solar pv|photovoltaic|inverter|renewable)\b": "Power & Energy",
    r"\b(marine|offshore|vessel|ship|yacht|nautical)\b": "Marine & Offshore",
    r"\b(defen[cs]e|military|security consulting|intelligence)\b": "Defence & Security",
    r"\b(real estate|property management|serviced accommodation|landlord|lettings|estate agent)\b": "Real Estate & Property",
    r"\b(publishing|small press|books|print production|media|journal)\b": "Publishing & Media",
    r"\b(motorhome|caravan|campervan|automotive dealer|used cars)\b": "Automotive Retail"
}

def umbrella_from_texts(texts):
    blob = " ".join(texts).lower()
    for pat, umbrella in BROAD_MAP.items():
        if re.search(pat, blob):
            return umbrella
    return "Specialist Sector"

def enforce_umbrella(label, texts):
    """Keep Broad labels as umbrella sectors only."""
    if not label or label.lower() in GENERIC_BAN:
        label = umbrella_from_texts(texts)
    #
    base = re.sub(r"[^a-z0-9]+"," ", label.lower()).strip()
    for umb in BROAD_UMBRELLAS:
        cand = re.sub(r"[^a-z0-9]+"," ", umb.lower()).strip()
        if cand in base or base in cand:
            return umb
    return title_case_keep_acronyms(label)


CANON_HEAD = {
    # Fire & Safety
    r"\b(fire alarm|addressable alarm|conventional alarm)\b": "Fire Alarm",
    r"\b(fire suppression|foam system|gas suppression|fm200|novec|inergen|co2 system)\b": "Fire Suppression",
    r"\b(fire protection|sprinkler|hydrant|hose reel|deluge|preaction)\b": "Fire Protection",
    r"\b(smoke control|smoke extraction|pressurization)\b": "Smoke Control",
    # Security
    r"\b(cctv|video surveillance|ip camera|nvr|dvr)\b": "Video Surveillance",
    r"\b(access control|turnstile|rfid reader|biometric|bio metric)\b": "Access Control",
    r"\b(intrusion alarm|burglar alarm|perimeter security)\b": "Intrusion Alarm",
    r"\b(psim|vms|milestone|genetec|avigilon)\b": "Security Integration",
    # Industrial Automation
    r"\b(plc|ladder logic|iec 61131|rtu|dcs)\b": "PLC",
    r"\b(scada|wincc|ifix|ignition|factorytalk|hmi)\b": "SCADA",
    r"\b(vfd|drive|servo|motion control)\b": "Drives",
    r"\b(robot|cobot|pick and place)\b": "Robotics",
    r"\b(bms|building management system)\b": "BMS",
    # HVAC / MEP
    r"\b(hvac|chiller|air handling unit|ahu|vrf|vrv|ducting|ventilation)\b": "HVAC",
    r"\b(heat pump|boiler|burner)\b": "Heating",
    r"\b(cold room|refrigeration)\b": "Refrigeration",
    # Water & Environmental
    r"\b(wastewater|sewage|effluent|stp|etp|waste water)\b": "Wastewater Treatment",
    r"\b(ultrafiltration|reverse osmosis|ro plant|demineralization|softener)\b": "Water Treatment",
    r"\b(desalination)\b": "Desalination",
    # Testing / Labs
    r"\b(calibration|metrology|gauges|dimensional|mass calibration|flow calibration)\b": "Calibration",
    r"\b(ndt|non destructive|ultrasonic testing|radiography|magnetic particle|penetrant)\b": "NDT",
    r"\b(materials testing|tensile|hardness|metallography|microstructure)\b": "Materials Testing",
    r"\b(environmental testing|emissions|stack monitoring|air quality|water testing)\b": "Environmental Testing",
    r"\b(electrical testing|hipot|insulation resistance|relay testing)\b": "Electrical Testing",
    # Construction / Civil
    r"\b(structural|civil engineering|geotech|rebar|concrete testing)\b": "Civil & Structural",
    r"\b(lifting equipment|cranes|hoists|eot)\b": "Lifting Equipment",
    # Energy
    r"\b(solar pv|photovoltaic|inverter|string combiner)\b": "Solar PV",
    r"\b(power quality|switchgear|substation|mv|lv)\b": "Power Systems",
    # Marine
    r"\b(marine|offshore|vessel|ship)\b": "Marine",
    # Coatings / Corrosion
    r"\b(corrosion|coating|blasting|painting)\b": "Corrosion Protection",
    # Bio/Pharma
    r"\b(biotech|biotechnology|pharma|pharmaceutical)\b": "Biotechnology",
    # Cyber
    r"\b(cybersecurity|soc|siem)\b": "Cybersecurity",
}

VALID_SUFFIXES = [
    "Manufacturers","Suppliers & Distributors","Suppliers","Distributors",
    "Installers","Integrators","Service Providers","Consultants","Contractors",
    "Calibration Labs","NDT Labs","Environmental Testing Labs","Materials Testing Labs","Electrical Test Labs",
    "Specialists","Platforms","Associations","Retailers","Equipment"
]

ROLE_HINTS = {
    "manufacturer":"Manufacturers",
    "supplier":"Suppliers & Distributors",
    "distributor":"Suppliers & Distributors",
    "installer":"Installers",
    "integrator":"Integrators",
    "consultant":"Consultants",
    "contractor":"Contractors",
    "service":"Service Providers",
    "association":"Associations",
    "platform":"Platforms",
}

def sanitize_label(s):
    s = re.sub(r"[^A-Za-z0-9\-&() /]+","", str(s).strip())
    s = re.sub(r"\s+"," ", s).strip()
    s = re.sub(r"\b(\w+)\b(?:\s+\1\b)+", r"\1", s, flags=re.I)  # collapse dup words
    for g in GENERIC_BAN:
        s = re.sub(rf"\b{re.escape(g)}\b", "", s, flags=re.I).strip()
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"^\-+\s*|\s*\-+$", "", s)  # trim stray dashes
    return s

def extract_tokens(blob):
    toks=[t for t in re.sub(r"[^a-z0-9 \-\/&]", " ", blob.lower()).split() if t and t not in STOPWORDS]
    return toks

def role_ratios(subdf: pd.DataFrame):
    pool=[]
    for col in ("roles","business_model"):
        if col in subdf:
            for s in subdf[col].tolist():
                pool += extract_tokens(" ".join(to_list(s)))
    blob = ""
    if "services" in subdf:  blob += " ".join(subdf["services"].astype(str).tolist()).lower()+" "
    if "ai_summary" in subdf: blob += " ".join(subdf["ai_summary"].astype(str).tolist()).lower()+" "
    weak = []
    for k,v in {
        "install":"installer","commission":"installer",
        "integrat":"integrator",
        "wholesale":"supplier","distribut":"supplier","stockist":"supplier","resell":"supplier","retail":"supplier",
        "manufactur":"manufacturer","fabricat":"manufacturer","producer":"manufacturer","oem":"manufacturer",
        "consult":"consultant","advis":"consultant","assessment":"consultant","survey":"consultant",
        "contract":"contractor",
        "maintenance":"service","servicing":"service","calibration":"service","testing":"service","laboratory":"service"," lab ":"service",
        "membership":"association","trade association":"association","federation":"association","society":"association","institute":"association","council":"association",
        "platform":"platform","marketplace":"platform"
    }.items():
        if k in blob: weak.append(v)
    canon = pool + weak
    cnt = Counter([t for t in canon if t in ROLE_HINTS])
    total = max(1, sum(cnt.values()))
    return {k: cnt.get(k,0)/total for k in ROLE_HINTS}

def canonical_head_from_texts(texts, keywords=None):
    kw_blob = " ".join(keywords or [])
    blob = (" ".join(texts) + " " + kw_blob).lower()
    for pat, head in CANON_HEAD.items():
        if re.search(pat, blob):
            return head
    toks = extract_tokens(blob)
    grams=[]
    for n in (3,2):
        grams += [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]
    cnt = Counter(g for g in grams if all(w not in GENERIC_BAN for w in g.split()))
    if cnt:
        best = cnt.most_common(1)[0][0]
        head = " ".join([w.title() for w in best.split()[:3]])
        if head: return head
    return "Specialist"

def lab_suffix_from_roles(info, blob_lower):
    # st
    if re.search(r"\b(calibration|metrology)\b", blob_lower): return "Calibration Labs"
    if re.search(r"\b(ndt|non destructive|ultrasonic testing|radiography|magnetic particle|penetrant)\b", blob_lower): return "NDT Labs"
    if re.search(r"\b(environmental testing|emissions|stack monitoring|air quality|water testing)\b", blob_lower): return "Environmental Testing Labs"
    if re.search(r"\b(materials testing|metallography|tensile|hardness)\b", blob_lower): return "Materials Testing Labs"
    if re.search(r"\b(electrical testing|hipot|insulation resistance|relay testing)\b", blob_lower): return "Electrical Test Labs"

    rr = info.get("role_ratio",{})
    if not rr: return "Specialists"
    top = max(rr.items(), key=lambda x:x[1])[0]
    return ROLE_HINTS.get(top, "Specialists")

SYSTEM_PROMPT_BROAD = (
"You are an industrial taxonomy expert. Create short, high-level umbrella sector labels.\n"
"- 2–4 words max\n"
"- Think of sectors (e.g., 'Testing & Inspection', 'Fire & Life Safety', 'Security & Surveillance', 'Industrial Automation').\n"
"- Do NOT include roles (Manufacturers, Labs, Consultants, Installers, Integrators, Suppliers, etc.) at this level.\n"
"- Only output the label."
)

SYSTEM_PROMPT_SUBNICHE = (
"You are an industrial taxonomy expert. Create short, precise cluster labels.\n"
"- 2–5 words max\n"
"- Use industry phrasing (e.g., 'Fire Alarm Manufacturers', 'PLC Integrators', 'Water Treatment Consultants').\n"
"- Valid suffixes: Manufacturers, Suppliers & Distributors, Suppliers, Distributors, Installers, Integrators, Service Providers, Consultants, Contractors, Calibration Labs, NDT Labs, Environmental Testing Labs, Materials Testing Labs, Electrical Test Labs, Platforms, Associations, Specialists.\n"
"- Prefer WHAT they do/make; avoid vague words (General, Services, Solutions, Systems).\n"
"- Only output the label."
)


def llm_propose(info, parent=None, level_name="Broad"):
    global llm_calls
    if client is None or llm_calls>=LLM_BUDGET: return None

    system_prompt = SYSTEM_PROMPT_BROAD if "Broad" in level_name else SYSTEM_PROMPT_SUBNICHE

    try:
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":make_llm_context(info, parent, level_name)}],
            temperature=LLM_TEMPERATURE, max_tokens=LLM_MAXTOK
        )
        llm_calls += 1
        cand = r.choices[0].message.content.strip().strip('"\'')

        return sanitize_label(cand) if cand else None
    except Exception as e:
        print("LLM error:", e)
        return None


def make_llm_context(info, parent=None, level_name="Broad"):
    parts=[]
    parts.append(f"Level: {level_name}" + (f" (child of '{parent}')" if parent else ""))
    parts.append(f"Size: {info.get('size',0)}")
    if info.get("company_names"): parts.append("Examples: " + ", ".join(info["company_names"]))
    if info.get("services"): parts.append("Services: " + "; ".join(info["services"]))
    if info.get("top_keywords"): parts.append("Key terms: " + ", ".join(info["top_keywords"][:16]))
    rr = info.get("role_ratio",{})
    if rr:
        tops = sorted(rr.items(), key=lambda x: -x[1])[:6]
        parts.append("Role mix: " + ", ".join(f"{k}:{v:.0%}" for k,v in tops))
    return "\n".join(parts)

def load_label_cache_from_excel(xlsx_path):
    if not os.path.exists(xlsx_path): return {}
    try:
        cache_df = pd.read_excel(xlsx_path, sheet_name="label_cache")
        if "cache_key" in cache_df and "label" in cache_df:
            return dict(zip(cache_df["cache_key"].astype(str), cache_df["label"].astype(str)))
    except Exception:
        pass
    return {}

def export_with_cache(main_df, cache_dict):
    cache_df = pd.DataFrame([{"cache_key":k, "label":v} for k,v in cache_dict.items()])
    with pd.ExcelWriter(OUTFILE, engine="openpyxl") as w:
        main_df.to_excel(w, sheet_name="Clusters", index=False)
        if not cache_df.empty:
            cache_df.to_excel(w, sheet_name="label_cache", index=False)

label_cache = load_label_cache_from_excel(OUTFILE)

def cluster_info(level, cid):
    if level=="broad": m = (df["cluster_broad"]==cid)
    elif level=="sub": m = (df["cluster_sub"]==cid)
    else: m = (df["cluster_niche"]==cid)
    sub = df[m].copy()

    def take(col, n):
        if col not in sub: return []
        return [x for x in sub[col].astype(str).tolist() if x.strip()][:n]

    rr = role_ratios(sub)
    info = {
        "size": len(sub),
        "top_keywords": [k for k,_ in Counter([t for s in sub.get("keywords","") for t in to_list(s)]).most_common(40)],
        "industries": take("industry", 12),
        "services":   take("services", 16),
        "summaries":  take("ai_summary", 12),
        "company_names": sub["company_name"].head(8).tolist() if "company_name" in sub else [],
        "business_models": take("business_model", 12),
        "role_ratio": rr
    }
    return info, sub

def cache_key(level_name, cid, parent, info):
    sig = "|".join([
        CACHE_VERSION, level_name, str(cid), str(parent or ""),
        str(info.get("size",0)),
        ",".join(info.get("top_keywords", [])[:12]),
        ",".join(info.get("company_names", [])[:6]),
        ",".join(info.get("industries", [])[:6]),
        ",".join(info.get("services", [])[:6]),
        ",".join(sorted([f"{k}:{info['role_ratio'].get(k,0):.2f}" for k in info.get("role_ratio",{})]))
    ])
    return sig

def llm_propose(info, parent=None, level_name="Broad"):
    global llm_calls
    if client is None or llm_calls >= LLM_BUDGET:
        return None

    system_prompt = SYSTEM_PROMPT_BROAD if "Broad" in level_name else SYSTEM_PROMPT_SUBNICHE

    try:
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":make_llm_context(info, parent, level_name)}
            ],
            temperature=LLM_TEMPERATURE, max_tokens=LLM_MAXTOK
        )
        llm_calls += 1
        cand = r.choices[0].message.content.strip().strip('"\'')

        cand = sanitize_label(cand)
        return cand if cand else None
    except Exception as e:
        print("LLM error:", e)
        return None


def force_english_head_suffix(label, info, level_name="Sub"):
    """Normalize any label to 'Head + Suffix' in clean English.
       For Broad: force umbrella only (no role suffix)."""
    label = sanitize_label(label)

    texts = info.get("services", []) + info.get("summaries", []) + info.get("industries", [])
    blob = (" ".join(texts + info.get("top_keywords", []))).lower()

    # ---- Broad: umbrella only
    if "Broad" in level_name:
        head = enforce_umbrella(label, texts)
        head = title_case_keep_acronyms(head)
        head = re.sub(r"\b(Manufacturers|Suppliers & Distributors|Suppliers|Distributors|Installers|Integrators|Service Providers|Consultants|Contractors|Calibration Labs|NDT Labs|Environmental Testing Labs|Materials Testing Labs|Electrical Test Labs|Platforms|Associations|Specialists)\b", "", head, flags=re.I).strip()
        head = sanitize_label(head)
        words = head.split()
        return " ".join(words[:4]) if len(words) > 4 else head


    found_suffix = None
    for suf in VALID_SUFFIXES:
        if re.search(rf"\b{suf}\b", label, flags=re.I):
            found_suffix = suf
            break

    if found_suffix:
        head_part = re.split(rf"\b{re.escape(found_suffix)}\b", label, flags=re.I)[0].strip()
        head = head_part if head_part and head_part.lower() not in GENERIC_BAN else None
    else:
        head = None

    if head is None:
        head = canonical_head_from_texts(texts, info.get("top_keywords"))

    head_words = [w for w in re.split(r"\s+", head) if w]
    head = " ".join(head_words[:3]).strip() if head_words else "Specialist"

    suffix = found_suffix if found_suffix else lab_suffix_from_roles(info, blob)

    out = f"{head} {suffix}"
    out = title_case_keep_acronyms(out)
    out = sanitize_label(out)

    words = out.split()
    if len(words) > MAX_LABEL_WORDS:
        keep_tail = 2 if "Suppliers" in out and "Distributors" in out else 1
        head_trim = MAX_LABEL_WORDS - keep_tail
        out = " ".join(words[:max(1, head_trim)] + words[-keep_tail:])

    out = re.sub(r"\b(Specialist|Specialists)\b\s+\b(Specialist|Specialists)\b", "Specialists", out, flags=re.I)
    return out


def deterministic_fallback(info, sub_df, level_name="Broad"):
    texts = info.get("services", []) + info.get("summaries", []) + info.get("industries", [])
    head = canonical_head_from_texts(texts, info.get("top_keywords"))

    if "Broad" in level_name:

        label = enforce_umbrella(head, texts)
        return title_case_keep_acronyms(label)

    #
    blob = (" ".join(texts + info.get("top_keywords", []))).lower()
    suffix = lab_suffix_from_roles(info, blob)
    label = f"{head} {suffix}"
    label = force_english_head_suffix(label, info, level_name=level_name)
    return label


def label_one(level, cid, parent_label=None):
    info, sub_df = cluster_info(level, cid)
    level_name = "Broad" if level=="broad" else ("Sub" if level=="sub" else "Niche")
    ck = cache_key(level_name, cid, parent_label, info)

    if RESPECT_CACHE and ck in label_cache:
        return label_cache[ck]

    cand = llm_propose(info, parent_label, level_name)
    if not cand:
        cand = deterministic_fallback(info, sub_df, level_name=level_name)

    # Frce structure again
    cand = force_english_head_suffix(cand, info, level_name=level_name)
    if not cand:
        cand = deterministic_fallback(info, sub_df, level_name=level_name)

    label_cache[ck] = cand
    return cand


def sibling_dedupe(labels_map, child_ids, info_getter, level):
    inv = defaultdict(list)
    for cid in child_ids: inv[labels_map[cid]].append(cid)
    for lbl, cids in inv.items():
        if len(cids)<=1: continue

        bags={}
        for cid in cids:
            info,_ = info_getter(level, cid)
            texts = info.get("services", []) + info.get("summaries", []) + info.get("top_keywords", [])
            toks = extract_tokens(" ".join(texts))
            grams = set([" ".join(toks[i:i+n]) for n in (2,3) for i in range(len(toks)-n+1)])
            grams = {g for g in grams if all(w not in GENERIC_BAN for w in g.split()) and len(g.split())<=3}
            bags[cid]=grams
        for cid in cids:
            uniq_grams = bags[cid] - set().union(*(bags[oc] for oc in cids if oc!=cid))
            spec = ""
            if uniq_grams:
                pref = sorted(uniq_grams, key=lambda g: (-len(g.split()), g))[:1]
                spec = " ".join([w.title() for w in pref[0].split()]) if pref else ""
            if spec:
                labels_map[cid] = sanitize_label(f"{labels_map[cid]} ({spec})")
    return labels_map

# ----
label_broad = {}
for b in sorted(df["cluster_broad"].unique()):
    label_broad[b] = label_one("broad", b, None)
    #
    print(f"[BROAD] {label_broad[b]}")

label_sub = {}
kids_by_broad = defaultdict(list)
for s in sorted(df["cluster_sub"].unique()):
    pl = label_broad.get(sub_parent.get(s))
    label_sub[s] = label_one("sub", s, pl)
    if sub_parent.get(s) is not None: kids_by_broad[sub_parent[s]].append(s)
for pb, kids in kids_by_broad.items():
    label_sub = sibling_dedupe(label_sub, kids, cluster_info, "sub")

label_niche = {}
kids_by_sub = defaultdict(list)
for n in sorted([x for x in df["cluster_niche"].unique() if x>=0]):
    ps = niche_parent.get(int(n))
    pl = label_sub.get(ps)
    label_niche[n] = label_one("niche", n, pl)
    if ps is not None: kids_by_sub[ps].append(n)
for ps, kids in kids_by_sub.items():
    label_niche = sibling_dedupe(label_niche, kids, cluster_info, "niche")

df["cluster_label_broad"] = df["cluster_broad"].map(label_broad)
df["cluster_label_sub"]   = df["cluster_sub"].map(label_sub)
df["cluster_label_niche"] = df["cluster_niche"].map(label_niche)

# ------------------ Append location dims--------------
for i in range(PCA_DIMS["vloc"]):
    df[f"loc_dim{i+1}"] = VIEWS["vloc"][:, i]
df["loc_text"] = series_or_blank(df, "location")

# ------------------  dim lists for plotting ------------------
broad_dim_cols = [c for c in df.columns if c.startswith("broad_dim")]
sub_dim_cols   = [c for c in df.columns if c.startswith("sub_dim")]
niche_dim_cols = [c for c in df.columns if c.startswith("niche_dim")]

if broad_dim_cols:
    df["broad_dims"] = df[broad_dim_cols].values.tolist()
if sub_dim_cols:
    df["sub_dims"]   = df[sub_dim_cols].values.tolist()
if niche_dim_cols:
    df["niche_dims"] = df[niche_dim_cols].values.tolist()

# ------------------ EXPORT ------------------
base_cols = [
    "company_name","website_url","ai_summary","industry","services","market_focus",
    "roles","customer_types","product_categories","customers_segments","business_model",
    "certifications","keywords"
]
label_cols = ["cluster_label_broad","cluster_label_sub","cluster_label_niche"]

after_label_cols = ["loc_text"] + [c for c in df.columns if c.startswith("loc_dim")]

OUT_cols = [c for c in base_cols if c in df.columns] \
         + label_cols \
         + broad_dim_cols + sub_dim_cols + niche_dim_cols \
         + ["broad_dims"] * bool(broad_dim_cols) \
         + ["sub_dims"]   * bool(sub_dim_cols) \
         + ["niche_dims"] * bool(niche_dim_cols) \
         + after_label_cols

OUT_cols = [c for c in OUT_cols if c]
OUT = df[OUT_cols].copy()

# Label cache sheet for reproducibility
label_cache = load_label_cache_from_excel(OUTFILE) if not 'label_cache' in locals() else label_cache
export_with_cache(OUT, label_cache)

print(f"\nExported {len(OUT):,} rows → {OUTFILE}")
print(f"LLM calls used: {llm_calls} / {LLM_BUDGET}")

# ------------------ Quick census  ------------------
def census(frame, id_col, name_col, title, k=20):
    if id_col in frame and name_col in frame:
        print("\n"+"-"*60); print(title); print("-"*60)
        c = (frame.groupby([id_col, name_col]).size()
             .reset_index(name="count").sort_values("count", ascending=False).head(k))
        for _, r in c.iterrows():
            print(f"{str(r[name_col])} (n={int(r['count'])})")

census(df, "cluster_broad", "cluster_label_broad", "Top Broad Clusters")
census(df, "cluster_sub", "cluster_label_sub", "Top Sub Clusters")
census(df, "cluster_niche", "cluster_label_niche", "Top Niche Clusters")

