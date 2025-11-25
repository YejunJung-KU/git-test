import json
from tqdm import tqdm
from pathlib import Path
from utils import * 
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset



# Default paths
ROOT = Path("dataset") # Root dataset directory
CORPUS_PATH = ROOT / "corpus.jsonl" # Product corpus file (JSON Lines): Each line contains a product ID and its associated text description.
EMB_PATH = ROOT / "corpus_bert_mean.pt"

# Task 1: Product category classification
LABEL_MAP_PATH = ROOT / "category_classification" 
LABEL2ID_PATH = LABEL_MAP_PATH / "label2labelid.json" 
ID2LABEL_PATH = LABEL_MAP_PATH / "labelid2label.json" 
PID2LABEL_TEST_PATH = LABEL_MAP_PATH / "pid2labelids_test.json" 
LABEL_EMB_PATH = LABEL_MAP_PATH / "category_labels_bert_mean.pt"

pid2text = load_corpus(CORPUS_PATH) # load corpus

label2id = load_json(LABEL2ID_PATH)
id2label = load_json(ID2LABEL_PATH)
pid2label_test = load_json(PID2LABEL_TEST_PATH)

# loading pre-trained embeddings
corpus_data = torch.load(EMB_PATH)  # {"ids": [...], "embeddings": Tensor}
pid_list = corpus_data["ids"]
pid2idx = {pid: i for i, pid in enumerate(pid_list)}
embeddings = corpus_data["embeddings"]

label_data = torch.load(LABEL_EMB_PATH)
label_emb = label_data["embeddings"]

# Unlabeled dataset: provides product embeddings without labels
class UnlabeledEmbeddingDataset(Dataset):
    def __init__(self, pids, pid2idx, embeddings):
        self.pids = list(pids)                       # list of product IDs
        self.indices = [pid2idx[pid] for pid in self.pids]  # map PIDs to embedding indices
        self.embeddings = embeddings

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        return {"pid": self.pids[idx], "X": self.embeddings[self.indices[idx]]}

# Unlabeled dataset loader: provide embeddings of unlabeled products
unlabeled_pids = pid_list
unlabeled_dataset = UnlabeledEmbeddingDataset(unlabeled_pids, pid2idx, embeddings)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=False)

# Constructing Silver Labels
import re, unicodedata, numpy as np, torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from tqdm import tqdm

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return s

@torch.no_grad()
def build_silver_labels_candidates(
    unlabeled_loader,
    pid_list, pid2idx, pid2text,
    label2id, label_emb,
    K=28,                    # Number of label candidates selected by TF-IDF
    alpha=0.6,               # Weighted combination of embedding and TF-IDF scores
    pct=0.86,                # daptive threshold using a top percentile of mixed scores across the batch
    margin_min=0.06,         # Minimum margin between top-1 and top-2 scores
    cap_per_label=40,       # Per-label maximum picks
    tfidf_kwargs=dict(min_df=2, max_df=0.95, ngram_range=(1,2), sublinear_tf=True, norm="l2"),
):
    # 1) Prepare label texts (full path + duplicate the leaf name to up-weight it)
    L = len(label2id)
    label_full = [""] * L
    label_leaf = [""] * L
    for name, lid in label2id.items():
        label_full[lid] = _norm(name)
        leaf = name.split(">")[-1].strip()
        label_leaf[lid] = _norm(leaf)

    label_texts = [f"{label_full[i]} {label_leaf[i]} {label_leaf[i]}" for i in range(L)]

    # 2) Build TF-IDF vectors (lexical view) to generate candidates
    if tfidf_kwargs is None:
        tfidf_kwargs = dict(min_df=2, max_df=0.95, ngram_range=(1,2), sublinear_tf=True, norm="l2")
    vectorizer = TfidfVectorizer(**tfidf_kwargs)
    tfidf_labels = vectorizer.fit_transform(label_texts)                # [L, V]

    prod_texts = [_norm(pid2text[pid]) for pid in pid_list]
    tfidf_products = vectorizer.transform(prod_texts)                   # [N, V]

    # 3) Compute cosine similarity in embedding space (with L2 normalization)
    lab_norm = F.normalize(label_emb, dim=1)               # [L, d]
    lab_norm_T = lab_norm.t().contiguous()

    silver = {}
    kept_per_label = defaultdict(int)

    for batch in tqdm(unlabeled_loader, desc="[silver-cand]"):
        pids = batch["pid"]
        X    = batch["X"]                    # [B, d]
        rows = [pid2idx[pid] for pid in pids]

        # 3-1) Select top-K label candidates by TF-IDF (use sparse argpartition to save memory)
        sims_tfidf = (tfidf_products[rows]).dot(tfidf_labels.T)        # [B, L] (sparse)
        topK_idx = []
        topK_tfs = []
        for i in range(sims_tfidf.shape[0]):
            row = sims_tfidf.getrow(i)
            if row.nnz == 0:
                topK_idx.append(np.array([], dtype=int))
                topK_tfs.append(np.array([], dtype=float))
                continue
            idx = row.indices
            dat = row.data
            if len(idx) > K:
                part = np.argpartition(dat, -K)[-K:]
                order = part[np.argsort(dat[part])[::-1]]
                idx = idx[order]; dat = dat[order]
            else:
                order = np.argsort(dat)[::-1]
                idx = idx[order]; dat = dat[order]
            topK_idx.append(idx)
            topK_tfs.append(dat)

        # 3-2) Compute embedding scores only for the candidates, then mix with TF-IDF scores
        Xn = F.normalize(X, dim=1)                                     # [B, d]
        top1 = []; top2 = []
        for bi, pid in enumerate(pids):
            cand = topK_idx[bi]
            if cand.size == 0:
                continue
            # emb score for candidates only
            emb_scores = (Xn[bi].unsqueeze(0) @ lab_norm[cand].T).squeeze(0).cpu().numpy()  # [k]
            mix = alpha * emb_scores + (1-alpha) * topK_tfs[bi]                             # [k]
            # top1/top2 and margin
            ord_ = np.argsort(mix)[::-1]
            l1, c1 = int(cand[ord_[0]]), float(mix[ord_[0]])
            c2 = float(mix[ord_[1]]) if len(ord_) > 1 else 0.0
            top1.append(c1); top2.append(c2)
            silver[pid] = (l1, c1, c2)

    # 4) Adaptive selection using percentile threshold + margin + per-label cap
    # Estimate the percentile threshold from the global score distribution (not per-batch)
    conf_all = np.array([v[1] for v in silver.values()], dtype=float)
    if conf_all.size == 0:
        print("[silver-cand] no candidates; relax K/alpha or vectorizer params.")
        return {}

    tau_adapt = float(np.quantile(conf_all, pct))
    final = {}
    for pid, (lab, c1, c2) in silver.items():
        if c1 >= tau_adapt and (c1 - c2) >= margin_min and kept_per_label[lab] < cap_per_label:
            final[pid] = (lab, c1)
            kept_per_label[lab] += 1

    print(f"[Silver-Candidates] kept {len(final)} / {len(pid_list)} "
          f"({len(final)/len(pid_list):.2%}), "
          f"tau(adapt p{int(pct*100)})={tau_adapt:.4f}, margin>={margin_min}, cap={cap_per_label}")
    return silver


silver = build_silver_labels_candidates(
    unlabeled_loader, pid_list, pid2idx, pid2text, label2id, label_emb
)


# Build Silver Labeled Dataset
from torch.utils.data import random_split

class SilverLabeledDataset(Dataset):
    def __init__(self, pid_list, pid2idx, embeddings, silver):
        # silver: {pid: (label_id, confidence)}
        self.items = [(pid, silver[pid][0]) for pid in pid_list if pid in silver]
        self.indices = [pid2idx[pid] for pid,_ in self.items]
        self.embeddings = embeddings
        self.y = torch.tensor([lab for _, lab in self.items], dtype=torch.long)

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, i):
        x = self.embeddings[self.indices[i]]
        y = self.y[i]
        return {"X": x, "y": y}

silver_train_dataset = SilverLabeledDataset(pid_list, pid2idx, embeddings, silver)

# Split into train/validation sets (80% / 20%)
val_ratio = 0.2
val_size = int(len(silver_train_dataset) * val_ratio)
train_size = len(silver_train_dataset) - val_size

train_split, val_split = random_split(silver_train_dataset, [train_size, val_size])

# DataLoaders for training and validation
train_loader = DataLoader(train_split, batch_size=32, shuffle=True)
val_loader = DataLoader(val_split, batch_size=64)

class ProductCategoryEmbeddingDataset(Dataset):
    def __init__(self, pid2label, pid2idx, embeddings):
        self.pids = list(pid2label.keys())
        self.labels = [pid2label[pid] for pid in self.pids]
        self.indices = [pid2idx[pid] for pid in self.pids]
        self.embeddings = embeddings

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        emb = self.embeddings[self.indices[idx]]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"X": emb, "y": label}

# Build test dataset and dataloader from precomputed embeddings
test_dataset = ProductCategoryEmbeddingDataset(pid2label_test, pid2idx, embeddings)
test_loader = DataLoader(test_dataset, batch_size=64)

# Model dimensions
input_dim = embeddings.shape[1]   # Size of embedding vector (feature dimension)
num_classes = len(label2id)       # Number of category classes

# Pseudo-labeled dataset: stores embeddings with assigned pseudo-labels for training
class TensorDatasetFromVectors(Dataset):
    def __init__(self, X_list, y_list):
        self.X = torch.stack(X_list)                      # list of embeddings -> tensor
        self.y = torch.tensor(y_list, dtype=torch.long)   # pseudo-labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"X": self.X[idx], "y": self.y[idx]}       # embedding + pseudo-label

# Classifier that uses label embeddings to make predictions
class InnerProductClassifier(nn.Module):
    def __init__(self, input_dim, label_embeddings, trainable_label_emb=True):
        super().__init__()
        # Project input features into the same dimension as label embeddings
        self.proj = nn.Linear(input_dim, label_embeddings.size(1))

        if trainable_label_emb:
            # Label embeddings are trainable parameters
            self.label_emb = nn.Parameter(label_embeddings.clone())
        else:
            # Label embeddings are fixed (not updated during training)
            self.register_buffer("label_emb", label_embeddings.clone())

    def forward(self, x):
        # Project input feature vectors
        x_proj = self.proj(x)
        # Compute logits as similarity with each label embedding
        logits = torch.matmul(x_proj, self.label_emb.T)
        return logits

model = InnerProductClassifier(input_dim, label_emb)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            X = batch["X"]
            y = batch["y"]
        
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {"accuracy": acc, "f1_macro": f1_macro}

# Training loop with self-training
patience = 5
pseudo_every = 2
threshold = 0.95

best_val_acc = -1
best_model_state = None
patience_counter = 0

val_acc_list  = []
test_acc_list = []

EPOCHS = 200

for epoch in range(1, EPOCHS + 1):
    
    # Base training loop
    model.train()
    total_loss = 0.0
    total_cnt = 0
    for batch in train_loader:
        X = batch["X"]
        y = batch["y"]

        logits = model(X)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_cnt += X.size(0)

    avg_loss = total_loss / max(1, total_cnt)
    print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

    # Validation & Test evaluation
    val_result = evaluate(model, val_loader)
    test_result = evaluate(model, test_loader)
    val_acc = val_result["accuracy"]
    test_acc = test_result["accuracy"]
    val_acc_list.append(val_acc)
    test_acc_list.append(test_acc)
    print_eval_result(val_result, stage="val", is_improved=(val_acc > best_val_acc))
    print_eval_result(test_result, stage="test")

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"[Early Stopping] No improvement for {patience} epochs.")
            break

    # Pseudo-label generation (every x epochs)
    if (epoch % pseudo_every) == 0:
        model.eval()
        picked_X, picked_y = [], []
        with torch.no_grad():
            for batch in unlabeled_loader:
                X_ulb = batch["X"]
                
                probs = torch.softmax(model(X_ulb), dim=-1)
                conf, pred = probs.max(dim=-1)
                keep = conf >= threshold
                if keep.any():
                    kept = keep.nonzero(as_tuple=True)[0]
                    picked_X.extend(X_ulb[kept].detach().cpu())
                    picked_y.extend(pred[kept].detach().cpu().tolist())

        if len(picked_X) > 0:
            pseudo_ds = TensorDatasetFromVectors(picked_X, picked_y)
            new_train_ds = ConcatDataset([train_loader.dataset, pseudo_ds])
            train_loader = DataLoader(
                new_train_ds,
                batch_size=train_loader.batch_size,
                shuffle=True
            )
            print(f"  + Added {len(pseudo_ds)} pseudo-labeled samples (thr={threshold})")
            
# === Final Evaluation ===
print("\n[Final Evaluation on Best Model]")
model.load_state_dict(best_model_state)
final_test_result = evaluate(model, test_loader)
print_eval_result(final_test_result, stage="final test", is_improved=True)


import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# === 1. Load test IDs ===
ROOT = Path("dataset") # Root dataset directory
LABEL_MAP_PATH = ROOT / "category_classification"
TEST_IDS_PATH = LABEL_MAP_PATH / "task1_test_ids.csv"

test_ids_df = pd.read_csv(TEST_IDS_PATH)  # has column "id"
test_ids = test_ids_df["id"].tolist()

# === 2. Custom Dataset (no labels) ===
class ProductCategoryTestDataset(Dataset):
    def __init__(self, pids, pid2idx, embeddings):
        self.pids = pids
        self.indices = [pid2idx[pid] for pid in self.pids]
        self.vecs = embeddings 
        
    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        emb = self.vecs[self.indices[idx]]
        return {"X": torch.tensor(emb, dtype=torch.float)}

# === 3. Build dataset and loader ===
test_dataset_kaggle = ProductCategoryTestDataset(test_ids, pid2idx, embeddings)
test_loader_kaggle = DataLoader(test_dataset_kaggle, batch_size=64)

# === 4. Run predictions ===
model.eval()
all_preds = []

with torch.no_grad():
    for batch in test_loader_kaggle:
        X = batch["X"]
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())

# === 5. Build submission file ===
submission = pd.DataFrame({
    "id": test_ids,
    "label": all_preds
})

SUBMISSION_PATH = ROOT / "submission/P3_submission.csv"
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission file saved to {SUBMISSION_PATH}")
print(submission.head())