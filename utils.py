import matplotlib.pyplot as plt
import json
import warnings
import numpy as np
from collections import defaultdict
import itertools
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress HuggingFace transformers logs
from transformers import logging
logging.set_verbosity_error()


# ------------------------
# Visualization
# ------------------------

def plot_results(results_dict, split="valid", metric='Accuracy'):
    """
    Plot metric values over epochs for multiple models.
    
    Args:
        results_dict: dict of dicts. 
            Example: results_dict["valid"]["mlp_partial"] = [0.8, 0.82, ...]
        split: "valid" or "test"
        metric: name of the metric to display
    """
    assert split in results_dict, f"{split} not in results_dict"

    plt.figure(figsize=(8, 5))
    for label, acc_list in results_dict[split].items():
        plt.plot(acc_list, label=label)

    plt.title(f"{split.capitalize()} {metric} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------------
# Printing evaluation results
# ------------------------

def print_eval_result(metrics: dict, stage="val", is_improved=False):
    """
    Print evaluation results (accuracy, F1-macro).
    
    Args:
        metrics: dict with keys 'accuracy' and 'f1_macro'
        stage: string label (e.g., "val", "test")
        is_improved: mark with '*' if results improved
    """
    star = " *" if is_improved else ""
    print(f"[{stage.upper():4}] Acc: {metrics['accuracy']:.4f} | "
          f"F1-macro: {metrics['f1_macro']:.4f}{star}")


def print_eval_result_esci(metrics: dict, stage="val", is_improved=False):
    """
    Print evaluation results including per-class accuracy for ESCI labels.
    
    Args:
        metrics: dict with 'accuracy', 'f1_macro', and optionally 'per_class_accuracy'
        stage: string label (e.g., "val", "test")
        is_improved: mark with '*' if results improved
    """
    star = " *" if is_improved else ""
    print(f"[{stage.upper():4}] Acc: {metrics['accuracy']:.4f} | "
          f"F1-macro: {metrics['f1_macro']:.4f}{star}")

    # Print per-class accuracy if available
    if "per_class_accuracy" in metrics:
        id2label = {0: "E", 1: "S", 2: "C", 3: "I"}
        per_class_acc_str = [
            f"{id2label[cls_id]}: {acc:.4f}" 
            for cls_id, acc in metrics["per_class_accuracy"].items()
        ]
        print("        " + " | ".join(per_class_acc_str))


# ------------------------
# Data loading utilities
# ------------------------

def load_json(path):
    """Load JSON file into Python object."""
    with open(path) as f:
        return json.load(f)

def load_queries(path):
    """Load queries from JSONL file -> {qid: query_text}."""
    qid2text = {}
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            qid2text[int(obj["query_id"])] = obj["query"].strip()
    return qid2text

def load_corpus(path):
    """Load corpus -> {pid: title + text}."""
    pid2text = {}
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            full_text = (obj["title"] + " " + obj["text"]).strip()
            pid2text[obj["_id"]] = full_text
    return pid2text

def dict2list(id2text):
    """
    Convert a {id: text} dict into two aligned lists.
    
    Returns:
        ids: list of keys
        texts: list of values
    """
    ids = list(id2text.keys())
    texts = [id2text[i] for i in ids]
    return ids, texts

def build_leaf_adj(id2label, label2id, decay=0.01, max_edges=10):
    """
    Build an adjacency matrix between leaf nodes based on hierarchical structure.

    Args:
        id2label: {id(str): full_path_str}, mapping from label ID to its full path
        label2id: {full_path_str: id(str)}, mapping from path string to ID
        decay: decay factor for ancestor depth (shallower common ancestors → lower weight)
        max_edges: maximum number of edges per node (keep top-k by weight)

    Returns:
        A_hat (torch.FloatTensor): normalized adjacency matrix (D^{-1/2} A D^{-1/2})
    """

    # --- 1) Store ancestor paths for each leaf ---
    # e.g., "A > B > C" → ["A", "A > B"]
    leaf2ancestors = {}
    for leaf_id, full_path in id2label.items():
        parts = full_path.split(" > ")
        ancestors = [" > ".join(parts[:d]) for d in range(1, len(parts))]
        leaf2ancestors[leaf_id] = ancestors

    # --- 2) Initialize adjacency matrix (with self-loops) ---
    n_labels = len(label2id)
    A = np.eye(n_labels, dtype=np.float32)

    all_ids = list(id2label.keys())

    # --- 3) Find common ancestors for each leaf pair and assign weights ---
    edges_by_node = defaultdict(list)  # {u: [(v, weight), ...]}
    for u, v in itertools.combinations(all_ids, 2):
        anc_u, anc_v = set(leaf2ancestors[u]), set(leaf2ancestors[v])
        common = anc_u.intersection(anc_v)
        if not common:
            continue

        # Weight based on the deepest common ancestor
        max_depth = max(len(c.split(" > ")) for c in common)
        depth_u = len(id2label[u].split(" > "))
        weight = decay ** (depth_u - max_depth)

        iu, iv = int(u), int(v)
        edges_by_node[iu].append((iv, weight))
        edges_by_node[iv].append((iu, weight))

    # --- 4) Keep only top-k edges per node ---
    for u, neigh_list in edges_by_node.items():
        top_neighbors = sorted(neigh_list, key=lambda x: x[1], reverse=True)[:max_edges]
        for v, w in top_neighbors:
            A[u, v] = max(A[u, v], w)
            A[v, u] = max(A[v, u], w)

    # --- 5) Normalize adjacency (GCN-style) ---
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    return torch.from_numpy(A_hat).float()
