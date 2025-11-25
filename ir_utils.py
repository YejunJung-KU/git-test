import matplotlib.pyplot as plt
import matplotlib
import json, csv, math, re
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress HuggingFace transformers logs
from transformers import logging
logging.set_verbosity_error()

# Download NLTK resources (run once)
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# English stopword list
EN_STOP = set(stopwords.words("english"))

# Relevance scores for ESCI labels
ESCI_SCORE = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}

# Cutoffs for evaluation
KS = [20, 100, 500]


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
            qid2text[str(obj["query_id"])] = obj["query"].strip()
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

def load_qrels_graded(path):
    """
    Load graded qrels with ESCI scores.
    Returns:
        qrels: {qid: {pid: score}}
        pos_stats: per-query positive label stats
    """
    qrels = defaultdict(dict)
    label_counter = defaultdict(Counter)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            qid, pid, lab = row[0], row[1], row[2].strip()
            score = ESCI_SCORE.get(lab, 0.0)
            qrels[qid][pid] = score
            if score > 0:
                label_counter[qid][lab] += 1

    pos_stats = {
        qid: {"n_pos": sum(cnt.values()), "labels": cnt}
        for qid, cnt in label_counter.items()
    }
    return qrels, pos_stats


# ------------------------
# Text processing
# ------------------------

def nltk_tokenize(text: str):
    """Lowercase → tokenize → keep alphanumeric → remove stopwords."""
    toks = word_tokenize(text.lower())
    return [t for t in toks if t.isalnum() and t not in EN_STOP]


# ------------------------
# Ranking metrics
# ------------------------

def dcg(gains):
    """Discounted cumulative gain."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))

def ndcg_at_k_single(ranked_pids, pid2gain, k=10):
    """Compute nDCG@k for a single query."""
    gains = [pid2gain.get(pid, 0.0) for pid in ranked_pids[:k]]
    ideal_gains = sorted([g for g in pid2gain.values() if g > 0.0], reverse=True)[:k]
    idcg = dcg(ideal_gains)
    return (dcg(gains) / idcg) if idcg > 0 else 0.0

def mrr_at_k_single(ranked_pids, pid2gain, k=10):
    """Compute MRR@k for a single query."""
    for r, pid in enumerate(ranked_pids[:k], start=1):
        if pid2gain.get(pid, 0.0) > 0.0:
            return 1.0 / r
    return 0.0

def recall_at_k_single_E(ranked_pids, pid2gain, k=10):
    """Recall@k considering only E (gain == 1.0) as relevant."""
    rel_E = {pid for pid, g in pid2gain.items() if g >= 1.0}
    if not rel_E:
        return None  # denominator 0 → excluded
    hit = sum(1 for pid in ranked_pids[:k] if pid in rel_E)
    return hit / len(rel_E)

def recall_at_k_single_ESC(ranked_pids, pid2gain, k=10):
    """Recall@k considering E, S, C (gain > 0) as relevant."""
    rel_ESC = {pid for pid, g in pid2gain.items() if g > 0.0}
    if not rel_ESC:
        return None
    hit = sum(1 for pid in ranked_pids[:k] if pid in rel_ESC)
    return hit / len(rel_ESC)


# ------------------------
# Evaluation
# ------------------------

def evaluate_run(run, qrels, ks=KS, desc="Evaluating"):
    """
    Evaluate ranking run with multiple metrics.
    
    Args:
        run: {qid: [pid1, pid2, ...]}
        qrels: {qid: {pid: gain}}
        ks: cutoff list
    Returns:
        dict of aggregated metrics
    """
    out = {}
    for k in ks:
        ndcgs, mrrs = [], []
        recE_vals, recESC_vals = [], []
        n_recE_den, n_recESC_den = 0, 0

        for qid, ranked in tqdm(run.items(), desc=f"{desc} @k={k}", leave=False):
            pid2gain = qrels.get(qid, {})

            ndcgs.append(ndcg_at_k_single(ranked, pid2gain, k))
            mrrs.append(mrr_at_k_single(ranked, pid2gain, k))

            rE = recall_at_k_single_E(ranked, pid2gain, k)
            if rE is not None:
                recE_vals.append(rE)
                n_recE_den += 1

            rESC = recall_at_k_single_ESC(ranked, pid2gain, k)
            if rESC is not None:
                recESC_vals.append(rESC)
                n_recESC_den += 1

        out[f"nDCG@{k}"] = round(np.mean(ndcgs), 4) if ndcgs else 0.0
        out[f"MRR@{k}"]  = round(np.mean(mrrs), 4)  if mrrs  else 0.0
        out[f"Recall_E@{k}"]   = round(np.mean(recE_vals), 4) if recE_vals else 0.0
        out[f"Recall_ESC@{k}"] = round(np.mean(recESC_vals), 4) if recESC_vals else 0.0
        out[f"(denoms) E@{k}"]   = n_recE_den
        out[f"(denoms) ESC@{k}"] = n_recESC_den

    out["n_queries"] = len(run)
    return out

def print_metrics(title, metrics, ks=KS):
    """Pretty-print evaluation results."""
    print(f"\n=== {title} ===")
    print(f"Queries evaluated: {metrics['n_queries']}")
    for k in ks:
        line = (f"@{k} | nDCG: {metrics[f'nDCG@{k}']:.4f} "
                f"| MRR: {metrics[f'MRR@{k}']:.4f} "
                f"| Recall(E): {metrics[f'Recall_E@{k}']:.4f} "
                f"| Recall(ESC): {metrics[f'Recall_ESC@{k}']:.4f}")
        print(line)


# ------------------------
# Plotting
# ------------------------

def plot_one_metric(metrics_list, labels, metric_name, ks=None, log_x=True, ylim=(0, 1.05), title=None):
    """
    Plot a single metric across multiple results.
    
    Args:
        metrics_list: list of metrics dicts (from evaluate_run)
        labels: list of legend labels
        metric_name: prefix (e.g., 'Recall_E', 'nDCG')
        ks: list of cutoff values; inferred if None
        log_x: log scale on x-axis
        ylim: y-axis range
        title: custom title
    """
    assert len(metrics_list) == len(labels), "metrics_list and labels must match"

    # Infer ks if not provided
    if ks is None:
        ks_detected = []
        pat = re.compile(rf"^{re.escape(metric_name)}@(\d+)$")
        for k in metrics_list[0].keys():
            m = pat.match(k)
            if m:
                ks_detected.append(int(m.group(1)))
        ks = sorted(set(ks_detected))
        if not ks:
            raise ValueError(f"No keys like '{metric_name}@k' found.")

    plt.figure(figsize=(7, 5))
    for metrics, label in zip(metrics_list, labels):
        vals = [metrics.get(f"{metric_name}@{k}", None) for k in ks]
        plt.plot(ks, vals, marker='o', label=label)

    plt.xlabel("k")
    plt.ylabel(metric_name)
    if ylim is not None:
        plt.ylim(*ylim)
    if log_x:
        plt.xscale("log")
    plt.title(title or f"{metric_name} @k")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend()

    plt.xticks(ks, ks)
    plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.show()
