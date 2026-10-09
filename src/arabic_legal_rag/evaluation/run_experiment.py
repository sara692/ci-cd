"""Grid of configs: one MLflow run each."""

import copy
import json
import os
import subprocess
import sys
from collections import defaultdict

import mlflow
import pandas as pd

from arabic_legal_rag.config import ROOT, load_params
from arabic_legal_rag.evaluation.ragas_eval import evaluate_config, load_items
from arabic_legal_rag.ingestion.index import build_index
from arabic_legal_rag.retrieval.retriever import Retriever

NEMO = dict(embedding_model="nvidia/nemotron-3-embed-1b:free", query_prefix="", passage_prefix="")
BGE_API = dict(embedding_model="baai/bge-m3", query_prefix="", passage_prefix="")

GRID = {
    "nemo_aren": {
        "embedding_model": "nvidia/nemotron-3-embed-1b:free",
        "query_prefix": "",
        "passage_prefix": "",
        "chunk_text": "ar_en",
        "index_dir": "data/index_nemotron",
        "min_score": 0.30,  # provisional: calibrate with the min_score table script
    },
    "bge_aren": {
        "embedding_model": "baai/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
        "chunk_text": "ar_en",
        "index_dir": "data/index_bge",
        # min_score inherited from params.yaml: set it for bge-m3 first
    },
}


def probe_metrics(params):
    """hit@1, hit@5, MRR per language + bilingual agreement, from retrieval_probe.jsonl."""
    path = ROOT / "data" / "eval" / "retrieval_probe.jsonl"
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    retriever = Retriever(params)  # same call as in ragas_eval.py; adapt if yours differs

    rank = defaultdict(list)  # lang -> rank of the gold article (0 = missed)
    top1 = defaultdict(dict)  # gold -> {lang: top-1 article}
    for r in rows:
        hits = retriever.search(r["question"], 10)
        nums = [h["article_number"] for h in hits]
        rank[r["lang"]].append(nums.index(r["gold"]) + 1 if r["gold"] in nums else 0)
        top1[r["gold"]][r["lang"]] = nums[0] if nums else None

    out = {"probe_n": len(rows)}
    for lang, ranks in rank.items():
        n = len(ranks)
        out[f"hit1_{lang}"] = sum(1 for x in ranks if x == 1) / n
        out[f"hit5_{lang}"] = sum(1 for x in ranks if 1 <= x <= 5) / n
        out[f"mrr_{lang}"] = sum(1 / x for x in ranks if x) / n
    pairs = [v for v in top1.values() if "ar" in v and "en" in v]
    if pairs:
        out["agreement"] = sum(1 for v in pairs if v["ar"] == v["en"]) / len(pairs)
    return out


def main(only=None):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(load_params()["mlflow_experiment"])
    items = load_items(ROOT / "data/eval/qa_20_ci.jsonl")
    for name, override in GRID.items():
        if only and name not in only:
            continue
        params = {**copy.deepcopy(load_params()), **override}
        index_dir = ROOT / params["index_dir"]
        manifest = ROOT / "data" / "index_runs" / f"{name}_manifest.json"
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    "chunk_size": "article",
                    "overlap": 0,
                    "embedding_model": params["embedding_model"],
                    "chunk_text": params["chunk_text"],
                    "index_dir": params["index_dir"],
                    "min_score": params["min_score"],
                    "top_k": params["top_k"],
                    "llm_model": os.environ.get("LLM_MODEL", ""),
                    "judge_model": os.environ.get("JUDGE_MODEL", ""),
                }
            )
            mlflow.set_tag("git_commit", subprocess.getoutput("git rev-parse --short HEAD"))
            if os.environ.get("REBUILD") == "1" or not index_dir.exists():
                build_index(params, manifest_path=manifest)
                mlflow.log_artifact(str(manifest))
            else:
                print(f"[{name}] reusing index at {index_dir}")
            m = probe_metrics(params)
            r, rows = evaluate_config(params, items)
            for k, v in {**m, **r}.items():
                if v is not None:
                    mlflow.log_metric(k, v)
            df = pd.DataFrame(rows).drop(columns=["contexts"])
            df.to_csv(ROOT / f"data/index_runs/{name}_results.csv", index=False)
            mlflow.log_artifact(str(ROOT / f"data/index_runs/{name}_results.csv"))
            mlflow.log_dict(params, "params.json")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
