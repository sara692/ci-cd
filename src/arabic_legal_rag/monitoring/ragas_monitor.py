import datetime as dt
import os

import mlflow

from arabic_legal_rag.config import ROOT, load_params
from arabic_legal_rag.evaluation.ragas_eval import evaluate_config, load_items


def main() -> None:
    p = load_params()
    items = load_items(ROOT / "data/eval/qa_50_ragas.jsonl")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("rag-monitoring")
    with mlflow.start_run(run_name=dt.datetime.now().strftime("%Y-%m-%d_%H%M")):
        mlflow.log_params(
            {
                "embedding_model": p["embedding_model"],
                "min_score": p["min_score"],
                "llm_model": os.environ.get("LLM_MODEL", ""),
            }
        )
        metrics, _ = evaluate_config(p, items)
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, v)


if __name__ == "__main__":
    main()
