"""Pick the best run, register its config in the MLflow Model Registry, set alias 'production'."""

import json

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from arabic_legal_rag.config import ROOT, load_params


class RagConfig(mlflow.pyfunc.PythonModel):
    """The registered 'model' is the pipeline configuration, not weights."""

    def predict(self, model_input, params=None):
        return self.config if hasattr(self, "config") else {}


def main():
    exp = mlflow.get_experiment_by_name(load_params()["mlflow_experiment"])
    runs = mlflow.search_runs([exp.experiment_id])
    runs = runs[
        runs["metrics.answer_rate"] >= 0.8
    ]  # a config that refuses everything is not "faithful"
    best = runs.sort_values(
        ["metrics.faithfulness", "metrics.gold_cited_rate"], ascending=False
    ).iloc[0]
    run_id = best["run_id"]
    print("best:", best["tags.mlflow.runName"], best["metrics.faithfulness"])

    params = json.loads(mlflow.artifacts.load_text(f"runs:/{run_id}/params.json"))
    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            name="config",
            python_model=RagConfig(),
            registered_model_name="ArabicLegalRAG-config",
        )
    client = MlflowClient()
    version = (
        client.get_latest_versions("ArabicLegalRAG-config")[0].version
        if hasattr(client, "get_latest_versions")
        else info.registered_model_version
    )
    client.set_registered_model_alias("ArabicLegalRAG-config", "production", version)

    keys = ["embedding_model", "query_prefix", "passage_prefix", "chunk_text"]
    chosen = {k: params[k] for k in keys}
    (ROOT / "reports" / "best_config.yaml").write_text(yaml.safe_dump(chosen, allow_unicode=True))
    print(chosen, "-> copy these into params.yaml, then: uv run dvc repro")


if __name__ == "__main__":
    main()
