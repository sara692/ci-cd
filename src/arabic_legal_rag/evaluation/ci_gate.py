"""CI gate: fail the build if answer quality drops."""

import json
import os
import sys

from arabic_legal_rag.config import ROOT, load_params
from arabic_legal_rag.evaluation.ragas_eval import evaluate_config, load_items

MIN_FAITHFULNESS = float(os.environ.get("FAITHFULNESS_MIN", "0.75"))
MIN_ANSWER_RATE = float(os.environ.get("ANSWER_RATE_MIN", "0.80"))


def main() -> int:
    params = load_params()
    items = load_items(ROOT / "data/eval/qa_20_ci.jsonl")
    metrics, _ = evaluate_config(params, items)

    out = ROOT / "reports" / "ci_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    faith = metrics.get("faithfulness")
    if faith is None:
        print("GATE ERROR: no faithfulness score (judge failed or nothing answered).")
        return 2  # infrastructure problem, not a quality verdict
    failures = []
    if faith < MIN_FAITHFULNESS:
        failures.append(f"faithfulness {faith:.3f} < {MIN_FAITHFULNESS}")
    if metrics["answer_rate"] < MIN_ANSWER_RATE:
        failures.append(f"answer_rate {metrics['answer_rate']:.2f} < {MIN_ANSWER_RATE}")
    if failures:
        print("GATE FAILED: " + "; ".join(failures))
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
