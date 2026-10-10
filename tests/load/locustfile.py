import json
import random
from pathlib import Path

from locust import HttpUser, between, task

QUESTIONS = [
    json.loads(line)["question"]
    for line in Path("data/eval/qa_50_ragas.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]


class RagUser(HttpUser):
    wait_time = between(1, 3)

    @task(4)
    def ask(self):
        self.client.post(
            "/ask", json={"question": random.choice(QUESTIONS)}, name="/ask", timeout=120
        )

    @task(1)
    def ask_stream(self):
        with self.client.post(
            "/ask_stream",
            json={"question": random.choice(QUESTIONS)},
            name="/ask_stream",
            stream=True,
            timeout=120,
            catch_response=True,
        ) as r:
            for _ in r.iter_content(1024):
                pass
            r.success()
