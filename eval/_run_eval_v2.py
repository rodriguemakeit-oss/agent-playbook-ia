"""Eval runner v2: query the deployed agent for each CSV question, writing the
result directly back into the same file's score_modele/reponse_modele columns
(rather than separate eval/_answers/NN.txt files), so expected and actual sit
side by side."""
import csv
import json
import os
import pathlib
import subprocess
import sys

APP = "https://workspace.example.invalid/v1/chat/completions"
HERE = pathlib.Path(__file__).parent
CSV_PATH = HERE / "question_and_answers_2.csv"
FIELDNAMES = [
    "id", "categorie", "question", "reponse_attendue", "confiance_attendue",
    "score_confiance", "resultat", "score_modele", "reponse_modele",
]

PROFILE = os.environ.get("DATABRICKS_PROFILE", "preprod")


def get_token() -> str:
    raw = subprocess.check_output(["databricks", "auth", "token", "-p", PROFILE])
    return json.loads(raw)["access_token"]


def ask(token: str, question: str):
    import requests
    payload = {"model": "databricks-claude-sonnet-4-6", "stream": True,
               "messages": [{"role": "user", "content": question}]}
    ans, conf = [], None
    with requests.post(APP, json=payload, headers={"Authorization": f"Bearer {token}"},
                       stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8")
            if not s.startswith("data:"):
                continue
            d = s[5:].strip()
            if d == "[DONE]":
                break
            try:
                obj = json.loads(d)
            except json.JSONDecodeError:
                continue
            delta = obj["choices"][0]["delta"]
            ev = delta.get("agent_event")
            if ev and ev.get("type") == "confidence":
                conf = ev
            elif "content" in delta and not ev:
                ans.append(delta["content"])
    return "".join(ans), conf


def main():
    only = {x.strip() for x in os.environ.get("IDS", "").split(",") if x.strip()}
    sleep_s = int(os.environ.get("SLEEP", "0"))
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    token = get_token()

    for i, row in enumerate(rows):
        qid = row["id"]
        if only and qid not in only:
            continue
        q = row["question"]
        if i and sleep_s:
            import time
            time.sleep(sleep_s)
        print(f"[{qid}] {q[:60]}…", flush=True)
        try:
            answer, conf = ask(token, q)
        except Exception as e:  # noqa: BLE001
            answer, conf = f"ERROR: {e}", None
        row["score_modele"] = conf.get("score") if conf else ""
        row["reponse_modele"] = answer
        print(f"    -> score_modele={row['score_modele']}", flush=True)

    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written back to {CSV_PATH}")


if __name__ == "__main__":
    sys.exit(main())
