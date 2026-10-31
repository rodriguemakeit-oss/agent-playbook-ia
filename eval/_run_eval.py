"""Temporary eval runner: query the deployed agent for each CSV question,
save each answer under _answers/NN.txt (with confidence header)."""
import csv
import json
import os
import pathlib
import subprocess

APP = "https://workspace.example.invalid/v1/chat/completions"
HERE = pathlib.Path(__file__).parent
CSV = HERE / "question_and_answers.csv"
OUT = HERE / "_answers"
OUT.mkdir(exist_ok=True)


def get_token() -> str:
    raw = subprocess.check_output(["databricks", "auth", "token", "-p", "preprod"])
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
    import time
    token = get_token()
    only = {x.strip() for x in os.environ.get("IDS", "").split(",") if x.strip()}
    sleep_s = int(os.environ.get("SLEEP", "0"))
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    for i, row in enumerate(rows):
        qid = row["id"]
        if only and qid not in only:
            continue
        q = row["question"]
        if i and sleep_s:
            time.sleep(sleep_s)
        print(f"[{qid}] {q[:60]}…", flush=True)
        try:
            answer, conf = ask(token, q)
        except Exception as e:  # noqa: BLE001
            answer, conf = f"ERROR: {e}", None
        score = conf.get("score") if conf else ""
        unsure = conf.get("unsure") if conf else ""
        logged = conf.get("logged") if conf else ""
        reason = conf.get("reason") if conf else ""
        f = OUT / f"{int(qid):02d}.txt"
        f.write_text(
            f"QUESTION: {q}\n"
            f"CONFIDENCE: score={score} unsure={unsure} logged={logged}\n"
            f"REASON: {reason}\n"
            f"{'='*70}\n{answer}\n",
            encoding="utf-8",
        )
        print(f"    -> score={score} unsure={unsure} logged={logged} saved={f.name}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
