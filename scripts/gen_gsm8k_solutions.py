"""Generate GSM8K solutions with an OUT-OF-POOL solver (gemma2:9b) and label by gold.

This is the error SOURCE for the GSM8K task family: each cached solution is a full
multi-step reasoning trace whose final answer either matches the gold (-> clean item) or
not (-> error item). gemma2:9b is deliberately NOT in any of the three verifier pools
(same_model=qwen7b, same_family=llama32_3b+llama31, cross_family_3=qwen7b+llama31+mistral7b),
so the generator never verifies its own output — the same generator!=verifier separation
MAST has. We harvest a shuffled pool of test (then train) questions until we have enough of
BOTH labels, then the loader samples 150 error + 75 clean from the cache.

Writes data/processed/gsm8k_solutions.jsonl (one labeled solution per line). Re-runnable:
if the cache already has enough of both labels it exits without hitting the backend.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs" / "models.yaml"))
HOST = CFG["backend"]["host"].rstrip("/")
OUT = ROOT / "data" / "processed" / "gsm8k_solutions.jsonl"

SOLVER = "gemma2:9b"          # out-of-pool generator
TEMPERATURE = 0.7
NUM_PREDICT = 512
NUM_CTX = 8192
SEED0 = 70000

N_WRONG_TARGET = 170         # buffer over the 150 error items the loader needs
N_CLEAN_TARGET = 90          # buffer over the 75 clean items
QUESTION_BUDGET = 2600       # hard cap on generations (bounds wall-clock); resumable if hit

SYS = ("You are solving a grade-school math word problem. Think step by step, then give the "
       "final numeric answer on the last line in the exact form '#### <number>'.")

_GOLD = re.compile(r"####\s*([-\$0-9,\.]+)")
_NUM = re.compile(r"-?\$?[0-9][0-9,]*\.?[0-9]*")


def _norm(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").rstrip(".")
    return s or None


def gold_answer(answer_field):
    m = _GOLD.search(answer_field)
    return _norm(m.group(1)) if m else None


def extract_pred(text):
    m = list(_GOLD.finditer(text))
    if m:
        return _norm(m[-1].group(1))
    nums = _NUM.findall(text)
    return _norm(nums[-1]) if nums else None


def same_number(pred, gold):
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred == gold


def solve(question, seed):
    body = {"model": SOLVER,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": question}],
            "stream": False, "keep_alive": "30m",
            "options": {"temperature": TEMPERATURE, "seed": int(seed),
                        "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX}}
    r = requests.post(f"{HOST}/api/chat", json=body, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]


def existing_counts():
    n_wrong = n_clean = 0
    seen = set()
    if OUT.exists():
        for line in open(OUT):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            seen.add((r["split"], r["idx"]))
            if r["parsed"]:
                n_wrong += int(r["is_error"])
                n_clean += int(not r["is_error"])
    return n_wrong, n_clean, seen


def main():
    from datasets import load_dataset
    import random

    # reachability + model presence
    tags = requests.get(f"{HOST}/api/tags", timeout=10).json()
    names = {m.get("name", "") for m in tags.get("models", [])}
    if SOLVER not in names:
        print(f"solver {SOLVER!r} not present at {HOST}: {sorted(names)}", file=sys.stderr)
        return 2

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_wrong, n_clean, seen = existing_counts()
    print(f"cache has wrong={n_wrong} clean={n_clean} (targets {N_WRONG_TARGET}/{N_CLEAN_TARGET})",
          file=sys.stderr)
    if n_wrong >= N_WRONG_TARGET and n_clean >= N_CLEAN_TARGET:
        print("targets already met; nothing to generate.", file=sys.stderr)
        return 0

    # shuffled pool: test first (canonical), then train for extra volume
    pool = []
    for split in ("test", "train"):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for i in range(len(ds)):
            pool.append((split, i, ds[i]["question"], gold_answer(ds[i]["answer"])))
    random.Random(SEED0).shuffle(pool)

    n_gen = 0
    n_unparsed = 0
    with open(OUT, "a", encoding="utf-8") as f:
        for gi, (split, idx, q, gold) in enumerate(pool):
            if n_wrong >= N_WRONG_TARGET and n_clean >= N_CLEAN_TARGET:
                break
            if n_gen >= QUESTION_BUDGET:
                print("hit QUESTION_BUDGET; stopping early.", file=sys.stderr)
                break
            if (split, idx) in seen or gold is None:
                continue
            # skip labels we already have enough of, to spend budget where it is needed
            try:
                out = solve(q, seed=SEED0 + gi)
            except requests.RequestException as e:
                print(f"  backend error on {split}#{idx}: {e}", file=sys.stderr)
                continue
            pred = extract_pred(out)
            parsed = pred is not None
            is_error = parsed and not same_number(pred, gold)
            # if this label is already satisfied, still record it (cheap) but it won't help targets
            rec = {"split": split, "idx": idx, "question": q, "gold": gold,
                   "pred": pred, "parsed": parsed, "is_error": bool(is_error),
                   "solution": out.strip()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_gen += 1
            if not parsed:
                n_unparsed += 1
            elif is_error:
                n_wrong += 1
            else:
                n_clean += 1
            if n_gen % 20 == 0:
                print(f"  gen={n_gen} wrong={n_wrong} clean={n_clean} unparsed={n_unparsed}",
                      file=sys.stderr, flush=True)

    print(f"DONE: generated {n_gen} this run; totals wrong={n_wrong} clean={n_clean} "
          f"unparsed={n_unparsed}; wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
