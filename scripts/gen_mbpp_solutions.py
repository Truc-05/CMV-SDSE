"""Generate MBPP solutions with an OUT-OF-POOL solver (gemma2:9b) and label by UNIT-TEST execution.

This is the error SOURCE for the MBPP code family: each cached solution is a full Python function
whose correctness is decided OBJECTIVELY by running the problem's `test_list` asserts. gemma2:9b is
deliberately NOT in any of the three verifier pools (same_model=qwen7b, same_family=llama32_3b+
llama31, cross_family_3=qwen7b+llama31+mistral7b), so the generator never verifies its own output —
the same generator!=verifier separation MAST/GSM8K have. The solver DOES see the tests (standard
MBPP protocol, so it uses the required function signature); the verifier later sees only the task
text + code, never the tests. We harvest a shuffled pool of problems until we have enough of BOTH
labels, then the loader samples 150 error + 75 clean from the cache.

Labelling runs model-generated code. It is executed in a separate, isolated (`python -I`)
subprocess with a hard wall-clock timeout and no stdin, in the scratch/cwd — MBPP tasks are pure
algorithmic stdlib code. Writes data/processed/mbpp_solutions.jsonl (one labeled solution per
line). Re-runnable: if the cache already has enough of both labels it exits without hitting the
backend.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs" / "models.yaml"))
HOST = CFG["backend"]["host"].rstrip("/")
OUT = ROOT / "data" / "processed" / "mbpp_solutions.jsonl"

SOLVER = "gemma2:9b"          # out-of-pool generator
TEMPERATURE = 0.7
NUM_PREDICT = 640
NUM_CTX = 8192
SEED0 = 80000

N_WRONG_TARGET = 170         # buffer over the 150 error items the loader needs
N_CLEAN_TARGET = 90          # buffer over the 75 clean items
QUESTION_BUDGET = 1600       # hard cap on generations (bounds wall-clock); resumable if hit
EXEC_TIMEOUT_S = 10          # per-solution unit-test wall-clock

SYS = ("You are an expert Python programmer. Write a single, self-contained Python function that "
       "solves the task and passes the given tests. Respond with ONE ```python code block "
       "containing only the function (plus any imports it needs) and nothing else.")

_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str):
    """Pull the first fenced code block; fall back to the raw text if the model forgot fences."""
    m = _CODE_BLOCK.search(text)
    code = (m.group(1) if m else text).strip()
    return code or None


def solve(text, test_list, seed):
    user = (f"{text.strip()}\n\nYour function must pass these tests:\n"
            + "\n".join(test_list))
    body = {"model": SOLVER,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": user}],
            "stream": False, "keep_alive": "30m",
            "options": {"temperature": TEMPERATURE, "seed": int(seed),
                        "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX}}
    r = requests.post(f"{HOST}/api/chat", json=body, timeout=240)
    r.raise_for_status()
    return r.json()["message"]["content"]


def run_tests(code, setup_code, test_list):
    """Execute `code` + asserts in an isolated subprocess. Returns (passed_all, n_tests, n_passed).

    n_passed counts asserts that hold when run cumulatively; passed_all is True only if the whole
    script (setup + code + every assert) exits 0. A syntax error, exception, failed assert, or
    timeout all yield passed_all=False (a genuine buggy solution == an error item.
    """
    n_tests = len(test_list)
    # Count how many leading asserts pass by probing prefixes — cheap (n<=~5) and gives n_passed
    # for diagnostics; the label only needs the full-suite verdict.
    n_passed = 0
    passed_all = False
    for j in range(n_tests + 1):
        script = (setup_code or "") + "\n" + code + "\n" + "\n".join(test_list[:j]) + "\n"
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as tf:
                tf.write(script)
                tf.flush()
                res = subprocess.run([sys.executable, "-I", tf.name],
                                     stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     timeout=EXEC_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            break
        except Exception:
            break
        if res.returncode != 0:
            break
        n_passed = j            # first j asserts (plus code) all ran clean
        if j == n_tests:
            passed_all = True
    return passed_all, n_tests, n_passed


def existing_counts():
    n_wrong = n_clean = 0
    seen = set()
    if OUT.exists():
        for line in open(OUT):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            seen.add((r["split"], r["task_id"]))
            if r["parsed"]:
                n_wrong += int(r["is_error"])
                n_clean += int(not r["is_error"])
    return n_wrong, n_clean, seen


def main():
    from datasets import load_dataset
    import random

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

    # Full MBPP config. Skip the 10-problem few-shot `prompt` split; harvest the rest.
    pool = []
    for split in ("test", "train", "validation"):
        ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
        for row in ds:
            pool.append((split, int(row["task_id"]), row["text"], row["code"],
                         list(row["test_list"]), row.get("test_setup_code", "")))
    random.Random(SEED0).shuffle(pool)

    n_gen = 0
    n_unparsed = 0
    with open(OUT, "a", encoding="utf-8") as f:
        for gi, (split, task_id, text, ref_code, test_list, setup) in enumerate(pool):
            if n_wrong >= N_WRONG_TARGET and n_clean >= N_CLEAN_TARGET:
                break
            if n_gen >= QUESTION_BUDGET:
                print("hit QUESTION_BUDGET; stopping early.", file=sys.stderr)
                break
            if (split, task_id) in seen or not test_list:
                continue
            try:
                out = solve(text, test_list, seed=SEED0 + gi)
            except requests.RequestException as e:
                print(f"  backend error on {split}#{task_id}: {e}", file=sys.stderr)
                continue
            code = extract_code(out)
            parsed = code is not None
            passed_all = n_tests = n_passed = None
            is_error = False
            if parsed:
                passed_all, n_tests, n_passed = run_tests(code, setup, test_list)
                is_error = not passed_all
            rec = {"split": split, "task_id": task_id, "text": text, "code": code,
                   "parsed": parsed, "is_error": bool(is_error),
                   "n_tests": n_tests, "n_passed": n_passed}
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
