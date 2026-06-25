"""
Synthetic (Query, Context) data generation for Nokast-secureRAG.

Calls a vLLM OpenAI-compatible endpoint (the teacher, Qwen3.6-35B-A3B-NVFP4) with
schema-constrained decoding (`guided_json`) so every response is a valid record.
Generates benign / borderline / attack examples plus judgment-flip pairs, and
captures GB10 hardware + throughput telemetry for the whole run.

Resilience: records are appended to a raw log AS THEY COMPLETE, generation is
count-driven and resumable (`--resume`), and a watchdog stops cleanly if the
vLLM engine wedges (it exits code 2 so a supervisor can restart + resume).

Examples:
    python -m src.data_gen --smoke
    bash scripts/run_datagen.sh --n-benign 3000 --n-adversarial 2000 --n-flips 300
"""
import os
import re
import sys
import json
import time
import random
import hashlib
import argparse
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple

import requests

from src import prompts
from src.schema import Record, GENERATION_JSON_SCHEMA, ALLOWED_LABELS, validate
from src.telemetry import HardwareMonitor

DEFAULT_ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("TEACHER_MODEL", "nvidia/Qwen3.6-35B-A3B-NVFP4")

# Adversarial mix (sums to 1.0); benign is generated separately.
ADV_MIX = {
    "ipi": 0.35, "poisoning": 0.25, "context-hijack": 0.20,
    "pii-exfil": 0.10, "borderline": 0.10,
}

_write_lock = threading.Lock()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse model output; tolerate stray prose or code fences."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


class TeacherClient:
    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, model: str = DEFAULT_MODEL,
                 temperature: float = 0.9, max_tokens: int = 700, timeout: float = 45.0):
        self.url = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.tokens_used = 0

    def chat(self, messages: List[Dict[str, str]], json_schema: Dict[str, Any],
             max_tokens: Optional[int] = None) -> Optional[Dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "guided_json": json_schema,                       # near-zero parse loss
            "chat_template_kwargs": {"enable_thinking": False},  # Qwen3.6 reasoning off
        }
        try:
            r = requests.post(self.url, json=payload, timeout=self.timeout)
            if r.status_code != 200:
                return None
            data = r.json()
            usage = data.get("usage") or {}
            self.tokens_used += int(usage.get("total_tokens", 0))
            content = data["choices"][0]["message"].get("content")
            return _extract_json(content)
        except Exception:
            return None


def _coerce_label(attack_type: str, label: str) -> str:
    allowed = ALLOWED_LABELS.get(attack_type, ("safe",))
    return label if label in allowed else allowed[0]


def gen_one(client: TeacherClient, attack_type: str) -> Optional[Record]:
    nonce = hashlib.md5(os.urandom(8)).hexdigest()[:8]
    domain = random.choice(prompts.DOMAINS)
    obj = client.chat(prompts.generation_prompt(attack_type, domain=domain, nonce=nonce),
                      GENERATION_JSON_SCHEMA)
    if not obj:
        return None
    rec = Record(
        query=str(obj.get("query", "")).strip(),
        context=str(obj.get("context", "")).strip(),
        label=_coerce_label(attack_type, obj.get("label", "")),
        attack_type=attack_type,
        reasoning=str(obj.get("reasoning", "")).strip(),
        source="teacher",
    )
    return rec if not validate(rec.to_dict()) else None


def gen_flip_pair(client: TeacherClient, sentence: str, pair_id: int) -> List[Record]:
    nonce = hashlib.md5(os.urandom(8)).hexdigest()[:8]
    # Flip pairs emit 7 text fields (two full contexts) -> need a larger budget.
    obj = client.chat(prompts.flip_prompt(sentence, nonce=nonce), prompts.FLIP_JSON_SCHEMA,
                      max_tokens=1300)
    if not obj:
        return []
    try:
        benign = Record(
            query=obj["benign_query"].strip(), context=obj["benign_context"].strip(),
            label="safe", attack_type="benign", reasoning=obj["benign_reasoning"].strip(),
            flip_pair_id=pair_id, split="test", source="teacher",
        )
        malic = Record(
            query=obj["malicious_query"].strip(), context=obj["malicious_context"].strip(),
            label="malicious-instruction", attack_type="ipi",
            reasoning=obj["malicious_reasoning"].strip(),
            flip_pair_id=pair_id, split="test", source="teacher",
        )
    except (KeyError, AttributeError):
        return []
    out = [r for r in (benign, malic) if not validate(r.to_dict())]
    return out if len(out) == 2 else []  # only keep complete pairs


# ---------------- resumable, checkpointing orchestration ----------------

def _append(raw_path: str, rec: Record):
    with _write_lock:
        with open(raw_path, "a") as f:
            f.write(json.dumps(rec.to_dict()) + "\n")


def _targets(n_benign: int, n_adversarial: int) -> Dict[str, int]:
    t = {"benign": n_benign}
    for k, frac in ADV_MIX.items():
        t[k] = round(n_adversarial * frac)
    return t


def load_raw(raw_path: str) -> Tuple[Counter, set]:
    counts, flip_pairs = Counter(), set()
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                counts[r.get("attack_type", "?")] += 1
                if r.get("flip_pair_id") is not None:
                    flip_pairs.add(r["flip_pair_id"])
    return counts, flip_pairs


def generate(client: TeacherClient, n_benign: int, n_adversarial: int, n_flips: int,
             workers: int, seed: int, mon: HardwareMonitor, raw_path: str,
             max_consec_fail: int = 25) -> bool:
    """Generate the deficit vs targets, appending live. Returns True if a wedge
    watchdog tripped (i.e. caller should restart the engine and resume)."""
    targets = _targets(n_benign, n_adversarial)
    counts, flip_pairs = load_raw(raw_path)
    deficits = {t: max(0, targets[t] - counts.get(t, 0)) for t in targets}
    flip_need = max(0, n_flips - len(flip_pairs))

    jobs: List[Tuple[str, Any]] = []
    for t, n in deficits.items():
        jobs += [("single", t)] * n
    next_pid = (max(flip_pairs) + 1) if flip_pairs else 0
    jobs += [("flip", next_pid + i) for i in range(flip_need)]

    print(f"targets={targets}")
    print(f"existing={dict(counts)} flip_pairs_done={len(flip_pairs)}")
    print(f"deficits={deficits} flip_need={flip_need} -> {len(jobs)} jobs, {workers} workers")
    if not jobs:
        print("nothing to generate; already at target.")
        return False

    rng = random.Random(seed + sum(counts.values()))
    rng.shuffle(jobs)

    def work(job):
        kind, arg = job
        if kind == "single":
            rec = gen_one(client, arg)
            return [rec] if rec else []
        return gen_flip_pair(client, random.choice(prompts.FLIP_SENTENCES), arg)

    produced, consec_fail, done = 0, 0, 0
    wedged = False
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            recs = fut.result()
            if recs:
                for r in recs:
                    _append(raw_path, r)
                    mon.mark(samples=1)
                produced += len(recs)
                consec_fail = 0
            else:
                consec_fail += 1
            if done % 25 == 0:
                print(f"  [{done}/{len(jobs)}] produced={produced} consec_fail={consec_fail}", flush=True)
            if consec_fail >= max_consec_fail:
                wedged = True
                print(f"\nWATCHDOG: {consec_fail} consecutive failures -> engine likely wedged.")
                print(f"Progress saved to {raw_path}. Supervisor will restart + resume.")
                for f2 in futs:
                    f2.cancel()
                break
    mon.mark(tokens=client.tokens_used)
    print(f"segment produced {produced} records (wedged={wedged})")
    return wedged


def finalize(raw_path: str, out_path: str, seed: int,
             ratios=(0.8, 0.1, 0.1)) -> Dict[str, Any]:
    """Read the raw append-log, dedup, assign ids + splits, write the clean dataset."""
    rows: List[Dict[str, Any]] = []
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    seen, deduped = set(), []
    for r in rows:
        h = hashlib.sha1((r["query"] + "\x00" + r["context"]).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(r)

    rng = random.Random(seed)
    counters: Dict[str, int] = {}
    free = []
    for r in deduped:
        t = r["attack_type"]
        n = counters.get(t, 0)
        r["id"] = f"{t}-{n:06d}"
        counters[t] = n + 1
        if r.get("flip_pair_id") is not None:
            r["split"] = "test"
        else:
            free.append(r)
    rng.shuffle(free)
    n_tr, n_va = int(len(free) * ratios[0]), int(len(free) * ratios[1])
    for i, r in enumerate(free):
        r["split"] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for r in deduped:
            f.write(json.dumps(r) + "\n")

    by_label, by_type, by_split = Counter(), Counter(), Counter()
    for r in deduped:
        by_label[r["label"]] += 1
        by_type[r["attack_type"]] += 1
        by_split[r["split"]] += 1
    return {
        "total": len(deduped), "raw_rows": len(rows), "duplicates_removed": len(rows) - len(deduped),
        "by_label": dict(by_label), "by_attack_type": dict(by_type), "by_split": dict(by_split),
        "flip_records": sum(1 for r in deduped if r.get("flip_pair_id") is not None),
    }


def _complete(raw_path, n_benign, n_adversarial, n_flips) -> bool:
    targets = _targets(n_benign, n_adversarial)
    counts, flip_pairs = load_raw(raw_path)
    if any(counts.get(t, 0) < targets[t] for t in targets):
        return False
    return len(flip_pairs) >= n_flips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-benign", type=int, default=3000)
    ap.add_argument("--n-adversarial", type=int, default=2000)
    ap.add_argument("--n-flips", type=int, default=300)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260624)
    ap.add_argument("--raw", default="data/dataset.teacher.raw.jsonl")
    ap.add_argument("--out", default="data/dataset.teacher.jsonl")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-consec-fail", type=int, default=25)
    ap.add_argument("--resume", action="store_true", help="continue from the raw log")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_benign, args.n_adversarial, args.n_flips, args.workers = 8, 8, 2, 8
        args.raw, args.out = "data/smoke.raw.jsonl", "data/smoke.teacher.jsonl"

    if not args.resume and os.path.exists(args.raw):
        os.remove(args.raw)
        print(f"--resume not set: cleared {args.raw} for a fresh run")

    os.makedirs(os.path.dirname(args.raw) or ".", exist_ok=True)
    random.seed(args.seed)
    client = TeacherClient(args.endpoint, args.model, temperature=args.temperature)

    mon = HardwareMonitor("datagen-smoke" if args.smoke else "datagen-full", interval=1.0)
    mon.start()
    t0 = time.time()
    wedged = generate(client, args.n_benign, args.n_adversarial, args.n_flips,
                      args.workers, args.seed, mon, args.raw, args.max_consec_fail)
    mon.stop()
    mon.print_summary()

    rep = finalize(args.raw, args.out, args.seed)
    rep["elapsed_s"] = round(time.time() - t0, 1)
    rep["telemetry"] = mon.summary
    os.makedirs("results", exist_ok=True)
    meta = "results/datagen_smoke.json" if args.smoke else "results/datagen_report.json"
    with open(meta, "w") as f:
        json.dump(rep, f, indent=2)

    print(f"\nWrote {rep['total']} records -> {args.out} (dups removed: {rep['duplicates_removed']})")
    print(f"  by_label: {rep['by_label']}")
    print(f"  by_split: {rep['by_split']}  flips: {rep['flip_records']}")
    print(f"  report -> {meta}")

    if _complete(args.raw, args.n_benign, args.n_adversarial, args.n_flips):
        print("DATAGEN COMPLETE")
        sys.exit(0)
    print("DATAGEN INCOMPLETE (resume needed)")
    sys.exit(2 if wedged else 3)


if __name__ == "__main__":
    main()
