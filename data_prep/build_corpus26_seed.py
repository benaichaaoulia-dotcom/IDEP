#!/usr/bin/env python3
"""Corpus-26 auxiliary SEED@f derivation for B3.

Applies the SAME deterministic group-level split algorithm as the frozen MADAR
split (data_prep/build_splits.py): normalize (NFC, lowercase, Arabic+space only),
exact-text union + char 3-gram Jaccard >= 0.5 edges (posting cap 2840,
co-occurrence >= 4), connected components = groups (id = min member row),
FNV-1a64("{group_id}|{SPLIT_SEED}") rank split: lowest 10% -> TEST (unused here),
lowest 10% of the rest -> SEED@10. Universe = MADAR-Corpus-26-train.tsv only
(41,600 rows; the B3 auxiliary resource never enters any other method's data).

Output: splits/corpus26_seed.jsonl — one record per row:
  {"row", "text", "label", "group_id", "seed": {"10": bool}}
"""
import os
import json
import sys
import unicodedata
from pathlib import Path

DATA_DIR = Path(os.path.join(os.environ.get("IDEP_DATA_ROOT", "data"), "madar_shared_task_2019/MADAR-Shared-Task-Subtask-1"))
TRAIN26 = DATA_DIR / "MADAR-Corpus-26-train.tsv"
OUT = Path(__file__).resolve().parent.parent / "splits" / "corpus26_seed.jsonl"

SPLIT_SEED = "2026-11-20-RUN-2026-11-20-02"
ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
                 (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))


def is_arabic(c):
    o = ord(c)
    return any(lo <= o <= hi for lo, hi in ARABIC_RANGES)


def normalize(text):
    t = unicodedata.normalize("NFC", text).lower()
    return "".join(c for c in t if is_arabic(c) or c == " ")


def fnv1a_64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def cgrams3(s):
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i + 3] for i in range(len(s) - 2)}


def main():
    rows = []
    with open(TRAIN26, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            text, label = line.split("\t")
            rows.append((len(rows), text, label))
    n = len(rows)
    print(f"[load] {n} rows (Corpus-26 train)")

    norm = [normalize(t) for _, t, _ in rows]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    exact_map = {}
    for i, s in enumerate(norm):
        if not s:
            continue
        if s in exact_map:
            union(exact_map[s], i)
        else:
            exact_map[s] = i

    gram_sets = [cgrams3(s) for s in norm]
    index = {}
    for i, gs in enumerate(gram_sets):
        for g in gs:
            index.setdefault(g, []).append(i)
    POSTING_CAP = 2840
    shared = {}
    for g, posting in index.items():
        if len(posting) > POSTING_CAP:
            continue
        L = len(posting)
        for a in range(L):
            ia = posting[a]
            for b in range(a + 1, L):
                ib = posting[b]
                key = (ia, ib) if ia < ib else (ib, ia)
                shared[key] = shared.get(key, 0) + 1
    candidates = [k for k, v in shared.items() if v >= 4]
    n_edges = 0
    for (ia, ib) in candidates:
        sa, sb = gram_sets[ia], gram_sets[ib]
        inter = shared[(ia, ib)]
        union_size = len(sa) + len(sb) - inter
        if union_size > 0 and inter / union_size >= 0.5:
            union(ia, ib)
            n_edges += 1
    print(f"[fuzzy] candidates={len(candidates)} edges={n_edges}")

    members = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)
    groups = {}
    for root, mem in members.items():
        gid = min(mem)
        groups[gid] = sorted(mem)
    print(f"[groups] {len(groups)} source groups")

    ranks = {gid: fnv1a_64(f"{gid}|{SPLIT_SEED}".encode("utf-8")) for gid in groups}
    order = sorted(groups.keys(), key=lambda g: (ranks[g], g))
    n_test = int(0.10 * len(groups))
    rest = order[n_test:]
    k = int(0.10 * len(rest))
    seed_set = set(rest[:k])

    row_group = [None] * n
    for gid, mem in groups.items():
        for r in mem:
            row_group[r] = gid

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_seed_rows = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for i, (row, text, label) in enumerate(rows):
            gid = row_group[i]
            rec = {"row": row, "text": text, "label": label, "group_id": gid,
                   "seed": {"10": gid in seed_set}}
            if rec["seed"]["10"]:
                n_seed_rows += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {"split_seed": SPLIT_SEED, "n_rows": n, "n_groups": len(groups),
               "seed10_groups": len(seed_set), "seed10_rows": n_seed_rows,
               "note": "auxiliary B3 training resource only"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
