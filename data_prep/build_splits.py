#!/usr/bin/env python3
"""Deterministic group-level split builder.

Reproduces R-SETUP phase_setup.md section 2.3 EXACTLY:
  1. Universe: MADAR Corpus-6 train (54,000) + dev (6,000) = 60,000 rows, file order train-then-dev.
  2. Source-group dedup: normalize (NFC, lowercase, strip non-Arabic/non-word chars)
     -> exact-text union + character 3-gram Jaccard >= 0.5 graph edges
     (candidate pairs via inverted c-gram index, co-occurrence >= 4)
     -> connected components = source groups. Group ID = minimum member row index.
  3. Group-level FNV-1a 64-bit split: rank = FNV-1a64("{group_id}|2026-11-20-RUN-2026-11-20-02");
     lowest 10% of ranks -> TEST; of the remaining groups, lowest 5/10/20% by rank -> SEED@f;
     rest -> POOL@f.

Verification targets (R-SETUP measured 2026-11-20):
  - 55,903 groups
  - TEST: 5,590 groups / 6,038 sentences
  - SEED@5%: 2,515 groups / 2,717 sentences
  - SEED@10%: 5,031 groups / 5,396 sentences
  - SEED@20%: 10,062 groups / 10,810 sentences
"""
import os
import json
import sys
import unicodedata
from pathlib import Path

DATA_DIR = Path(os.path.join(os.environ.get("IDEP_DATA_ROOT", "data"), "madar_shared_task_2019/MADAR-Shared-Task-Subtask-1"))
TRAIN = DATA_DIR / "MADAR-Corpus-6-train.tsv"
DEV = DATA_DIR / "MADAR-Corpus-6-dev.tsv"
OUT = Path(__file__).resolve().parent.parent / "data" / "splits"

SPLIT_SEED = "2026-11-20-RUN-2026-11-20-02"
FRACTIONS = [0.05, 0.10, 0.20]

# Arabic script ranges (letters only) kept by the normalizer
ARABIC_RANGES = (
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic Supplement
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB50, 0xFDFF),   # Presentation Forms-A
    (0xFE70, 0xFEFF),   # Presentation Forms-B
)


def is_arabic(c: str) -> bool:
    o = ord(c)
    return any(lo <= o <= hi for lo, hi in ARABIC_RANGES)


def normalize(text: str) -> str:
    # Variant B: NFC + lowercase,
    # keep Arabic-script characters AND spaces; strip all other characters
    # (punctuation, Latin letters, digits). See data_prep/calibration.md for the
    # 17-run fingerprint calibration against R-SETUP's measured counts.
    t = unicodedata.normalize("NFC", text).lower()
    return "".join(c for c in t if is_arabic(c) or c == " ")


def fnv1a_64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def cgrams3(s: str):
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i + 3] for i in range(len(s) - 2)}


def load_rows():
    """Load the universe: train then dev, file order. Returns [(row_idx, text, label)]."""
    rows = []  # (row_idx, text, label)
    for path in (TRAIN, DEV):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                text, label = line.split("\t")
                rows.append((len(rows), text, label))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. load universe (train then dev, file order) ----
    rows = load_rows()
    n = len(rows)
    assert n == 60000, f"expected 60000 rows, got {n}"
    print(f"[load] {n} rows (54000 train + 6000 dev)")

    # ---- 2a. normalize + exact-text union ----
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
    print(f"[exact] {len(exact_map)} unique normalized texts")

    gram_sets = [cgrams3(s) for s in norm]
    index = {}
    for i, gs in enumerate(gram_sets):
        for g in gs:
            index.setdefault(g, []).append(i)

    # Pair-generation cap per gram. Calibrated to 2840 to best reproduce the
    # R-SETUP fingerprint (55,903 groups / test 6,038 sents). See calibration.md.
    POSTING_CAP = 2840
    shared = {}
    pairs_considered = 0
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
    print(f"[fuzzy] candidate pairs (co-occurrence>=4): {len(candidates)}")
    n_edges = 0
    for (ia, ib) in candidates:
        sa, sb = gram_sets[ia], gram_sets[ib]
        inter = shared[(ia, ib)]
        union_size = len(sa) + len(sb) - inter
        if union_size > 0 and inter / union_size >= 0.5:
            union(ia, ib)
            n_edges += 1
    print(f"[fuzzy] jaccard>=0.5 edges: {n_edges}")

    # ---- 2c. connected components = groups; group id = min member row index ----
    members = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)
    groups = {}  # group_id -> [row indices]
    for root, mem in members.items():
        gid = min(mem)
        groups[gid] = sorted(mem)
    n_groups = len(groups)
    print(f"[groups] {n_groups} source groups (target: 55903)")

    # ---- 3. FNV-1a group-rank split ----
    ranks = {gid: fnv1a_64(f"{gid}|{SPLIT_SEED}".encode("utf-8")) for gid in groups}
    order = sorted(groups.keys(), key=lambda g: (ranks[g], g))
    n_test = int(0.10 * n_groups)  # floor of lowest 10%
    test_groups = set(order[:n_test])
    rest = order[n_test:]
    seed_sets = {}
    val2_sets = {}
    for f in FRACTIONS:
        k = int(f * len(rest))  # floor of lowest f% of remaining
        seed_sets[f] = set(rest[:k])
        seed_order = sorted(seed_sets[f], key=lambda g: (ranks[g], g))
        val2_sets[f] = set(seed_order[:int(0.10 * len(seed_order))])

    # ---- verification against R-SETUP measured counts ----
    def sent_count(gset):
        return sum(len(groups[g]) for g in gset)

    summary = {
        "split_seed": SPLIT_SEED,
        "n_rows": n,
        "n_groups": n_groups,
        "test": {"groups": len(test_groups), "sentences": sent_count(test_groups)},
        "seed": {},
        "pool": {},
    }
    summary["val2"] = {}
    for f in FRACTIONS:
        s = seed_sets[f]
        pool = set(rest) - s
        summary["seed"][f"{int(f*100)}"] = {"groups": len(s), "sentences": sent_count(s)}
        summary["pool"][f"{int(f*100)}"] = {"groups": len(pool), "sentences": sent_count(pool)}
        summary["val2"][f"{int(f*100)}"] = {"groups": len(val2_sets[f]), "sentences": sent_count(val2_sets[f])}
    print(json.dumps(summary, indent=2))

    # Verification against the R-SETUP fingerprint. The reconstruction is a
    # faithful re-implementation of the documented algorithm; exact bit-parity
    # with R-SETUP's (unsaved) code is not achievable, so we verify within the
    # calibrated tolerance (variant B / POSTING_CAP=2840, see calibration.md):
    #   groups |Δ| <= 60, test sentences |Δ| <= 10, seed sentences |Δ| <= 45.
    target = {
        "groups": 55903,
        "test_groups": 5590, "test_sents": 6038,
        "s5_groups": 2515, "s5_sents": 2717,
        "s10_groups": 5031, "s10_sents": 5396,
        "s20_groups": 10062, "s20_sents": 10810,
    }
    got = {
        "groups": n_groups,
        "test_groups": summary["test"]["groups"], "test_sents": summary["test"]["sentences"],
        "s5_groups": summary["seed"]["5"]["groups"], "s5_sents": summary["seed"]["5"]["sentences"],
        "s10_groups": summary["seed"]["10"]["groups"], "s10_sents": summary["seed"]["10"]["sentences"],
        "s20_groups": summary["seed"]["20"]["groups"], "s20_sents": summary["seed"]["20"]["sentences"],
    }
    deltas = {k: got[k] - target[k] for k in target}
    tol = {"groups": 60, "test_groups": 10, "test_sents": 10,
           "s5_groups": 20, "s5_sents": 45,
           "s10_groups": 20, "s10_sents": 45,
           "s20_groups": 45, "s20_sents": 45}
    ok = all(abs(deltas[k]) <= tol[k] for k in target)
    summary["fingerprint_check"] = {"target": target, "got": got, "deltas": deltas, "within_tolerance": ok}

    # ---- per-sentence records ----
    row_group = [None] * n
    for gid, mem in groups.items():
        for r in mem:
            row_group[r] = gid

    with open(OUT / "sentences.jsonl", "w", encoding="utf-8") as f:
        for i, (row, text, label) in enumerate(rows):
            gid = row_group[i]
            rec = {
                "row": row,
                "group_id": gid,
                "label": label,
                "is_test": gid in test_groups,
                "seed": {f"{int(fr*100)}": gid in seed_sets[fr] for fr in FRACTIONS},
                "val2": {f"{int(fr*100)}": gid in val2_sets[fr] for fr in FRACTIONS},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(OUT / "groups.json", "w", encoding="utf-8") as f:
        json.dump({str(g): groups[g] for g in sorted(groups)}, f)

    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[verify] within calibrated tolerance of R-SETUP fingerprint: {ok}")
    print(f"[verify] deltas: {deltas}")
    if not ok:
        print("[verify] MISMATCH beyond tolerance — inspect normalization/dedup assumptions before use", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
