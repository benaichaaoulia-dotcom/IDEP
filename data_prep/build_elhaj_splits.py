#!/usr/bin/env python3
"""ElHaj 2018 weak-regime split builder.

Official train/dev/test kept as the split skeleton; the weak regime is applied
to TRAIN only:
  * source-group dedup on train: normalize (NFC, lowercase, keep Arabic-script
    chars + spaces — the frozen MADAR normalizer, data_prep/build_splits.py)
    -> exact-text union + character 3-gram Jaccard >= 0.6 graph edges
    (candidate pairs via inverted c-gram index, co-occurrence >= 4)
    -> connected components = source groups; group ID = min member row index.

  * group-level FNV-1a 64-bit split with SPLIT_SEED 2026-11-20-RUN-2026-11-20-02:
    rank = FNV-1a64("{group_id}|{SPLIT_SEED}"); lowest 5/10/20% of ranks ->
    SEED@f (labeled); rest -> POOL@f (fine labels withheld).
  * dev = validation; test locked (one final_eval read per arm).

All ElHaj fibers are singletons (5 classes EGY/NOR/GLF/LAV/MSA, no
non-injective fiber) => the IDEP 5-class regime runs with K_p = 0 (all
replaced rows exact-lifted).

Deterministic: pure function of the three checksummed CSVs + SPLIT_SEED
(no timestamps, no RNG) => byte-identical regeneration.
"""
import os
import csv
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

DATA_DIR = Path(os.path.join(os.environ.get("IDEP_DATA_ROOT", "data"), "elhaj_arabic_dialects_2018"))
FILES = {
    "train": DATA_DIR / "arabic_dialects_train.csv",
    "dev": DATA_DIR / "arabic_dialects_dev.csv",
    "test": DATA_DIR / "arabic_dialects_test.csv",
}
OUT = Path(__file__).resolve().parent.parent / "splits" / "elhaj"

SPLIT_SEED = "2026-11-20-RUN-2026-11-20-02"
FRACTIONS = [5, 10, 20]
JACCARD_MIN = 0.6
CO_OCCUR_MIN = 4

# frozen md5s
EXPECTED_MD5 = {
    "arabic_dialects_train.csv": "bca5485507601a324125266110230a75",
    "arabic_dialects_dev.csv": "276e368cc661561f009304f45d6f8dc8",
    "arabic_dialects_test.csv": "e7333634e778d81eae208629fcc0ab40",
}

ARABIC_RANGES = (
    (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF), (0xFE70, 0xFEFF),
)


def is_arabic(c: str) -> bool:
    o = ord(c)
    return any(lo <= o <= hi for lo, hi in ARABIC_RANGES)


def normalize(text: str) -> str:
    """Frozen normalizer (identical to data_prep/build_splits.py)."""
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


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["sentence", "dialect"], f"unexpected header in {path}: {rows[0]}"
    return [(text, label) for text, label in rows[1:]]


def main() -> int:
    # ---- 0. checksum verification ----
    for name, path in FILES.items():
        got = _md5(path)
        assert got == EXPECTED_MD5[path.name], f"{path.name}: md5 {got} != {EXPECTED_MD5[path.name]}"
    print(f"[load] checksums verified (3/3)")

    train = load_csv(FILES["train"])
    dev = load_csv(FILES["dev"])
    test = load_csv(FILES["test"])
    n_train, n_dev, n_test = len(train), len(dev), len(test)
    print(f"[load] train={n_train} dev={n_dev} test={n_test}")
    assert (n_train, n_dev, n_test) == (8079, 1010, 1013), (n_train, n_dev, n_test)

    # ---- 1. train source-group dedup (exact union + Jaccard >= 0.6) ----
    n = n_train
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

    norm = [normalize(t) for t, _ in train]
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

    shared = {}
    for g, posting in index.items():
        L = len(posting)
        for a in range(L):
            ia = posting[a]
            for b in range(a + 1, L):
                ib = posting[b]
                key = (ia, ib) if ia < ib else (ib, ia)
                shared[key] = shared.get(key, 0) + 1

    candidates = [k for k, v in shared.items() if v >= CO_OCCUR_MIN]
    n_edges = 0
    for (ia, ib) in candidates:
        sa, sb = gram_sets[ia], gram_sets[ib]
        inter = shared[(ia, ib)]
        union_size = len(sa) + len(sb) - inter
        if union_size > 0 and inter / union_size >= JACCARD_MIN:
            union(ia, ib)
            n_edges += 1

    members = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)
    groups = {}
    for root, mem in members.items():
        gid = min(mem)
        groups[gid] = sorted(mem)
    print(f"[groups] train source groups: {len(groups)} (jaccard>={JACCARD_MIN} edges: {n_edges})")

    # ---- 2. FNV-1a group-rank split (seed fractions; rest = pool) ----
    ranks = {gid: fnv1a_64(f"{gid}|{SPLIT_SEED}".encode("utf-8")) for gid in groups}
    order = sorted(groups.keys(), key=lambda g: (ranks[g], g))
    seed_sets = {}
    for f in FRACTIONS:
        k = int(f / 100.0 * len(order))   # floor of lowest f% of ranks
        seed_sets[f] = set(order[:k])

    row_group = [None] * n
    for gid, mem in groups.items():
        for r in mem:
            row_group[r] = gid

    # ---- 3. write pinned manifest (train rows: seed flags; dev/test: locked) ----
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "sentences.jsonl", "w", encoding="utf-8") as f:
        row = 0
        for text, label in train:
            rec = {"row": row, "source": "train", "group_id": row_group[row], "label": label,
                   "seed": {str(fr): row_group[row] in seed_sets[fr] for fr in FRACTIONS}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            row += 1
        for text, label in dev:
            f.write(json.dumps({"row": row, "source": "dev", "group_id": -1, "label": label},
                               ensure_ascii=False) + "\n")
            row += 1
        for text, label in test:
            f.write(json.dumps({"row": row, "source": "test", "group_id": -1, "label": label},
                               ensure_ascii=False) + "\n")
            row += 1

    def sent_count(gset):
        return sum(len(groups[g]) for g in gset)

    class_counts = {}
    for _, label in train:
        class_counts[label] = class_counts.get(label, 0) + 1

    summary = {
        "split_seed": SPLIT_SEED,
        "dataset": "elhaj_arabic_dialects_2018",
        "generator": "data_prep/build_elhaj_splits.py",
        "jaccard_min": JACCARD_MIN,
        "n_rows": {"train": n_train, "dev": n_dev, "test": n_test},
        "n_train_groups": len(groups),
        "seed": {str(fr): {"groups": len(seed_sets[fr]), "sentences": sent_count(seed_sets[fr])}
                 for fr in FRACTIONS},
        "pool": {str(fr): {"groups": len(order) - len(seed_sets[fr]),
                           "sentences": n_train - sent_count(seed_sets[fr])}
                 for fr in FRACTIONS},
        "class_counts_train": class_counts,
        "manifest_md5": {p.name: _md5(p) for p in FILES.values()},
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
