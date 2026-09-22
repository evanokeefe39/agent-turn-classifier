"""Assign domains to the coverage sample and report coverage.

These assignments are a HUMAN judgement pass, not a measured classifier — the
`--by` flag exists to make that explicit in the output. An earlier attempt to
automate this scored 33% agreement against hand labels and was discarded; the
honest thing is a stated sample with a stated author, not a fake spread.

Usage:
    python examples/coverage_assign.py            # report
    python examples/coverage_assign.py --list-unmapped
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Judgement pass over .out/coverage_sample.json (seed 20260922, 120 spans).
# Value is a declared domain, "UNMAPPED" (agent work, not data engineering),
# or "NONE" (not work).
# fmt: off
# Assignments are keyed by SPAN ID, not sample index. An index-keyed table only
# reproduces against the exact sample it was written from — and the corpus drifts
# as work continues, so a re-run with the same seed picks different spans and an
# index table would silently apply yesterday's labels to different work. Keyed by
# id, a moved sample FAILS LOUDLY instead of lying.
#
# `ASSIGNMENTS` is a judgement pass, not a reproduction: it is one reader's
# placement of spans into the declared catalog. Treat it as a measurement with
# error, and read `--passes` for how far the number moves with the vocabulary.
ASSIGNMENTS_FILE = Path(__file__).with_name("coverage_assignments.json")
FALLBACK_FILE = Path(".out/assignments.json")

# The catalog's domain axis, complete. A pass offering a subset of these cannot
# report "domain X never appeared" — it can only report that it never asked.
CATALOG_DOMAINS = (
    "Data Integration", "Data Modeling", "Data Storage", "Data Quality",
    "Data Governance", "Data Security", "Data Architecture", "Metadata",
    "Analytics and BI", "Master Data", "ML and AI", "Orchestration",
    "Platform", "Product", "Project Management",
)


def load_assignments(sample: dict) -> dict[str, str]:
    """Load id -> domain, failing loudly if it does not match the sample."""
    path = ASSIGNMENTS_FILE if ASSIGNMENTS_FILE.is_file() else FALLBACK_FILE
    if not path.is_file():
        raise SystemExit(
            f"no assignments found at {ASSIGNMENTS_FILE} or {FALLBACK_FILE}. "
            "A judgement pass cannot be inferred — it has to be recorded."
        )
    rec = json.loads(path.read_text(encoding="utf-8"))
    table = rec["assignments"] if "assignments" in rec else rec
    by_id = {v["id"] if isinstance(v, dict) else k: (v["domain"] if isinstance(v, dict) else v)
             for k, v in table.items()}

    ids = [s["id"] for s in sample["spans"]]
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(
            f"{path} does not cover this sample: {len(missing)} of {len(ids)} span ids "
            f"are unassigned (first: {missing[0]}). The corpus moved since the pass was "
            "recorded, so these placements describe different spans. Re-judge, then "
            "re-record. Refusing to apply stale labels."
        )
    return {i: by_id[i] for i in ids}

# Which projects count as "data work" is USER-SPECIFIC and must not be hardcoded
# in a public repo. It is supplied as a flat list, one project name per line, via
# --data-projects (or --data-projects from the sample's own stamp if the checker
# recorded one). Coverage within data-purpose projects is the fair test of the
# taxonomy; coverage across everything mostly measures which app the user
# happened to be building.
def load_data_projects(path: str | None, stamp: dict) -> set[str]:
    if path:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}
    return set(stamp.get("data_projects") or [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-file", default=".out/coverage_sample.json")
    ap.add_argument("--data-projects",
                    help="file listing data-purpose project names, one per line")
    args = ap.parse_args()

    data = json.loads(Path(args.sample_file).read_text(encoding="utf-8"))
    spans = data["spans"]
    data_projects = load_data_projects(args.data_projects, data.get("stamp") or {})
    if not data_projects:
        print("NOTE: no data-project list supplied (--data-projects), so the "
              "data-vs-non-data split is skipped. This split is user-specific "
              "and is deliberately not hardcoded.\n")

    assigned = load_assignments(data)

    rows = []
    for s in spans:
        rows.append({**s, "domain": assigned[s["id"]], "is_data_project": s["project"] in data_projects})

    def report(subset, label):
        n = len(subset)
        if not n:
            print(f"{label}: empty")
            return
        placed = [r for r in subset if r["domain"] not in ("UNMAPPED", "NONE")]
        unmapped = [r for r in subset if r["domain"] == "UNMAPPED"]
        none = [r for r in subset if r["domain"] == "NONE"]
        print(f"\n{label}  (n={n})")
        print(f"  placed in a declared domain   {len(placed):3}  ({len(placed)/n:5.1%})")
        print(f"  agent work, no data workflow  {len(unmapped):3}  ({len(unmapped)/n:5.1%})")
        print(f"  not work (NONE)               {len(none):3}  ({len(none)/n:5.1%})")
        c = Counter(r["domain"] for r in placed)
        if c:
            print(f"  domains seen: " + ", ".join(f"{k}×{v}" for k, v in c.most_common()))

    report(rows, "ALL sampled spans")
    print("\n  by band — reply_only spans are the agent ANSWERING a question, so they")
    print("  answer a different question from action spans and are reported apart:")
    for b in ("reply_only", "small_1_3", "medium_4_10", "large_11_25", "very_large_26plus"):
        sub = [r for r in rows if r.get("band") == b]
        if sub:
            placed = sum(1 for r in sub if r["domain"] not in ("UNMAPPED", "NONE"))
            print(f"    {b:20s} n={len(sub):3d}  placed {placed:3d} ({placed/len(sub):5.1%})")
    if data_projects:
        report([r for r in rows if r["is_data_project"]], "Spans in DATA-purpose projects")
        report([r for r in rows if not r["is_data_project"]], "Spans in NON-data projects")

    unmapped_data = [r for r in rows if r["is_data_project"] and r["domain"] == "UNMAPPED"]
    if unmapped_data:
        print("\n--- UNMAPPED spans inside data-purpose projects (the taxonomy's real gaps) ---")
        for r in unmapped_data:
            hint = (r["intents"][:3] if r["intents"] else [r["reply"][:110]])
            print(f"\n  {r['project']} · {r['band']}")
            for h in hint:
                print(f"      {h[:104]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
