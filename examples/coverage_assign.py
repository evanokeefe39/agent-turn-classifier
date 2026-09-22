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
D = {
 1:"UNMAPPED",  2:"UNMAPPED",  3:"UNMAPPED",  4:"UNMAPPED",  5:"UNMAPPED",
 6:"UNMAPPED",  7:"UNMAPPED",  8:"UNMAPPED",  9:"UNMAPPED", 10:"Analytics and BI",
11:"UNMAPPED", 12:"UNMAPPED", 13:"UNMAPPED", 14:"UNMAPPED", 15:"UNMAPPED",
16:"UNMAPPED", 17:"NONE",     18:"UNMAPPED", 19:"UNMAPPED", 20:"UNMAPPED",
21:"UNMAPPED", 22:"UNMAPPED", 23:"UNMAPPED", 24:"UNMAPPED", 25:"UNMAPPED",
26:"Data Storage", 27:"UNMAPPED", 28:"UNMAPPED", 29:"UNMAPPED", 30:"UNMAPPED",
31:"UNMAPPED", 32:"NONE",     33:"Data Integration", 34:"UNMAPPED", 35:"UNMAPPED",
36:"Data Security", 37:"UNMAPPED", 38:"UNMAPPED", 39:"UNMAPPED", 40:"UNMAPPED",
41:"UNMAPPED", 42:"UNMAPPED", 43:"UNMAPPED", 44:"NONE",     45:"UNMAPPED",
46:"Analytics and BI", 47:"Data Integration", 48:"UNMAPPED", 49:"UNMAPPED", 50:"UNMAPPED",
51:"UNMAPPED", 52:"UNMAPPED", 53:"NONE",     54:"UNMAPPED", 55:"UNMAPPED",
56:"UNMAPPED", 57:"UNMAPPED", 58:"UNMAPPED", 59:"Data Integration", 60:"Data Integration",
61:"UNMAPPED", 62:"UNMAPPED", 63:"UNMAPPED", 64:"UNMAPPED", 65:"UNMAPPED",
66:"UNMAPPED", 67:"UNMAPPED", 68:"NONE",     69:"Data Integration", 70:"UNMAPPED",
71:"UNMAPPED", 72:"UNMAPPED", 73:"UNMAPPED", 74:"UNMAPPED", 75:"UNMAPPED",
76:"UNMAPPED", 77:"UNMAPPED", 78:"UNMAPPED", 79:"UNMAPPED", 80:"UNMAPPED",
81:"UNMAPPED", 82:"UNMAPPED", 83:"Data Quality", 84:"UNMAPPED", 85:"UNMAPPED",
86:"UNMAPPED", 87:"UNMAPPED", 88:"UNMAPPED", 89:"UNMAPPED", 90:"UNMAPPED",
91:"UNMAPPED", 92:"UNMAPPED", 93:"UNMAPPED", 94:"UNMAPPED", 95:"UNMAPPED",
96:"UNMAPPED", 97:"UNMAPPED", 98:"UNMAPPED", 99:"UNMAPPED", 100:"UNMAPPED",
101:"UNMAPPED", 102:"UNMAPPED", 103:"UNMAPPED", 104:"NONE",    105:"UNMAPPED",
106:"UNMAPPED", 107:"UNMAPPED", 108:"UNMAPPED", 109:"UNMAPPED", 110:"UNMAPPED",
111:"UNMAPPED", 112:"Data Integration", 113:"UNMAPPED", 114:"UNMAPPED", 115:"UNMAPPED",
116:"UNMAPPED", 117:"NONE",    118:"UNMAPPED", 119:"UNMAPPED", 120:"UNMAPPED",
}
# fmt: on

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
    ap.add_argument("--list-unmapped", action="store_true")
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

    rows = []
    for i, s in enumerate(spans, 1):
        rows.append({**s, "domain": D[i], "is_data_project": s["project"] in data_projects})

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
    if data_projects:
        report([r for r in rows if r["is_data_project"]], "Spans in DATA-purpose projects")
        report([r for r in rows if not r["is_data_project"]], "Spans in NON-data projects")

    if args.list_unmapped:
        print("\n--- UNMAPPED spans in data-purpose projects (the taxonomy's real gaps) ---")
        for i, r in enumerate(rows, 1):
            if r["is_data_project"] and r["domain"] == "UNMAPPED":
                hint = (r["intents"][:3] if r["intents"] else [r["reply"][:110]])
                print(f"\n[{i}] {r['project']} · {r['band']}")
                for h in hint:
                    print(f"    {h[:104]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
