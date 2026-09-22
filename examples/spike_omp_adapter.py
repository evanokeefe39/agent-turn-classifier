"""SPIKE 2 — the OMP adapter against real transcripts.

Validates the assumptions the plan makes about real session data, which the
synthetic fixture cannot exercise:

  1. SCHEMA. The private pipeline's SQL reads `role`, `ctype`, `ts` as if they
     were top-level columns. Real transcripts nest them:
         role       -> message.role
         stopReason -> message.stopReason
         ctype      -> customType
         ts         -> timestamp
     An adapter written against the SQL's column names silently reads None for
     every one of those fields. That failure mode is what this spike exists to
     catch, and it caught it on the first real file.

  2. BYTE IDENTITY. The plan claims a turn's line span can be reconciled against
     the file bytewise. Verify on a real 5MB transcript.

  3. TURN COUNT RECONCILIATION. Compare this adapter's user-turn count against
     the private pipeline's figures, per session, so the numbers are attributable
     rather than asserted.

Usage:
    python examples/spike_omp_adapter.py                 # default: largest session
    python examples/spike_omp_adapter.py --session <id>   # a specific session
    python examples/spike_omp_adapter.py --all            # every session in -repos
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SESSIONS_ROOT = Path(os.environ.get("ATC_SESSIONS_ROOT", r"C:\Users\evano\.omp\agent\sessions"))

# The seven harness-probe / dot-dir noise directories the plan excludes. They are
# OMP's own test scratch, not real work, and counting them inflates the corpus.
NOISE_DIRS = {
    "-", "--C--tmp--", "--C--tmp-hookprobe--", "--C--tmp-probe-main--",
    "-.omp", "-.cache-otel-spike",
}


@dataclass
class Entry:
    """One parsed line of a transcript, with its byte span preserved."""
    index: int
    byte_start: int
    byte_end: int
    line: str
    kind: str
    role: str | None
    stop_reason: str | None
    custom_type: str | None
    timestamp: str | None
    data: dict = field(default_factory=dict)

    def js(self, *path: str):
        """Nested lookup, mirroring the private pipeline's c.j() helper."""
        cur = self.data
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur


def read_entries(path: Path) -> tuple[list[Entry], int]:
    """Parse a transcript, tracking exact byte spans.

    Reads binary and decodes per line so the byte accounting is exact — a
    character-count approximation would drift on any non-ASCII turn, and these
    transcripts contain emoji and CJK.
    """
    entries: list[Entry] = []
    offset = 0
    raw = path.read_bytes()  # binary: byte spans are the point
    for i, chunk in enumerate(raw.split(b"\n")):
        if not chunk.strip():
            offset += len(chunk) + 1
            continue
        try:
            data = json.loads(chunk.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            offset += len(chunk) + 1
            continue
        start = offset
        offset += len(chunk) + 1
        entries.append(Entry(
            index=i,
            byte_start=start,
            byte_end=start + len(chunk),
            line=chunk.decode("utf-8"),
            kind=data.get("type"),
            # NESTED, not top-level. This is the correction the spike produced.
            role=(data.get("message") or {}).get("role"),
            stop_reason=(data.get("message") or {}).get("stopReason"),
            custom_type=data.get("customType"),
            timestamp=data.get("timestamp"),
            data=data,
        ))
    return entries, len(raw)


def classify(prev: Entry | None, cur: Entry) -> str:
    """The plan's turn-kind CASE, ported from SQL to Python.

    Copied in structure from harness-research/scripts/session_retro/
    query_sessions.py:54 (turn_kinds), which is the private pipeline's
    established semantics — this tool replaces that regex classifier, so the
    labels must line up before they can be compared.

    The SQL reads `ctype` and `role` as top-level; here they are read from the
    parsed entry, which is where the real data keeps them.
    """
    if prev is None:
        return "first_of_session"
    gap = _gap_seconds(prev.timestamp, cur.timestamp)
    if prev.role == "assistant" and prev.stop_reason == "stop":
        return "reply_fast" if gap is not None and gap <= 120 else "reply_slow"
    if prev.custom_type == "tool_execution_start" and gap is not None and gap <= 10:
        return "interrupt_mid_run"
    if prev.role == "toolResult" and gap is not None and gap <= 10:
        return "interrupt_mid_run"
    if prev.role == "assistant" and prev.stop_reason == "aborted":
        return "after_abort"
    if prev.role == "assistant" and prev.stop_reason == "error":
        return "after_error"
    return "other"


def _gap_seconds(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    from datetime import datetime
    try:
        fa = datetime.fromisoformat(a.replace("Z", "+00:00"))
        fb = datetime.fromisoformat(b.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (fb - fa).total_seconds()


def user_turns(entries: list[Entry]) -> list[tuple[Entry, str]]:
    """User-role messages with the kind of turn each one starts.

    `prev` advances ONLY over `message`/`custom` entries, mirroring the private
    pipeline's window (`where type in ('message','custom')`). Advancing over
    every row lets a `title`, `session` or `compaction` header become a user
    turn's predecessor, which silently drops `first_of_session` and inflates
    `other` — the two symptoms that made the first run's distribution wrong.
    """
    out = []
    prev = None
    for e in entries:
        if e.kind not in ("message", "custom"):
            continue  # not in the SQL's partition window
        if e.kind == "message" and e.role == "user":
            out.append((e, classify(prev, e)))
        prev = e
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", help="session id or path fragment")
    ap.add_argument("--all", action="store_true", help="every real-repo session")
    ap.add_argument("--since", help="ISO timestamp lower bound (for reconciling "
                                    "against the private pipeline's windowed figures)")
    args = ap.parse_args()

    if not SESSIONS_ROOT.is_dir():
        print(f"sessions root not found: {SESSIONS_ROOT}")
        return 2

    if args.session:
        candidates = [p for p in SESSIONS_ROOT.rglob("*.jsonl") if args.session in p.name]
    elif args.all:
        candidates = [p for d in SESSIONS_ROOT.iterdir() if d.is_dir()
                      and d.name not in NOISE_DIRS and (d / "").exists()
                      for p in d.glob("*.jsonl")]
    else:
        candidates = [max(glob.glob(str(SESSIONS_ROOT / "-repos" / "*.jsonl")),
                          key=os.path.getsize)]

    if not candidates:
        print("no sessions matched")
        return 1

    total_turns = 0
    kind_totals: Counter[str] = Counter()
    sessions_hit = 0
    since = None
    if args.since:
        from datetime import datetime
        since = datetime.fromisoformat(args.since)
    for path in candidates:
        p = Path(path)
        entries, n_bytes = read_entries(p)
        turns = user_turns(entries)

        # Window filter, applied to the TURN (the private pipeline's predicate),
        # not to the file — a session straddling the boundary contributes only
        # its in-window turns, which is how 738/721/638 were counted.
        if since is not None:
            turns = [t for t in turns
                     if t[0].timestamp
                     and datetime.fromisoformat(t[0].timestamp.replace("Z", "+00:00")) >= since]
            if not turns:
                continue
        sessions_hit += 1

        # Byte identity: do the recorded spans reconstruct the file exactly?
        spans = sum(e.byte_end - e.byte_start for e in entries) + len(entries)
        identity = "exact" if spans == n_bytes else f"DRIFT {spans - n_bytes:+d}"

        kinds = Counter(k for _, k in turns)
        kind_totals.update(kinds)
        total_turns += len(turns)

        if not args.all:
            print(f"session      {p.name[:44]}")
            print(f"size         {n_bytes:,} bytes, {len(entries):,} entries")
            print(f"byte spans   {identity}")
            print(f"user turns   {len(turns)}")
            # Not entries[0]: the first record is a `title` header with
            # `updatedAt`, not a `timestamp` — reading it yields None.
            stamps = [e.timestamp for e in entries if e.timestamp]
            print(f"first ts     {min(stamps)}")
            print(f"last ts      {max(stamps)}")
            print(f"\nturn kinds   {dict(kinds.most_common())}")
            print("\nfirst 6 turns:")
            for e, k in turns[:6]:
                text = _text_of(e)
                print(f"  {e.timestamp[:19]}  {k:18} {text[:58]}")
            print("\nrole counts  "
                  f"{dict(Counter(e.role for e in entries if e.kind == 'message').most_common())}")
            print("customTypes  "
                  f"{dict(Counter(e.custom_type for e in entries if e.custom_type).most_common(6))}")

    if args.all:
        print(f"sessions     {sessions_hit}")
        print(f"user turns   {total_turns}")
        print(f"kinds        {dict(kind_totals.most_common())}")
    return 0


def _text_of(entry: Entry) -> str:
    """First text block of a message, collapsed — for a one-line preview."""
    content = (entry.data.get("message") or {}).get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return " ".join(str(block.get("text", "")).split())
    if isinstance(content, str):
        return " ".join(content.split())
    return ""


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# SPIKE 2 RESULT — 2026-09-22
#
# Validated against the live corpus (323 sessions, all project dirs, 36 repos).
#
# WHAT IT FOUND — one real bug, caught on the first real file:
#
#   The private pipeline's SQL reads `role`, `ctype`, `ts` as top-level columns.
#   Real transcripts NEST them:
#       role        -> message.role
#       stopReason  -> message.stopReason
#       ctype       -> customType
#       ts          -> timestamp
#   An adapter written against the SQL's column names reads None for every one
#   of those fields and silently classifies every turn as 'other'. This is the
#   failure the spike existed to catch, and the synthetic fixture could never
#   have surfaced it (the fixture has flat, friendly fields).
#
#   Second bug, in the reconciliation code rather than the adapter: an exclusion
#   list matched by SUBSTRING, so excluding "-repos" also excluded every
#   "-repos-<project>" directory and returned 0 turns — which reads as "no data"
#   rather than "bug". Exclusions must match exact directory names.
#
# WHAT IT CONFIRMED:
#
#   byte identity      EXACT on a 5,273,209-byte, 1,107-entry transcript.
#                      Recorded spans reconstruct the file with zero drift, so
#                      the plan's bytewise turn-span reconciliation holds.
#   role counts        toolResult 389 / assistant 271 / user 27
#   customTypes        tool_execution_start 389, advisor 19, async-result 2,
#                      launch-completion 1, orchestrate-notice 1, session_exit 1
#   turn kinds (1 session, corrected)
#                      reply_slow 19, reply_fast 4, first_of_session 1,
#                      after_abort 1, interrupt_mid_run 1, after_error 1
#   full corpus        4,459 user turns / 323 sessions, all time
#                      reply_slow 1907, reply_fast 936, interrupt_mid_run 542,
#                      after_abort 525, first_of_session 304, other 137,
#                      after_error 108
#   windowed 09-14     852 turns / 57 sessions; 795 excluding scratch dirs
#
# THIRD BUG, found by re-running after the advisory flagged the first numbers:
#   `prev` advanced over EVERY entry, so a `title`/`session`/`compaction`
#   header could become a user turn's predecessor. The private SQL restricts
#   its window to `where type in ('message','custom')`; the adapter did not.
#   Symptom: `first_of_session` was absent from the distribution entirely (it
#   can only fire when prev is None) and `other` was inflated 8x in one session
#   (1068 -> 137 corpus-wide). The corrected counts above are post-fix.
#   This is the same silent-mismatch class as the nested-field bug: the numbers
#   looked plausible, and only the missing bucket revealed the error.
#
# RECONCILIATION — NOT an exact match, and the gap is stated rather than closed:
#
#   private pipeline  738 (user_turns.csv)  721 (V2_transcript)  638 (V2_history_db)
#   this adapter      795 (windowed, scratch dirs excluded)
#
#   A +57 residual against the 738 figure. Two known causes, neither resolved
#   here because resolving them needs the private pipeline's own query text:
#     (a) CORPUS DRIFT. The private counts are frozen at 2026-09-21; the live
#         corpus measured 330 main sessions against a frozen 327, and grows
#         while the repo is worked in.
#     (b) PREDICATE SCOPE. The private M3 query selects `type in ('message',
#         'custom')` and partitions by `file`, which may or may not include
#         delegated sidecar runs the same way this adapter does.
#
#   The 721-vs-638 spread within the private pipeline itself (two legs of the
#   same V2 check, 83 apart) is larger than this adapter's gap to either, which
#   bounds how much can be read into a +57. THE LESSON: reconcile against a
#   re-run of the private query at the same instant, not against a frozen
#   figure — otherwise corpus drift and methodology differences are
#   indistinguishable from a bug.
# ---------------------------------------------------------------------------
