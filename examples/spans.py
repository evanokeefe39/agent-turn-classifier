"""Group a session transcript into spans of agent work.

A *span* is the unit this tool labels: everything the agent did between two of
the user's messages — including spans the user opened by interjecting or
steering mid-run. The user's own message is context for a span, not evidence
for its label; the evidence is the agent's actions and its own one-line
descriptions of them.

Why not label user turns: a user turn records what was *asked for*. The
question this tool answers is what the *agent worked on*. Those are different
populations in the same transcript, and sampling the wrong one produces a
coherent dataset that answers the wrong question.

Transcript shape (validated against real OMP sessions — see
`examples/spike_omp_adapter.py` for the field-level evidence):

  * fields are nested: ``role`` is at ``message.role``, not top-level;
    tool records carry ``customType`` (not ``ctype``) with the payload at
    ``data.toolName`` / ``data.intent`` / ``data.args``.
  * a span boundary is a row where ``type == 'message'`` and
    ``message.role == 'user'``.
  * the walk is restricted to ``type in ('message', 'custom')``, mirroring the
    private pipeline's window. Advancing over ``title``/``session``/
    ``compaction`` rows mis-assigns predecessors and inflates the kind counts.

No personal session text lives in this module — it takes a path and returns
structures. Callers decide what to do with the content.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# A bare acknowledgement carries no work and no intent to judge. "address
# advisory" is deliberately NOT in here: the user treats it as a real steer.
_ACK = re.compile(
    r"^(\.|\.\.\.|keep going|continue|ok|okay|thanks?|yes|no|y|n|k|carry on)\.?$",
    re.IGNORECASE,
)
_STEER = re.compile(r"address advis|advisory", re.IGNORECASE)

# Interruption signatures. These spans record the agent dying (rate limit,
# exhausted credit, dropped connection) rather than doing work. Treating them
# as work would inject a failure mode into the label distribution.
_INTERRUPT = re.compile(
    r"\b(429|rate limit|rate-limit|credit|out of credits|quota|exhaust|"
    r"connection (error|reset)|econnreset|timed? ?out|socket hang up|500|503)\b",
    re.IGNORECASE,
)

# Below this many characters a reply is not substantive enough to label.
MIN_REPLY_CHARS = 120


def load_rows(path: str | Path) -> list[dict]:
    """Read a transcript, keeping only the rows a span walk may traverse."""
    out: list[dict] = []
    for line in Path(path).read_bytes().split(b"\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if row.get("type") in ("message", "custom"):
            out.append(row)
    return out


def _text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in (content or [])
        if isinstance(block, dict) and block.get("type") == "text"
    )


def spans_of(rows: list[dict]) -> list[dict]:
    """Split rows into spans.

    Returns one dict per span with the fields a labeller needs:

        user            the instruction that opened the span (context only)
        ts              timestamp of that instruction
        tools           [{name, intent, args}] — the agent's actions
        texts, thinking counts of assistant text/thinking blocks
        reply           the agent's reply text, concatenated
        thinking_text   the first thinking block, for spans that only reason
        errors, aborted counts of stopReason error/aborted
        err_text        truncated error text, used for interruption detection
    """
    spans: list[dict] = []
    current: dict | None = None
    for row in rows:
        if row.get("type") == "message" and (row.get("message") or {}).get("role") == "user":
            if current is not None:
                spans.append(current)
            current = {
                "user": _text_of(row["message"]).strip(),
                "ts": row.get("timestamp"),
                "tools": [],
                "texts": 0,
                "thinking": 0,
                "reply": "",
                "thinking_text": "",
                "errors": 0,
                "aborted": 0,
                "err_text": [],
            }
            continue
        if current is None:
            continue
        if row.get("type") == "custom" and row.get("customType") == "tool_execution_start":
            data = row.get("data") or {}
            current["tools"].append(
                {
                    "name": data.get("toolName"),
                    "intent": (data.get("intent") or "").strip(),
                    "args": data.get("args") or {},
                }
            )
        elif row.get("type") == "message":
            message = row.get("message") or {}
            if message.get("role") != "assistant":
                continue
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    current["texts"] += 1
                    text = block.get("text", "")
                    if text:
                        current["reply"] += ("\n" if current["reply"] else "") + text
                elif block.get("type") == "thinking":
                    current["thinking"] += 1
                    text = block.get("thinking", "")
                    if text and not current["thinking_text"]:
                        current["thinking_text"] = text
            stop = message.get("stopReason")
            if stop == "error":
                current["errors"] += 1
                text = _text_of(message)
                if text:
                    current["err_text"].append(text[:300])
            elif stop == "aborted":
                current["aborted"] += 1
    if current is not None:
        spans.append(current)
    return spans


def classify_span(span: dict) -> tuple[bool, str]:
    """Is this span labelable work, and why?

    Returns ``(is_work, reason)`` so every exclusion is auditable rather than
    silent — an exclusion rule that under-matches produces a plausible
    distribution over the wrong population.
    """
    if _ACK.match(span["user"]) and not _STEER.search(span["user"]):
        return False, "acknowledgement"

    blob = span["user"] + " " + " ".join(span["err_text"])
    has_tools = bool(span["tools"])
    produced = span["texts"] or span["thinking"]

    if not has_tools and _INTERRUPT.search(blob):
        return False, "interruption"
    if has_tools and span["aborted"] and not produced:
        return False, "interruption"
    if has_tools:
        return True, "actions"
    if span["reply"] and len(span["reply"]) >= MIN_REPLY_CHARS:
        return True, "reply_only"
    if span["thinking"] and span["thinking_text"]:
        return True, "reasoning_only"
    return False, "empty"


def span_view(span: dict, max_intents: int = 12) -> str:
    """Render the agent-side evidence for a span, for a labeller to read.

    Deliberately agent-first: the instruction is labelled as context so a
    reader cannot mistake it for the evidence.
    """
    lines = [f"INSTRUCTION (context only): {span['user'][:300]}"]
    if span["tools"]:
        from collections import Counter

        counts = Counter(t["name"] for t in span["tools"])
        lines.append(
            f"ACTIONS: {len(span['tools'])} tool calls — "
            + ", ".join(f"{name}x{n}" for name, n in counts.most_common())
        )
        intents = [t["intent"] for t in span["tools"] if t["intent"]]
        if intents:
            shown = intents[:max_intents]
            lines.append("INTENTS: " + " | ".join(shown))
            if len(intents) > len(shown):
                lines.append(f"  …and {len(intents) - len(shown)} more")
    if span["reply"]:
        lines.append(f"REPLY ({len(span['reply'])} chars): {span['reply'][:600]}")
    if span["errors"]:
        lines.append(f"ERRORS: {span['errors']}")
    if span["aborted"]:
        lines.append(f"ABORTED: {span['aborted']}")
    return "\n".join(lines)
