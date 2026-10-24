"""Reasoning-based retrieval over the catchme activity tree.

Inspired by PageIndex: no embedding, no vector DB.  The LLM navigates the
tree hierarchy level-by-level, selecting relevant branches, reading their
summaries, and deciding when it has collected enough context to answer.

The retrieval can drill all the way down from Day → Session → App → Location
→ Action → raw events (keyboard text, mouse cluster summaries, screenshots).

Usage::

    from catchme.pipelines.retrieve import retrieve

    for step in retrieve("What did I work on in Campus-Mind yesterday?"):
        print(step)          # dict with type: browse | read | inspect | answer | error
"""

from __future__ import annotations

import glob
import json
import logging
import os
from collections.abc import Iterator
from datetime import datetime, timedelta
from datetime import time as dtime

from ..extractors.file import read_file_content
from ..extractors.url import fetch_url_content
from ..services.llm import LLM

log = logging.getLogger(__name__)

from ..config import get_default_config as _get_default_cfg


def _get_tree_dir() -> str:
    return str(_get_default_cfg().tree_dir)


def _get_blob_dir() -> str:
    return str(_get_default_cfg().blob_dir)


# ── Config & prompt loading ─────────────────────────────────────────────────


def _load_retrieve_config() -> dict:
    from ..services import load_config

    return load_config().get("retrieve", {})


def _get_prompt(key: str) -> str:
    """Load a prompt from prompts.yaml (same loader as summarize.py)."""
    from .summarize import _get_prompt as _sp

    return _sp(key, fallback="")


def _cfg(key: str, default):
    return _load_retrieve_config().get(key, default)


# ── Truncation guard ────────────────────────────────────────────────────────


def _truncate_prompt(text: str, max_chars: int | None = None) -> str:
    """Single point of truncation. Applied only at final prompt assembly."""
    if max_chars is None:
        max_chars = _cfg("max_prompt_chars", 42000)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n... (truncated due to length)"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _load_all_trees() -> list[dict]:
    """Load every *_time.json tree from disk, sorted by date."""
    if not os.path.isdir(_get_tree_dir()):
        return []
    trees: list[dict] = []
    for path in sorted(glob.glob(os.path.join(_get_tree_dir(), "*_time.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("tree"):
                trees.append(meta)
        except (json.JSONDecodeError, OSError):
            continue
    return trees


def _node_index(node: dict, out: dict | None = None) -> dict[str, dict]:
    """Build a flat node_id -> node lookup from a tree dict."""
    if out is None:
        out = {}
    nid = node.get("node_id")
    if nid:
        out[nid] = node
    for ch in node.get("children", []):
        _node_index(ch, out)
    return out


def _summarise_for_toc(node: dict) -> str:
    """Return the full summary string for display in a ToC.

    No truncation here — truncation only happens at prompt assembly via
    ``_truncate_prompt``.
    """
    s = node.get("summary", "")
    if s:
        return s.replace("\n", " ").strip()
    children = node.get("children", [])
    if children:
        parts = []
        for ch in children[:6]:
            t = ch.get("title", "")
            cs = (ch.get("summary") or "").replace("\n", " ").strip()
            parts.append(f"{t}: {cs}" if cs else t)
        return " | ".join(parts)
    return node.get("title", "(no summary)")


def _format_toc(nodes: list[dict], explored: set[str]) -> str:
    """Format a list of nodes as a numbered ToC for the LLM prompt."""
    lines: list[str] = []
    for n in nodes:
        nid = n.get("node_id", "?")
        if nid in explored:
            continue
        kind = n.get("kind", "")
        title = n.get("title", "")
        summary = _summarise_for_toc(n)
        lines.append(f"- [{nid}] ({kind}) {title}\n  {summary}")
    return "\n".join(lines) if lines else "(empty)"


def _format_details(nodes: list[dict]) -> str:
    """Format selected nodes with full summary, evidence, and children overview."""
    parts: list[str] = []
    for n in nodes:
        nid = n.get("node_id", "?")
        kind = n.get("kind", "")
        title = n.get("title", "")
        summary = n.get("summary", "(no summary)")
        evidence = n.get("evidence", "")
        children = n.get("children", [])

        section = f"### [{nid}] ({kind}) {title}\n{summary}"
        if evidence:
            section += f"\n\n**Evidence:**\n{evidence}"
        if children:
            child_lines = [
                f"  - {ch.get('title', '?')}: {_summarise_for_toc(ch)}" for ch in children[:8]
            ]
            section += "\n  Children:\n" + "\n".join(child_lines)
        parts.append(section)
    return "\n\n".join(parts)


def _format_collected(collected: list[dict]) -> str:
    if not collected:
        return "(nothing yet)"
    parts = [f"- [{c['node_id']}] {c['extract']}" for c in collected]
    return "\n".join(parts)


def _toc_entry(node: dict) -> dict:
    """Minimal dict for frontend display in a browse step."""
    summary = _summarise_for_toc(node)
    return {
        "node_id": node.get("node_id", ""),
        "kind": node.get("kind", ""),
        "title": node.get("title", ""),
        "summary_preview": summary[:200] + ("..." if len(summary) > 200 else ""),
    }


def _llm_json(
    llm: LLM, prompt: str, temperature: float = 0.3, max_tokens: int | None = None
) -> dict:
    """Call LLM expecting a JSON response, with robust parsing."""
    prompt = _truncate_prompt(prompt)
    raw = llm.complete(
        [{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("LLM returned non-JSON: %s", raw[:200])
        return {"reasoning": raw, "action": "answer", "selected": [], "useful": []}


# ── Time-aware pre-filtering ───────────────────────────────────────────────


def _resolve_time_range(query: str, llm: LLM) -> dict | None:
    """Call LLM to extract a time range from the user query.

    Returns a dict with keys ``has_time``, ``reasoning``, ``dates``,
    ``start_hour``, ``end_hour`` — or *None* when no time info is found.
    """
    cfg_rc = _load_retrieve_config()
    temp = cfg_rc.get("temperature_time_resolve", 0.1)
    max_tok = cfg_rc.get("max_tokens_time_resolve", 300)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M %A")
    prompt = _get_prompt("retrieve_time_resolve").format(now=now_str, query=query)
    result = _llm_json(llm, prompt, temperature=temp, max_tokens=max_tok)

    if not result.get("has_time"):
        return None

    # Normalise LLM output: ensure dates is a list and hours are int|None.
    dates = result.get("dates")
    if isinstance(dates, str):
        dates = [dates]
    result["dates"] = dates

    for key in ("start_hour", "end_hour"):
        val = result.get(key)
        if val is None:
            continue
        if isinstance(val, bool):
            result[key] = None
            continue
        try:
            hour = int(val)
        except (OverflowError, TypeError, ValueError):
            result[key] = None
            continue

        max_hour = 23 if key == "start_hour" else 24
        result[key] = hour if 0 <= hour <= max_hour else None

    return result


def _filter_trees_by_dates(trees: list[dict], dates: list[str]) -> list[dict]:
    """Keep only trees whose ``date`` field is in *dates*."""
    date_set = set(dates)
    return [t for t in trees if t.get("date") in date_set]


def _sessions_in_range(
    day_node: dict,
    start_hour: int | None,
    end_hour: int | None,
) -> list[dict]:
    """Return session children whose time span overlaps ``[start_hour, end_hour)``.

    Handles cross-midnight windows (e.g. 22–06) by splitting into two
    sub-windows: ``[start_hour, 24)`` and ``[0, end_hour)``.
    """
    sessions = day_node.get("children", [])
    if start_hour is None and end_hour is None:
        return sessions

    sh = start_hour if start_hour is not None else 0
    eh = end_hour if end_hour is not None else 24

    if sh == eh:
        return []

    result: list[dict] = []
    for s in sessions:
        s_start = datetime.fromtimestamp(s["start"])
        s_end = datetime.fromtimestamp(s["end"])
        day_date = s_start.date()

        if sh < eh:
            windows = [
                (
                    datetime.combine(day_date, dtime(sh, 0)),
                    datetime.combine(day_date, dtime(eh, 0))
                    if eh < 24
                    else datetime.combine(day_date + timedelta(days=1), dtime(0, 0)),
                )
            ]
        else:
            # Cross-midnight: split into [sh, 24:00) and [00:00, eh)
            windows = [
                (
                    datetime.combine(day_date, dtime(sh, 0)),
                    datetime.combine(day_date + timedelta(days=1), dtime(0, 0)),
                ),
                (datetime.combine(day_date, dtime(0, 0)), datetime.combine(day_date, dtime(eh, 0))),
            ]

        if any(s_start < w_end and s_end > w_start for w_start, w_end in windows):
            result.append(s)
    return result


# ── Action-level deep dive ─────────────────────────────────────────────────


def _resolve_blob(relative_path: str) -> str | None:
    """Convert a relative blob path to an absolute filesystem path."""
    full = os.path.join(_get_blob_dir(), relative_path)
    return full if os.path.isfile(full) else None


def _expand_action_context(node: dict) -> list[dict]:
    """Expand an action node's context into virtual sub-items.

    Returns a list of dicts that look like tree nodes (with node_id, kind,
    title, summary, etc.) but are synthesized from the action's raw data.
    They are stored in the node_idx so the main loop can find them.
    """
    ctx = node.get("context", {})
    nid = node.get("node_id", "")
    items: list[dict] = []

    text = ctx.get("text", "")
    if text:
        items.append(
            {
                "node_id": f"{nid}::kb",
                "kind": "raw_keyboard",
                "title": "Keyboard input",
                "summary": f"Raw keystroke stream ({len(text)} chars): {text[:500]}{'...' if len(text) > 500 else ''}",
                "_raw_text": text,
            }
        )

    mouse_summaries = ctx.get("mouse_summaries", [])
    mouse_actions = ctx.get("mouse_actions", [])
    for i, ms in enumerate(mouse_summaries):
        screenshots = _find_screenshots_for_cluster(
            ms.get("start", 0), ms.get("end", 0), mouse_actions
        )
        items.append(
            {
                "node_id": f"{nid}::mouse_{i}",
                "kind": "raw_mouse",
                "title": f"Mouse cluster {i} ({_fmt_ts(ms.get('start', 0))} – {_fmt_ts(ms.get('end', 0))})",
                "summary": ms.get("summary", ""),
                "_screenshots": screenshots,
            }
        )

    items.sort(key=lambda x: x.get("_sort_ts", x.get("node_id", "")))
    return items


def _find_screenshots_for_cluster(
    start: float, end: float, mouse_actions: list[dict]
) -> list[dict]:
    """Find mouse_actions (with screenshot paths) that fall within a time range."""
    results = []
    for ma in mouse_actions:
        ts = ma.get("ts", 0)
        if start <= ts <= end and ma.get("full"):
            results.append(
                {
                    "ts": ts,
                    "action": ma.get("action", ""),
                    "full": ma.get("full", ""),
                    "detail": ma.get("detail", ""),
                }
            )
    return results


def _expand_location_sources(node: dict) -> list[dict]:
    """Create virtual explorable nodes for files/URLs referenced by a location.

    If the location's full_location is a readable file path or a fetchable URL,
    produce a virtual node the LLM can select to read the actual content.
    """
    ctx = node.get("context", {})
    nid = node.get("node_id", "")
    full_loc = ctx.get("full_location", "")
    items: list[dict] = []

    if _is_file_path(full_loc) and os.path.isfile(full_loc):
        items.append(
            {
                "node_id": f"{nid}::file",
                "kind": "raw_file",
                "title": f"Read file: {os.path.basename(full_loc)}",
                "summary": f"Read the actual content of {full_loc}",
                "_file_path": full_loc,
            }
        )
    elif _is_url(full_loc):
        items.append(
            {
                "node_id": f"{nid}::url",
                "kind": "raw_url",
                "title": f"Fetch URL: {full_loc[:80]}",
                "summary": f"Fetch and read the web page at {full_loc}",
                "_url": full_loc,
            }
        )

    return items


def _is_file_path(s: str) -> bool:
    """Heuristic: looks like an absolute or project-relative file path."""
    if not s:
        return False
    if s.startswith("/") or s.startswith("~"):
        return True
    return len(s) > 2 and s[1] == ":" and s[2] in ("/", "\\")


def _is_url(s: str) -> bool:
    """Heuristic: looks like an HTTP(S) URL."""
    return s.startswith("http://") or s.startswith("https://")


def _get_workspace_dir() -> str:
    return str(_get_default_cfg().workspace_dir)


def _fmt_ts(ts: float) -> str:
    """Format a unix timestamp as HH:MM:SS."""
    import time

    try:
        return time.strftime("%H:%M:%S", time.localtime(ts))
    except (OSError, ValueError):
        return str(ts)


# ── Main retrieval generator ───────────────────────────────────────────────


def retrieve(query: str) -> Iterator[dict]:
    """Yield step dicts as the LLM navigates the activity tree.

    Step types:
      - browse:  LLM is scanning a level's ToC and selecting nodes
      - read:    LLM is evaluating selected nodes' content
      - inspect: LLM is examining a screenshot via vision
      - answer:  final answer generated from collected context
      - error:   something went wrong
    """
    llm = LLM()
    trees = _load_all_trees()

    if not trees:
        yield {"type": "error", "message": "No activity trees found. Record some activity first."}
        return

    # -- Time-aware pre-filtering ----------------------------------------------
    time_range = _resolve_time_range(query, llm)
    has_dates = time_range and time_range.get("dates")
    has_hours = time_range and (
        time_range.get("start_hour") is not None or time_range.get("end_hour") is not None
    )

    if has_dates:
        trees = _filter_trees_by_dates(trees, time_range["dates"])

    if time_range:
        yield {
            "type": "time_filter",
            "dates": time_range.get("dates"),
            "start_hour": time_range.get("start_hour"),
            "end_hour": time_range.get("end_hour"),
            "reasoning": time_range.get("reasoning", ""),
        }

    if not trees:
        yield {"type": "error", "message": "No activity trees found for the specified time range."}
        return

    node_idx: dict[str, dict] = {}
    for t in trees:
        _node_index(t["tree"], node_idx)

    # -- Level 0: Day selection ------------------------------------------------
    day_nodes = [t["tree"] for t in trees]

    collected: list[dict] = []
    explored: set[str] = set()

    cfg_rc = _load_retrieve_config()
    temp_select = cfg_rc.get("temperature_select", 0.3)
    temp_answer = cfg_rc.get("temperature_answer", 0.7)
    max_iters = cfg_rc.get("max_iterations", 15)
    max_file = cfg_rc.get("max_file_chars", 8000)
    max_nodes = cfg_rc.get("max_select_nodes", 5)
    max_tok_step = cfg_rc.get("max_tokens_step", 800)
    max_tok_answer = cfg_rc.get("max_tokens_answer", 2000)

    if has_dates:
        # Dates already resolved — skip the Day-selection LLM call
        selected_nodes = day_nodes
        explored.update(n.get("node_id", "") for n in selected_nodes)

        yield {
            "type": "browse",
            "level": "day",
            "candidates": [_toc_entry(n) for n in day_nodes],
            "selected": [_toc_entry(n) for n in selected_nodes],
            "reasoning": "Auto-selected by time filter",
        }
    else:
        # Original LLM-based day selection
        toc_text = _format_toc(day_nodes, explored)
        prompt = _get_prompt("retrieve_select").format(
            query=query,
            collected=_format_collected(collected),
            prev_action="initial",
            toc=toc_text,
            max_nodes=max_nodes,
        )

        result = _llm_json(llm, prompt, temperature=temp_select, max_tokens=max_tok_step)

        selected_ids = result.get("selected", [])
        selected_nodes = [node_idx[nid] for nid in selected_ids if nid in node_idx]
        explored.update(selected_ids)

        yield {
            "type": "browse",
            "level": "day",
            "candidates": [_toc_entry(n) for n in day_nodes],
            "selected": [_toc_entry(n) for n in selected_nodes],
            "reasoning": result.get("reasoning", ""),
        }

        if result.get("action") == "sufficient" or not selected_nodes:
            if collected:
                yield from _generate_answer(llm, query, collected, temp_answer, max_tok_answer)
            else:
                yield {
                    "type": "answer",
                    "content": result.get("reasoning", "No relevant information found."),
                }
            return

    # -- Expand to session level and enter main loop ---------------------------
    frontier: list[dict] = []
    for day in selected_nodes:
        if has_hours:
            frontier.extend(
                _sessions_in_range(
                    day,
                    time_range.get("start_hour"),
                    time_range.get("end_hour"),
                )
            )
        else:
            frontier.extend(day.get("children", []))

    if not frontier and has_hours:
        log.info(
            "Hour filter returned no sessions; falling back to all sessions for selected days."
        )
        for day in selected_nodes:
            frontier.extend(day.get("children", []))

    prev_action = "initial"

    for _iteration in range(max_iters):
        available = [n for n in frontier if n.get("node_id") not in explored]
        if not available:
            break

        # -- Step A: LLM Select ------------------------------------------------
        toc_text = _format_toc(available, explored)
        prompt = _get_prompt("retrieve_select").format(
            query=query,
            collected=_format_collected(collected),
            prev_action=prev_action,
            toc=toc_text,
            max_nodes=max_nodes,
        )
        sel_result = _llm_json(llm, prompt, temperature=temp_select, max_tokens=max_tok_step)

        # When the previous evaluate explicitly asked for "deeper", do NOT
        # allow the select step to bail out with "sufficient".
        if sel_result.get("action") == "sufficient" and prev_action != "deeper":
            yield {
                "type": "browse",
                "level": available[0].get("kind", "node") if available else "node",
                "candidates": [_toc_entry(n) for n in available],
                "selected": [],
                "reasoning": sel_result.get("reasoning", ""),
            }
            break

        sel_ids = sel_result.get("selected", [])
        sel_nodes = [node_idx[nid] for nid in sel_ids if nid in node_idx]

        # Fallback: if prev evaluate asked "deeper" but LLM still selected
        # nothing, auto-pick the first few available candidates.
        if not sel_nodes and prev_action == "deeper":
            sel_nodes = available[:max_nodes]
            sel_ids = [n.get("node_id", "") for n in sel_nodes]

        explored.update(sel_ids)

        level_kind = sel_nodes[0].get("kind", "node") if sel_nodes else "node"
        yield {
            "type": "browse",
            "level": level_kind,
            "candidates": [_toc_entry(n) for n in available],
            "selected": [_toc_entry(n) for n in sel_nodes],
            "reasoning": sel_result.get("reasoning", ""),
        }

        if not sel_nodes:
            break

        # -- Step B: LLM Evaluate ----------------------------------------------
        details_text = _format_details(sel_nodes)
        prompt = _get_prompt("retrieve_evaluate").format(
            query=query,
            collected=_format_collected(collected),
            prev_action=prev_action,
            details=details_text,
        )
        eval_result = _llm_json(llm, prompt, temperature=temp_select, max_tokens=max_tok_step)

        useful = eval_result.get("useful", [])
        action = eval_result.get("action", "answer")

        for u in useful:
            if u.get("node_id") and u.get("extract"):
                collected.append(u)

        yield {
            "type": "read",
            "nodes": [
                {
                    "node_id": n.get("node_id", ""),
                    "title": n.get("title", ""),
                    "useful": any(u.get("node_id") == n.get("node_id") for u in useful),
                    "extract": next(
                        (u["extract"] for u in useful if u.get("node_id") == n.get("node_id")),
                        None,
                    ),
                }
                for n in sel_nodes
            ],
            "collected_count": len(collected),
            "action": action,
            "reasoning": eval_result.get("reasoning", ""),
        }

        prev_action = action

        if action == "answer":
            break
        elif action == "deeper":
            frontier = []
            useful_ids = {u["node_id"] for u in useful if u.get("node_id")}
            for n in sel_nodes:
                if n.get("node_id") not in useful_ids:
                    continue
                kind = n.get("kind", "")
                if kind == "action":
                    expanded = _expand_action_context(n)
                    for vn in expanded:
                        node_idx[vn["node_id"]] = vn
                    frontier.extend(expanded)
                elif kind in ("raw_keyboard", "raw_mouse", "raw_file", "raw_url"):
                    yield from _inspect_raw_node(llm, query, n, collected, max_file_chars=max_file)
                elif kind == "location":
                    children = n.get("children", [])
                    frontier.extend(children)
                    # Directly read file / fetch URL instead of creating
                    # virtual nodes that require another LLM round-trip.
                    nid = n.get("node_id", "")
                    ctx = n.get("context", {})
                    full_loc = ctx.get("full_location", "")
                    if _is_file_path(full_loc) and os.path.isfile(full_loc):
                        yield from _inspect_raw_node(
                            llm,
                            query,
                            {
                                "node_id": f"{nid}::file",
                                "kind": "raw_file",
                                "title": f"File: {os.path.basename(full_loc)}",
                                "_file_path": full_loc,
                            },
                            collected,
                            max_file_chars=max_file,
                        )
                    elif _is_url(full_loc):
                        yield from _inspect_raw_node(
                            llm,
                            query,
                            {
                                "node_id": f"{nid}::url",
                                "kind": "raw_url",
                                "title": f"URL: {full_loc[:80]}",
                                "_url": full_loc,
                            },
                            collected,
                            max_file_chars=max_file,
                        )
                else:
                    frontier.extend(n.get("children", []))
            if not frontier:
                break
        elif action == "siblings":
            pass

    # -- Generate answer -------------------------------------------------------
    yield from _generate_answer(llm, query, collected, temp_answer, max_tok_answer)


# ── Screenshot inspection ──────────────────────────────────────────────────


def _inspect_raw_node(
    llm: LLM,
    query: str,
    node: dict,
    collected: list[dict],
    max_file_chars: int = 8000,
) -> Iterator[dict]:
    """Inspect a raw_keyboard, raw_mouse, raw_file, or raw_url virtual node."""
    nid = node.get("node_id", "")
    kind = node.get("kind", "")

    if kind == "raw_keyboard":
        raw_text = node.get("_raw_text", node.get("summary", ""))
        collected.append(
            {
                "node_id": nid,
                "extract": f"Keyboard input: {raw_text}",
            }
        )
        yield {
            "type": "read",
            "nodes": [
                {
                    "node_id": nid,
                    "title": "Keyboard input",
                    "useful": True,
                    "extract": raw_text[:300],
                }
            ],
            "collected_count": len(collected),
            "action": "continue",
            "reasoning": "Loaded raw keyboard input.",
        }
        return

    if kind == "raw_file":
        filepath = node.get("_file_path", "")
        try:
            content, file_type = read_file_content(filepath, max_chars=max_file_chars)
        except Exception as e:
            log.warning("File read failed for %s: %s", filepath, e)
            content, file_type = "", "error"
        if content:
            collected.append(
                {
                    "node_id": nid,
                    "extract": f"File content ({file_type}) of {os.path.basename(filepath)}:\n{content}",
                }
            )
            yield {
                "type": "read",
                "nodes": [
                    {
                        "node_id": nid,
                        "title": f"File: {os.path.basename(filepath)}",
                        "useful": True,
                        "extract": content[:300] + "...",
                    }
                ],
                "collected_count": len(collected),
                "action": "continue",
                "reasoning": f"Read {file_type} file: {filepath}",
            }
        else:
            yield {
                "type": "read",
                "nodes": [
                    {
                        "node_id": nid,
                        "title": f"File: {os.path.basename(filepath)}",
                        "useful": False,
                        "extract": None,
                    }
                ],
                "collected_count": len(collected),
                "action": "continue",
                "reasoning": f"Could not read file: {filepath}",
            }
        return

    if kind == "raw_url":
        url = node.get("_url", "")
        try:
            content = fetch_url_content(
                url, max_chars=max_file_chars, workspace_dir=_get_workspace_dir()
            )
        except Exception as e:
            log.warning("URL fetch failed for %s: %s", url, e)
            content = ""
        if content:
            collected.append(
                {
                    "node_id": nid,
                    "extract": f"Web page content of {url}:\n{content}",
                }
            )
            yield {
                "type": "read",
                "nodes": [
                    {
                        "node_id": nid,
                        "title": f"URL: {url[:60]}",
                        "useful": True,
                        "extract": content[:300] + "...",
                    }
                ],
                "collected_count": len(collected),
                "action": "continue",
                "reasoning": f"Fetched web page: {url}",
            }
        else:
            yield {
                "type": "read",
                "nodes": [
                    {"node_id": nid, "title": f"URL: {url[:60]}", "useful": False, "extract": None}
                ],
                "collected_count": len(collected),
                "action": "continue",
                "reasoning": f"Could not fetch URL: {url}",
            }
        return

    if kind == "raw_mouse":
        screenshots = node.get("_screenshots", [])
        if not screenshots:
            return

        first_ss = screenshots[0]
        full_path = _resolve_blob(first_ss.get("full", ""))
        if not full_path:
            return

        context_text = f"Mouse cluster summary: {node.get('summary', '')[:500]}"
        prompt = _get_prompt("retrieve_inspect_image").format(query=query, context=context_text)
        prompt = _truncate_prompt(prompt)

        try:
            raw = llm.complete_with_vision(
                prompt=prompt,
                image_paths=[full_path],
                detail="auto",
                temperature=0.3,
            )
        except Exception as e:
            log.warning("Vision call failed: %s", e)
            return

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
        try:
            vresult = json.loads(raw)
        except json.JSONDecodeError:
            vresult = {"reasoning": raw, "useful": True, "extract": raw[:500]}

        is_useful = vresult.get("useful", False)
        extract = vresult.get("extract", "")
        if is_useful and extract:
            collected.append({"node_id": nid, "extract": extract})

        yield {
            "type": "inspect",
            "node_id": nid,
            "title": node.get("title", "Screenshot"),
            "image_url": f"/blobs/{first_ss.get('full', '')}",
            "useful": is_useful,
            "extract": extract,
            "reasoning": vresult.get("reasoning", ""),
            "collected_count": len(collected),
            "has_detail": bool(first_ss.get("detail")),
        }

        if is_useful and first_ss.get("detail"):
            detail_path = _resolve_blob(first_ss["detail"])
            if detail_path:
                yield from _inspect_detail(
                    llm, query, nid, first_ss["detail"], detail_path, collected
                )


def _inspect_detail(
    llm: LLM,
    query: str,
    parent_nid: str,
    detail_rel: str,
    detail_path: str,
    collected: list[dict],
) -> Iterator[dict]:
    """Inspect a detail crop screenshot for even more precision."""
    prompt = _get_prompt("retrieve_inspect_image").format(
        query=query,
        context="This is a cropped detail view of a region of interest from a full screenshot.",
    )
    prompt = _truncate_prompt(prompt)

    try:
        raw = llm.complete_with_vision(
            prompt=prompt,
            image_paths=[detail_path],
            detail="high",
            temperature=0.3,
        )
    except Exception as e:
        log.warning("Detail vision call failed: %s", e)
        return

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]
    try:
        vresult = json.loads(raw)
    except json.JSONDecodeError:
        vresult = {"reasoning": raw, "useful": True, "extract": raw[:500]}

    detail_nid = f"{parent_nid}::detail"
    extract = vresult.get("extract", "")
    if vresult.get("useful", False) and extract:
        collected.append({"node_id": detail_nid, "extract": extract})

    yield {
        "type": "inspect",
        "node_id": detail_nid,
        "title": "Detail crop",
        "image_url": f"/blobs/{detail_rel}",
        "useful": vresult.get("useful", False),
        "extract": extract,
        "reasoning": vresult.get("reasoning", ""),
        "collected_count": len(collected),
        "has_detail": False,
    }


# ── Answer generation ──────────────────────────────────────────────────────


def _generate_answer(
    llm: LLM,
    query: str,
    collected: list[dict],
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> Iterator[dict]:
    """Produce the final answer step."""
    if not collected:
        yield {
            "type": "answer",
            "content": "I couldn't find relevant information in your activity history to answer this question.",
        }
        return
    prompt = _get_prompt("retrieve_answer").format(
        query=query,
        collected=_format_collected(collected),
    )
    prompt = _truncate_prompt(prompt)
    answer = llm.complete(
        [{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    sources = list({c["node_id"] for c in collected})
    yield {"type": "answer", "content": answer, "sources": sources}
