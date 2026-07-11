from __future__ import annotations

from dataclasses import dataclass, replace
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_WATCHLIST_PATH = Path(__file__).resolve().parents[1] / "data_cache" / "watchlist.txt"
_SYMBOL_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


@dataclass(frozen=True)
class WatchlistEntry:
    """One watchlist row with optional management metadata."""

    symbol: str
    name: str = ""
    industry: str = ""
    tags: tuple[str, ...] = ()
    note: str = ""


def parse_watchlist_text(text: str) -> list[str]:
    """Parse A-share symbols from free-form watchlist text.

    Accepts newlines, commas, Chinese commas, optional names/comments, and returns
    first-seen unique 6-digit symbols in a stable order.
    """
    normalized = text.replace("，", ",").replace(";", ",").replace("；", ",")
    symbols: list[str] = []
    seen: set[str] = set()
    for raw_line in normalized.splitlines():
        line = raw_line.split("#", 1)[0]
        for symbol in _SYMBOL_RE.findall(line):
            if symbol not in seen:
                symbols.append(symbol)
                seen.add(symbol)
    return symbols


def _split_tags(value: str) -> tuple[str, ...]:
    normalized = value.replace("，", ",").replace("、", ",")
    return tuple(tag.strip() for tag in normalized.split(",") if tag.strip())


def _parse_entry_line(raw_line: str) -> WatchlistEntry | None:
    note_from_comment = ""
    line = raw_line.strip()
    if not line:
        return None
    if "#" in line:
        line, note_from_comment = line.split("#", 1)
        line = line.strip()
        note_from_comment = note_from_comment.strip()
    if not line:
        return None

    symbol_match = _SYMBOL_RE.search(line)
    if not symbol_match:
        return None
    symbol = symbol_match.group(1)
    before_meta, *meta_parts = [part.strip() for part in line.split("|")]
    name = before_meta[symbol_match.end() :].strip(" ，,;；")
    industry = ""
    tags: tuple[str, ...] = ()
    note = note_from_comment
    for part in meta_parts:
        key, separator, value = part.partition(":")
        if not separator:
            key, separator, value = part.partition("：")
        if not separator:
            continue
        key = key.strip().lower()
        value = value.strip()
        if key in {"行业", "industry", "sector"}:
            industry = value
        elif key in {"标签", "tag", "tags"}:
            tags = _split_tags(value)
        elif key in {"备注", "note", "memo"}:
            note = value
    return WatchlistEntry(symbol=symbol, name=name, industry=industry, tags=tags, note=note)


def parse_watchlist_entries(text: str) -> list[WatchlistEntry]:
    """Parse watchlist rows with optional name, industry, tags and note metadata.

    Supported row format: ``000001 平安银行 | 行业:银行 | 标签:蓝筹,低估 | 备注:核心观察``.
    A trailing ``# comment`` is treated as a note when no explicit note exists.
    Duplicate symbols keep the first occurrence to match the symbol-only parser.
    """
    entries: list[WatchlistEntry] = []
    seen: set[str] = set()
    for raw_line in text.replace(";", "\n").replace("；", "\n").splitlines():
        entry = _parse_entry_line(raw_line)
        if entry is None or entry.symbol in seen:
            continue
        entries.append(entry)
        seen.add(entry.symbol)
    return entries


def _format_watchlist_entry(entry: WatchlistEntry) -> str:
    base = entry.symbol if not entry.name else f"{entry.symbol} {entry.name}"
    parts = [base]
    if entry.industry:
        parts.append(f"行业:{entry.industry}")
    if entry.tags:
        parts.append(f"标签:{','.join(entry.tags)}")
    if entry.note:
        parts.append(f"备注:{entry.note}")
    return " | ".join(parts)


def enrich_watchlist_names(
    entries: Iterable[WatchlistEntry],
    names_by_symbol: dict[str, str],
) -> list[WatchlistEntry]:
    """Fill missing company names without overwriting user-provided names."""
    enriched: list[WatchlistEntry] = []
    for entry in entries:
        resolved_name = str(names_by_symbol.get(entry.symbol, "")).strip()
        enriched.append(replace(entry, name=entry.name or resolved_name))
    return enriched


def without_watchlist_tags(entries: Iterable[WatchlistEntry]) -> list[WatchlistEntry]:
    """Return entries without legacy tag metadata."""
    return [replace(entry, tags=()) for entry in entries]


def upsert_watchlist_entry(
    entries: Iterable[WatchlistEntry],
    new_entry: WatchlistEntry,
) -> list[WatchlistEntry]:
    """Add a stock or update its editable metadata without creating duplicates."""
    updated: list[WatchlistEntry] = []
    matched = False
    for entry in entries:
        if entry.symbol != new_entry.symbol:
            updated.append(replace(entry, tags=()))
            continue
        updated.append(
            WatchlistEntry(
                symbol=entry.symbol,
                name=new_entry.name or entry.name,
                industry=new_entry.industry or entry.industry,
                note=new_entry.note or entry.note,
            )
        )
        matched = True
    if not matched:
        updated.append(replace(new_entry, tags=()))
    return updated


def remove_watchlist_entries(
    entries: Iterable[WatchlistEntry],
    symbols: Iterable[str],
) -> list[WatchlistEntry]:
    """Remove selected symbols while preserving order and remaining metadata."""
    removed = {str(symbol).strip() for symbol in symbols}
    return [replace(entry, tags=()) for entry in entries if entry.symbol not in removed]


def format_watchlist_entries(entries: Iterable[WatchlistEntry]) -> str:
    """Serialize unique entries for both the editor and the on-disk file."""
    rows: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if entry.symbol in seen:
            continue
        rows.append(_format_watchlist_entry(entry))
        seen.add(entry.symbol)
    return "".join(f"{row}\n" for row in rows)


def save_watchlist(path: str | Path, symbols: Iterable[str]) -> Path:
    """Save a stable, deduplicated watchlist as one symbol per line."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clean_symbols = parse_watchlist_text("\n".join(str(symbol).strip() for symbol in symbols))
    destination.write_text("".join(f"{symbol}\n" for symbol in clean_symbols), encoding="utf-8")
    return destination


def save_watchlist_entries(path: str | Path, entries: Iterable[WatchlistEntry]) -> Path:
    """Save metadata-aware watchlist entries, deduplicated by first symbol."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(format_watchlist_entries(entries), encoding="utf-8")
    return destination


def load_watchlist(path: str | Path = DEFAULT_WATCHLIST_PATH) -> list[str]:
    """Load symbols from a watchlist file, returning an empty list if missing."""
    source = Path(path)
    if not source.exists():
        return []
    return parse_watchlist_text(source.read_text(encoding="utf-8"))


def load_watchlist_entries(path: str | Path = DEFAULT_WATCHLIST_PATH) -> list[WatchlistEntry]:
    """Load metadata-aware watchlist entries, returning an empty list if missing."""
    source = Path(path)
    if not source.exists():
        return []
    return parse_watchlist_entries(source.read_text(encoding="utf-8"))


def annotate_scan_candidates_with_watchlist(
    candidates: pd.DataFrame,
    entries: Iterable[WatchlistEntry],
) -> pd.DataFrame:
    """Return scan candidates enriched with watchlist name/industry/tags/note columns.

    The scanner owns market/strategy metrics while the watchlist owns human
    metadata.  Keeping the join here lets app.py display richer candidate rows
    without coupling scanner internals to watchlist parsing.
    """
    metadata_by_symbol = {
        entry.symbol: {
            "name": entry.name,
            "industry": entry.industry,
            "tags": ",".join(entry.tags),
            "note": entry.note,
        }
        for entry in entries
    }

    annotated = candidates.copy()
    original_attrs = dict(candidates.attrs)
    for column in ["name", "industry", "tags", "note"]:
        annotated[column] = annotated["symbol"].map(
            lambda symbol, column=column: metadata_by_symbol.get(str(symbol), {}).get(column, "")
        )

    ordered_columns = [
        "symbol",
        "name",
        "industry",
        "tags",
        "note",
        *[column for column in annotated.columns if column not in {"symbol", "name", "industry", "tags", "note"}],
    ]
    annotated = annotated[ordered_columns]
    annotated.attrs.update(original_attrs)
    return annotated
