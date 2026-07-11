from pathlib import Path

import pandas as pd

from ashare_quant.watchlist import (
    annotate_scan_candidates_with_watchlist,
    enrich_watchlist_names,
    format_watchlist_entries,
    load_watchlist,
    load_watchlist_entries,
    parse_watchlist_entries,
    parse_watchlist_text,
    remove_watchlist_entries,
    save_watchlist,
    save_watchlist_entries,
    upsert_watchlist_entry,
    without_watchlist_tags,
    WatchlistEntry,
)


def test_parse_watchlist_text_normalizes_separators_and_removes_duplicates():
    text = "000001 平安银行\n600519, 300750\n000001\n# 观察中的龙头\n  688981 中芯国际  "

    symbols = parse_watchlist_text(text)

    assert symbols == ["000001", "600519", "300750", "688981"]


def test_save_and_load_watchlist_round_trips_unique_symbols(tmp_path: Path):
    path = tmp_path / "watchlist.txt"

    save_watchlist(path, ["000001", "600519", "000001", " 300750 "])
    symbols = load_watchlist(path)

    assert symbols == ["000001", "600519", "300750"]
    assert path.read_text(encoding="utf-8") == "000001\n600519\n300750\n"


def test_parse_watchlist_entries_preserves_name_industry_tags_and_note():
    text = "000001 平安银行 | 行业:银行 | 标签:蓝筹,低估 | 备注:等回踩20日线\n600519 贵州茅台 # 已跟踪"

    entries = parse_watchlist_entries(text)

    assert [entry.symbol for entry in entries] == ["000001", "600519"]
    assert entries[0].name == "平安银行"
    assert entries[0].industry == "银行"
    assert entries[0].tags == ("蓝筹", "低估")
    assert entries[0].note == "等回踩20日线"
    assert entries[1].name == "贵州茅台"
    assert entries[1].note == "已跟踪"


def test_save_and_load_watchlist_entries_round_trips_metadata(tmp_path: Path):
    path = tmp_path / "watchlist_meta.txt"
    entries = parse_watchlist_entries("000001 平安银行 | 行业:银行 | 标签:蓝筹 | 备注:核心观察\n000001 重复")

    save_watchlist_entries(path, entries)
    loaded = load_watchlist_entries(path)

    assert len(loaded) == 1
    assert loaded[0].symbol == "000001"
    assert loaded[0].industry == "银行"
    assert loaded[0].tags == ("蓝筹",)
    assert loaded[0].note == "核心观察"
    assert path.read_text(encoding="utf-8") == "000001 平安银行 | 行业:银行 | 标签:蓝筹 | 备注:核心观察\n"


def test_enrich_watchlist_names_only_fills_missing_names():
    entries = parse_watchlist_entries("000001\n600519 手工名称 | 标签:核心")

    enriched = enrich_watchlist_names(entries, {"000001": "平安银行", "600519": "贵州茅台"})

    assert enriched[0].name == "平安银行"
    assert enriched[1].name == "手工名称"
    assert enriched[1].tags == ("核心",)


def test_format_watchlist_entries_keeps_code_name_and_metadata():
    entries = parse_watchlist_entries("000001 平安银行 | 行业:银行\n600519 贵州茅台")

    formatted = format_watchlist_entries(entries)

    assert formatted == "000001 平安银行 | 行业:银行\n600519 贵州茅台\n"


def test_upsert_watchlist_entry_adds_and_updates_without_duplicates_or_tags():
    entries = parse_watchlist_entries("000001 平安银行 | 标签:旧标签 | 备注:观察")

    added = upsert_watchlist_entry(entries, WatchlistEntry(symbol="600519", name="贵州茅台"))
    updated = upsert_watchlist_entry(added, WatchlistEntry(symbol="000001", industry="银行"))

    assert [item.symbol for item in updated] == ["000001", "600519"]
    assert updated[0].name == "平安银行"
    assert updated[0].industry == "银行"
    assert updated[0].note == "观察"
    assert all(not item.tags for item in updated)


def test_remove_watchlist_entries_deletes_selected_symbols_and_preserves_order():
    entries = parse_watchlist_entries("000001 平安银行\n600519 贵州茅台\n300750 宁德时代")

    remaining = remove_watchlist_entries(entries, ["600519", "300750"])

    assert [(item.symbol, item.name) for item in remaining] == [("000001", "平安银行")]


def test_without_watchlist_tags_removes_legacy_tags_only():
    entries = parse_watchlist_entries("000001 平安银行 | 标签:蓝筹 | 备注:长期观察")

    cleaned = without_watchlist_tags(entries)

    assert cleaned[0].tags == ()
    assert cleaned[0].note == "长期观察"


def test_annotate_scan_candidates_with_watchlist_adds_metadata_without_losing_attrs():
    candidates = pd.DataFrame(
        [
            {"symbol": "000001", "selection_score": 88.0},
            {"symbol": "600519", "selection_score": 80.0},
        ]
    )
    candidates.attrs["scan_summary"] = {"candidate_count": 2}
    entries = parse_watchlist_entries(
        "000001 平安银行 | 行业:银行 | 标签:蓝筹,低估 | 备注:等回踩20日线\n"
        "600519 贵州茅台 | 行业:白酒 | 标签:消费"
    )

    annotated = annotate_scan_candidates_with_watchlist(candidates, entries)

    assert annotated.attrs["scan_summary"] == {"candidate_count": 2}
    assert annotated.columns[:5].tolist() == ["symbol", "name", "industry", "tags", "note"]
    assert annotated.loc[0, "name"] == "平安银行"
    assert annotated.loc[0, "industry"] == "银行"
    assert annotated.loc[0, "tags"] == "蓝筹,低估"
    assert annotated.loc[0, "note"] == "等回踩20日线"
    assert annotated.loc[1, "note"] == ""
