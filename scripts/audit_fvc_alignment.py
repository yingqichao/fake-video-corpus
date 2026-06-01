#!/usr/bin/env python3
"""Audit alignment between the FVC seed, duplicate, and query CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


REQUIRED_COLUMNS = ("cascade_id", "video_url", "label")


@dataclass(frozen=True)
class VideoRow:
    line_number: int
    cascade_id: str
    video_url: str
    label: str


@dataclass(frozen=True)
class Table:
    path: Path
    header: list[str]
    rows: list[VideoRow]
    malformed_rows: list[tuple[int, int, int]]
    missing_required: list[str]


def load_alignment_columns(path: Path) -> Table:
    """Load the first three alignment columns, while tracking malformed rows.

    FVC_text_queries.csv contains translated text with unquoted commas in some
    rows. For alignment checks we only need cascade_id, video_url, and label,
    which are the first three fields in the checked-in file.
    """

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return Table(path, [], [], [], list(REQUIRED_COLUMNS))

        missing_required = [column for column in REQUIRED_COLUMNS if column not in header]
        positions = {column: header.index(column) for column in REQUIRED_COLUMNS if column in header}
        expected_columns = len(header)
        rows: list[VideoRow] = []
        malformed_rows: list[tuple[int, int, int]] = []

        for line_number, raw_row in enumerate(reader, start=2):
            if len(raw_row) != expected_columns:
                malformed_rows.append((line_number, expected_columns, len(raw_row)))

            values = {}
            for column, position in positions.items():
                values[column] = raw_row[position].strip() if position < len(raw_row) else ""

            rows.append(
                VideoRow(
                    line_number=line_number,
                    cascade_id=values.get("cascade_id", ""),
                    video_url=values.get("video_url", ""),
                    label=values.get("label", ""),
                )
            )

    return Table(path, header, rows, malformed_rows, missing_required)


def count_duplicate_values(rows: list[VideoRow], attr: str) -> dict[str, list[int]]:
    locations: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        value = getattr(row, attr)
        if value:
            locations[value].append(row.line_number)
    return {value: lines for value, lines in locations.items() if len(lines) > 1}


def label_counts(rows: list[VideoRow]) -> Counter[str]:
    return Counter(row.label or "<blank>" for row in rows)


def platform_counts(rows: list[VideoRow]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        host = urlparse(row.video_url).netloc.lower().removeprefix("www.")
        counts[host or "<blank>"] += 1
    return counts


def rows_by_cascade(rows: list[VideoRow]) -> dict[str, VideoRow]:
    result: dict[str, VideoRow] = {}
    for row in rows:
        if row.cascade_id and row.cascade_id not in result:
            result[row.cascade_id] = row
    return result


def print_counter(title: str, counter: Counter[str]) -> None:
    print(f"{title}:")
    for key, value in sorted(counter.items()):
        print(f"  {key}: {value}")


def print_examples(title: str, values: list[str], limit: int) -> None:
    print(f"{title}: {len(values)}")
    for value in values[:limit]:
        print(f"  {value}")
    if len(values) > limit:
        print(f"  ... {len(values) - limit} more")


def summarize_table(name: str, table: Table, example_limit: int) -> int:
    print(f"\n== {name}: {table.path} ==")
    print(f"columns: {', '.join(table.header) if table.header else '<none>'}")
    print(f"rows: {len(table.rows)}")
    print(f"unique cascade_id: {len({row.cascade_id for row in table.rows if row.cascade_id})}")
    print(f"unique video_url: {len({row.video_url for row in table.rows if row.video_url})}")
    print_counter("labels", label_counts(table.rows))
    print_counter("platforms", platform_counts(table.rows))

    warning_count = 0
    if table.missing_required:
        warning_count += len(table.missing_required)
        print_examples("missing required columns", table.missing_required, example_limit)

    duplicate_urls = count_duplicate_values(table.rows, "video_url")
    if duplicate_urls:
        warning_count += len(duplicate_urls)
        print(f"duplicate video_url values: {len(duplicate_urls)}")
        for url, lines in list(duplicate_urls.items())[:example_limit]:
            print(f"  {url} on lines {', '.join(map(str, lines))}")
        if len(duplicate_urls) > example_limit:
            print(f"  ... {len(duplicate_urls) - example_limit} more")

    if table.malformed_rows:
        warning_count += len(table.malformed_rows)
        print(
            "malformed full CSV rows: "
            f"{len(table.malformed_rows)} "
            "(alignment columns were still read from the first fields)"
        )
        for line_number, expected, actual in table.malformed_rows[:example_limit]:
            print(f"  line {line_number}: expected {expected} columns, saw {actual}")
        if len(table.malformed_rows) > example_limit:
            print(f"  ... {len(table.malformed_rows) - example_limit} more")

    return warning_count


def audit(args: argparse.Namespace) -> int:
    seed = load_alignment_columns(args.fvc)
    duplicates = load_alignment_columns(args.duplicates)
    queries = load_alignment_columns(args.queries)

    warnings = 0
    warnings += summarize_table("seed videos", seed, args.examples)
    warnings += summarize_table("near duplicates", duplicates, args.examples)
    warnings += summarize_table("text queries", queries, args.examples)

    seed_ids = {row.cascade_id for row in seed.rows if row.cascade_id}
    duplicate_ids = {row.cascade_id for row in duplicates.rows if row.cascade_id}
    query_ids = {row.cascade_id for row in queries.rows if row.cascade_id}

    seed_by_id = rows_by_cascade(seed.rows)
    query_by_id = rows_by_cascade(queries.rows)

    print("\n== Alignment checks ==")
    core_issues = 0

    missing_queries = sorted(seed_ids - query_ids)
    extra_queries = sorted(query_ids - seed_ids)
    duplicate_ids_without_seed = sorted(duplicate_ids - seed_ids)
    seed_without_duplicates = sorted(seed_ids - duplicate_ids)

    print_examples("seed cascade_ids missing from FVC_text_queries.csv", missing_queries, args.examples)
    print_examples("extra FVC_text_queries.csv cascade_ids not in FVC.csv", extra_queries, args.examples)
    print_examples("FVC_dup.csv cascade_ids not in FVC.csv", duplicate_ids_without_seed, args.examples)
    print_examples("seed cascade_ids with no duplicate rows", seed_without_duplicates, args.examples)

    core_issues += len(missing_queries)
    core_issues += len(extra_queries)
    core_issues += len(duplicate_ids_without_seed)

    url_mismatches = []
    label_mismatches = []
    for cascade_id in sorted(seed_ids & query_ids):
        seed_row = seed_by_id[cascade_id]
        query_row = query_by_id[cascade_id]
        if seed_row.video_url != query_row.video_url:
            url_mismatches.append(
                f"{cascade_id}: FVC={seed_row.video_url} query={query_row.video_url}"
            )
        if seed_row.label != query_row.label:
            label_mismatches.append(
                f"{cascade_id}: FVC={seed_row.label} query={query_row.label}"
            )

    print_examples("FVC.csv vs FVC_text_queries.csv URL mismatches", url_mismatches, args.examples)
    print_examples("FVC.csv vs FVC_text_queries.csv label mismatches", label_mismatches, args.examples)

    core_issues += len(url_mismatches)
    core_issues += len(label_mismatches)

    combined_dataset_urls = {
        row.video_url
        for row in [*seed.rows, *duplicates.rows]
        if row.video_url
    }
    print("\n== Dataset interpretation ==")
    print(f"FVC.csv seed rows: {len(seed.rows)}")
    print(f"FVC_text_queries.csv rows: {len(queries.rows)}")
    print(f"FVC_dup.csv near-duplicate rows: {len(duplicates.rows)}")
    print(f"seed + duplicate rows, excluding query metadata: {len(seed.rows) + len(duplicates.rows)}")
    print(f"unique video URLs across seed + duplicates: {len(combined_dataset_urls)}")
    print(
        "FVC_text_queries.csv is query metadata for the same seed cascade_ids; "
        "it is not another duplicate-video list."
    )

    print("\n== Duplicate labels by seed prefix ==")
    labels_by_prefix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in duplicates.rows:
        prefix = row.cascade_id[:1] or "<blank>"
        labels_by_prefix[prefix][row.label or "<blank>"] += 1
    for prefix in sorted(labels_by_prefix):
        print_counter(f"cascade prefix {prefix}", labels_by_prefix[prefix])

    if core_issues:
        print(f"\nFAIL: found {core_issues} core alignment issue(s).")
        return 1

    if args.strict and warnings:
        print(f"\nFAIL: found {warnings} warning(s) and --strict was set.")
        return 1

    print("\nPASS: core cascade_id, URL, and label alignment checks passed.")
    if warnings:
        print(f"Warnings reported above: {warnings}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit alignment between FVC.csv, FVC_dup.csv, and FVC_text_queries.csv."
    )
    parser.add_argument("--fvc", type=Path, default=Path("FVC.csv"))
    parser.add_argument("--duplicates", type=Path, default=Path("FVC_dup.csv"))
    parser.add_argument("--queries", type=Path, default=Path("FVC_text_queries.csv"))
    parser.add_argument("--examples", type=int, default=10, help="Example rows/IDs to print.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero on warnings such as malformed query rows or duplicate URLs.",
    )
    return parser.parse_args()


def main() -> int:
    return audit(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
