#!/usr/bin/env python3
"""Download YouTube videos listed in FVC.csv into ./FVC."""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse


@dataclass(frozen=True)
class VideoRow:
    cascade_id: str
    url: str
    label: str


def slug(value: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = value.strip("._-")
    return value or fallback


def parse_cookies_from_browser(value: str) -> tuple[str, str | None, str | None, str | None]:
    match = re.fullmatch(
        r"""(?x)
        (?P<name>[^+:]+)
        (?:\s*\+\s*(?P<keyring>[^:]+))?
        (?:\s*:\s*(?!:)(?P<profile>.+?))?
        (?:\s*::\s*(?P<container>.+))?
        """,
        value,
    )
    if not match:
        raise ValueError(f"invalid --cookies-from-browser value: {value}")

    browser_name, keyring, profile, container = match.group(
        "name", "keyring", "profile", "container"
    )
    return browser_name.lower(), profile, keyring.upper() if keyring else None, container


def youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", maxsplit=1)[0]
        return video_id or None

    if host.endswith("youtube.com"):
        query_ids = parse_qs(parsed.query).get("v")
        if query_ids:
            return query_ids[0]

        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"embed", "shorts", "v"}:
            return parts[1]

    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None


def is_youtube_url(url: str) -> bool:
    return youtube_video_id(url) is not None


def load_csv(csv_path: Path) -> list[VideoRow]:
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if "video_url" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain a 'video_url' column")

        rows: list[VideoRow] = []
        for line_number, row in enumerate(reader, start=2):
            url = (row.get("video_url") or "").strip()
            if not url:
                continue
            rows.append(
                VideoRow(
                    cascade_id=slug(row.get("cascade_id") or "", f"row{line_number}"),
                    url=url,
                    label=slug(row.get("label") or "", "unknown"),
                )
            )
    return rows


def group_rows_by_cascade(rows: Iterable[VideoRow]) -> dict[str, list[VideoRow]]:
    grouped: dict[str, list[VideoRow]] = {}
    for row in rows:
        grouped.setdefault(row.cascade_id, []).append(row)
    return grouped


def select_rows(
    rows: Iterable[VideoRow],
    urls: list[str],
    cascade_ids: list[str],
    limit: int | None,
) -> list[VideoRow]:
    selected = list(rows)

    if urls:
        by_url = {row.url: row for row in selected}
        selected = [
            by_url.get(url, VideoRow(cascade_id=f"manual_{index}", url=url, label="manual"))
            for index, url in enumerate(urls, start=1)
        ]

    if cascade_ids:
        wanted = set(cascade_ids)
        selected = [row for row in selected if row.cascade_id in wanted]

    if limit is not None:
        selected = selected[:limit]

    return selected


def load_download_archive(archive_path: Path) -> set[str]:
    if not archive_path.exists():
        return set()

    video_ids: set[str] = set()
    with archive_path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2:
                video_ids.add(parts[-1])
    return video_ids


def find_downloaded_file(row: VideoRow, output_dir: Path) -> Path | None:
    label_dir = output_dir / row.label
    if not label_dir.exists():
        return None

    video_id = youtube_video_id(row.url)
    patterns = [f"{row.cascade_id}.*"]
    if video_id:
        patterns.append(f"{row.cascade_id}__{video_id}__*")
    patterns.append(f"{row.cascade_id}__*")

    for pattern in patterns:
        for path in sorted(label_dir.glob(pattern)):
            if path.is_file() and not path.name.endswith((".part", ".ytdl")):
                return path
    return None


def already_downloaded(
    row: VideoRow,
    output_dir: Path,
    archive_ids: set[str],
) -> str | None:
    downloaded_file = find_downloaded_file(row, output_dir)
    if downloaded_file:
        return str(downloaded_file)

    video_id = youtube_video_id(row.url)
    if video_id and video_id in archive_ids:
        return "download_archive"

    return None


def fallback_candidates(
    row: VideoRow,
    fallback_rows: dict[str, list[VideoRow]],
    args: argparse.Namespace,
) -> list[VideoRow]:
    candidates = [row]
    if not fallback_rows:
        return candidates

    seen_urls = {row.url}
    fallback_count = 0
    for fallback in fallback_rows.get(row.cascade_id, []):
        if fallback.url in seen_urls:
            continue
        if not args.fallback_any_duplicate_label and fallback.label != row.label:
            continue
        if args.fallback_platform == "youtube" and not is_youtube_url(fallback.url):
            continue

        seen_urls.add(fallback.url)
        fallback_count += 1
        candidates.append(
            VideoRow(
                cascade_id=row.cascade_id,
                url=fallback.url,
                label=row.label,
            )
        )
        if args.max_fallbacks is not None and fallback_count >= args.max_fallbacks:
            break

    return candidates


def tqdm_iter(rows: list[VideoRow], enabled: bool):
    if not enabled:
        return rows

    try:
        from tqdm import tqdm
    except ImportError:
        sys.exit(
            "Missing dependency: tqdm\n"
            "Install dependencies with: python -m pip install -r requirements.txt"
        )

    return tqdm(rows, desc="FVC downloads", unit="video")


def download_one(row: VideoRow, output_dir: Path, args: argparse.Namespace) -> int:
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        sys.exit(
            "Missing dependency: yt-dlp\n"
            "Install it with: python -m pip install -r requirements.txt"
        )

    label_dir = output_dir / row.label
    label_dir.mkdir(parents=True, exist_ok=True)

    outtmpl = str(label_dir / f"{row.cascade_id}.%(ext)s")
    ydl_opts = {
        "format": args.format,
        "outtmpl": outtmpl,
        "continuedl": True,
        "ignoreerrors": False,
        "noplaylist": True,
        "restrictfilenames": args.restrict_filenames,
        "quiet": args.quiet or not args.no_progress,
        "no_warnings": args.quiet,
        "noprogress": True,
    }

    if args.download_archive and not args.no_skip_existing:
        archive_path = Path(args.download_archive)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        ydl_opts["download_archive"] = str(archive_path)

    if args.cookies:
        ydl_opts["cookiefile"] = args.cookies
    if args.cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = args.cookies_from_browser

    if args.no_progress:
        print(f"[download] {row.cascade_id} ({row.label}) {row.url}", flush=True)
    with YoutubeDL(ydl_opts) as ydl:
        return ydl.download([row.url])


def write_report(report_path: Path, report_rows: list[dict[str, str]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["cascade_id", "label", "url", "status", "detail", "error"],
        )
        writer.writeheader()
        writer.writerows(report_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download YouTube videos from FVC.csv into ./FVC."
    )
    parser.add_argument("--csv", default="FVC.csv", help="CSV containing video_url rows.")
    parser.add_argument("--output-dir", default="FVC", help="Download destination.")
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Download one URL instead of the full CSV. Repeat for multiple URLs.",
    )
    parser.add_argument(
        "--id",
        dest="cascade_ids",
        action="append",
        default=[],
        help="Download one cascade_id from the CSV. Repeat for multiple IDs.",
    )
    parser.add_argument("--limit", type=int, help="Maximum number of selected rows to process.")
    parser.add_argument(
        "--format",
        default="best[ext=mp4]/best",
        help="yt-dlp format selector. Defaults to a progressive MP4 when available.",
    )
    parser.add_argument(
        "--download-archive",
        help="yt-dlp archive file used to skip completed downloads. Defaults to <output-dir>/downloaded.txt.",
    )
    parser.add_argument("--cookies", help="Netscape cookies.txt file for private/age-gated videos.")
    parser.add_argument(
        "--cookies-from-browser",
        help=(
            "Load cookies from a signed-in browser, e.g. 'chrome', 'firefox', "
            "or 'chrome:Profile 1'. Uses yt-dlp's BROWSER[+KEYRING][:PROFILE][::CONTAINER] syntax."
        ),
    )
    parser.add_argument(
        "--restrict-filenames",
        action="store_true",
        help="Use ASCII-only filenames generated by yt-dlp.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce yt-dlp output.")
    parser.add_argument(
        "--interval-min",
        type=float,
        default=0.0,
        help="Minimum random sleep between downloads in seconds.",
    )
    parser.add_argument(
        "--interval-max",
        type=float,
        default=1.0,
        help="Maximum random sleep between downloads in seconds.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip rows already present in the archive or output directory.",
    )
    parser.add_argument(
        "--fallback-duplicates",
        type=Path,
        help=(
            "CSV of near-duplicate URLs to try when a seed URL fails, usually FVC_dup.csv. "
            "By default only same-cascade, same-label YouTube URLs are tried."
        ),
    )
    parser.add_argument(
        "--fallback-any-duplicate-label",
        action="store_true",
        help="Allow fallback duplicate URLs with labels different from the seed label.",
    )
    parser.add_argument(
        "--fallback-platform",
        choices=("youtube", "all"),
        default="youtube",
        help="Which duplicate URL platforms are allowed as fallbacks.",
    )
    parser.add_argument(
        "--max-fallbacks",
        type=int,
        default=10,
        help="Maximum duplicate fallback URLs to try per failed seed URL. Use -1 for no limit.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output.")
    parser.add_argument(
        "--check-youtube-cookies",
        action="store_true",
        help="Load configured cookies and report whether YouTube/Google cookies are visible.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected rows only.")
    return parser.parse_args()


def check_youtube_cookies(args: argparse.Namespace) -> int:
    if not args.cookies and not args.cookies_from_browser:
        print("No cookie source configured. Pass --cookies or --cookies-from-browser.")
        return 1

    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        sys.exit(
            "Missing dependency: yt-dlp\n"
            "Install it with: python -m pip install -r requirements.txt"
        )

    ydl_opts = {
        "quiet": False,
        "no_warnings": False,
    }
    if args.cookies:
        ydl_opts["cookiefile"] = args.cookies
    if args.cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = args.cookies_from_browser

    with YoutubeDL(ydl_opts) as ydl:
        cookiejar = ydl.cookiejar

    youtube_count = 0
    google_count = 0
    authish_count = 0
    authish_names = {
        "SAPISID",
        "__Secure-1PAPISID",
        "__Secure-3PAPISID",
        "SID",
        "__Secure-1PSID",
        "__Secure-3PSID",
        "LOGIN_INFO",
    }
    for cookie in cookiejar:
        domain = (cookie.domain or "").lower()
        if "youtube.com" in domain:
            youtube_count += 1
        if "google.com" in domain or "youtube.com" in domain:
            google_count += 1
            if cookie.name in authish_names:
                authish_count += 1

    print(f"visible YouTube cookies: {youtube_count}")
    print(f"visible Google/YouTube cookies: {google_count}")
    print(f"visible login-like cookies: {authish_count}")

    if google_count == 0 or authish_count == 0:
        print(
            "Cookie source loaded, but it does not appear to expose signed-in "
            "YouTube/Google auth cookies."
        )
        return 1

    print(
        "Cookie source exposes login-like cookies. If age-restricted downloads "
        "still fail, confirm the selected browser profile is signed in to an "
        "age-verified YouTube account."
    )
    return 0


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    if args.download_archive is None:
        args.download_archive = str(output_dir / "downloaded.txt")
    if args.interval_min < 0 or args.interval_max < 0:
        print("Sleep intervals must be non-negative.", file=sys.stderr)
        return 1
    if args.interval_min > args.interval_max:
        print("--interval-min cannot be greater than --interval-max.", file=sys.stderr)
        return 1
    if args.max_fallbacks is not None and args.max_fallbacks < -1:
        print("--max-fallbacks must be -1 or non-negative.", file=sys.stderr)
        return 1
    if args.max_fallbacks == -1:
        args.max_fallbacks = None
    if args.cookies_from_browser:
        try:
            args.cookies_from_browser = parse_cookies_from_browser(args.cookies_from_browser)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1

    if args.check_youtube_cookies:
        return check_youtube_cookies(args)

    rows = load_csv(csv_path)
    selected = select_rows(rows, args.url, args.cascade_ids, args.limit)
    if not selected:
        print("No videos selected.", file=sys.stderr)
        return 1

    if args.dry_run:
        for row in selected:
            print(f"{row.cascade_id},{row.label},{row.url}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    fallback_rows: dict[str, list[VideoRow]] = {}
    if args.fallback_duplicates:
        fallback_rows = group_rows_by_cascade(load_csv(args.fallback_duplicates))

    archive_path = Path(args.download_archive) if args.download_archive else None
    archive_ids = load_download_archive(archive_path) if archive_path else set()
    report_rows: list[dict[str, str]] = []
    failures = 0
    skipped = 0
    downloaded = 0

    progress = tqdm_iter(selected, enabled=not args.no_progress)
    for index, row in enumerate(progress, start=1):
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(id=row.cascade_id, label=row.label)

        skip_detail = None
        if not args.no_skip_existing:
            skip_detail = already_downloaded(row, output_dir, archive_ids)

        if skip_detail:
            skipped += 1
            report_rows.append(
                {
                    "cascade_id": row.cascade_id,
                    "label": row.label,
                    "url": row.url,
                    "status": "skipped",
                    "detail": skip_detail,
                    "error": "",
                }
            )
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(id=row.cascade_id, status="skipped")
        else:
            error = ""
            detail = ""
            status = "failed"
            successful_row = None
            attempt_errors: list[str] = []

            candidates = fallback_candidates(row, fallback_rows, args)
            for attempt_index, candidate in enumerate(candidates):
                try:
                    return_code = download_one(candidate, output_dir, args)
                    if return_code == 0:
                        status = "ok"
                        successful_row = candidate
                        break

                    status = f"failed:{return_code}"
                    attempt_errors.append(f"{candidate.url}: return code {return_code}")
                except Exception as exc:  # yt-dlp raises extractor/network errors here.
                    status = "failed"
                    attempt_errors.append(f"{candidate.url}: {exc}")
                    print(
                        f"[error] {candidate.cascade_id}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

                if attempt_index + 1 < len(candidates):
                    next_candidate = candidates[attempt_index + 1]
                    print(
                        "[fallback] "
                        f"{row.cascade_id}: trying duplicate URL {next_candidate.url}",
                        flush=True,
                    )

            if successful_row:
                downloaded += 1
                video_id = youtube_video_id(successful_row.url)
                if video_id:
                    archive_ids.add(video_id)
                if successful_row.url != row.url:
                    detail = f"fallback_url={successful_row.url}"
            else:
                failures += 1
                error = " | ".join(attempt_errors)

            report_rows.append(
                {
                    "cascade_id": row.cascade_id,
                    "label": row.label,
                    "url": row.url,
                    "status": status,
                    "detail": detail,
                    "error": error,
                }
            )
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(id=row.cascade_id, status=status)

        if not skip_detail and index < len(selected):
            time.sleep(random.uniform(args.interval_min, args.interval_max))

    write_report(output_dir / "download_report.csv", report_rows)
    print(
        "Wrote report: "
        f"{output_dir / 'download_report.csv'} "
        f"({downloaded} downloaded, {skipped} skipped, {failures} failed)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
