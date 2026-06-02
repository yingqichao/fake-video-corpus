#!/usr/bin/env python3
"""Download videos referenced by the JianLab/rumor Hugging Face datasets."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, NamedTuple
from urllib.parse import parse_qs, unquote, urlparse


DEFAULT_DATASETS = [f"JianLab/rumor-{index:02d}" for index in range(5)]
DEFAULT_OUTPUT_DIR = Path("/media/yingqichao/Lenovo/FVC_HF")
VIDEO_EXTENSIONS = {
    ".3gp",
    ".avi",
    ".flv",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mts",
    ".ogv",
    ".ts",
    ".webm",
    ".wmv",
}
KNOWN_VIDEO_HOSTS = {
    "facebook.com",
    "fb.watch",
    "instagram.com",
    "tiktok.com",
    "twitter.com",
    "vimeo.com",
    "x.com",
    "youtube.com",
    "youtu.be",
}
LABEL_COLUMNS = (
    "label",
    "labels",
    "class",
    "target",
    "veracity",
    "truth",
    "truthfulness",
    "annotation",
    "rumor_label",
)
ID_COLUMNS = (
    "id",
    "video_id",
    "uid",
    "guid",
    "post_id",
    "tweet_id",
    "source_id",
    "cascade_id",
)
URL_FIELD_HINTS = (
    "video",
    "url",
    "link",
    "media",
    "file",
    "path",
    "mp4",
    "webm",
    "mov",
    "mkv",
    "avi",
)


@dataclass(frozen=True)
class MediaCandidate:
    kind: str
    source: str
    column_path: str
    bytes_value: bytes | None = None


@dataclass(frozen=True)
class DownloadItem:
    dataset_id: str
    split: str
    row_index: int
    record_id: str
    label: str
    media_index: int
    candidate: MediaCandidate


class HfDatasetFile(NamedTuple):
    repo_id: str
    revision: str
    filename: str


@dataclass(frozen=True)
class MetadataRecord:
    record_id: str
    label: str
    raw_label: str
    metadata_source: str
    metadata_class: str = ""
    subject: str = ""
    cascade_id: str = ""


def slug(value: Any, fallback: str) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or fallback


def normalize_label(value: Any) -> str:
    text = str(value).strip()
    key = text.lower()
    label_map = {
        "\u771f": "real",
        "\u5047": "fake",
        "\u8f9f\u8c23": "debunked",
        "true": "real",
        "false": "fake",
        "real": "real",
        "fake": "fake",
        "mixture": "mixture",
    }
    return slug(label_map.get(key, text), "unknown")


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


def hf_token_arg(args: argparse.Namespace) -> str | bool | None:
    if args.token:
        return args.token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if args.use_login_token:
        return True
    return None


def maybe_import_datasets():
    try:
        from datasets import Video, get_dataset_split_names, load_dataset
    except ImportError:
        sys.exit(
            "Missing dependency: datasets\n"
            "Install dependencies with: python -m pip install -r requirements.txt"
        )
    return Video, get_dataset_split_names, load_dataset


def maybe_import_tqdm():
    try:
        from tqdm import tqdm
    except ImportError:
        sys.exit(
            "Missing dependency: tqdm\n"
            "Install dependencies with: python -m pip install -r requirements.txt"
        )
    return tqdm


def url_host(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host.removeprefix("www.")


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_hf_dataset_file(value: str) -> bool:
    return value.startswith("hf://datasets/")


def parse_hf_dataset_file(value: str) -> HfDatasetFile:
    prefix = "hf://datasets/"
    if not value.startswith(prefix):
        raise ValueError(f"not an hf dataset file URI: {value}")

    rest = value[len(prefix) :]
    if "@" not in rest:
        raise ValueError(f"hf dataset URI does not include a revision: {value}")

    repo_id, revision_and_filename = rest.split("@", maxsplit=1)
    if "/" not in revision_and_filename:
        raise ValueError(f"hf dataset URI does not include a filename: {value}")

    revision, filename = revision_and_filename.split("/", maxsplit=1)
    if not repo_id or not revision or not filename:
        raise ValueError(f"invalid hf dataset file URI: {value}")

    return HfDatasetFile(repo_id=repo_id, revision=revision, filename=filename)


def has_video_extension(value: str) -> bool:
    suffix = Path(unquote(urlparse(value).path)).suffix.lower()
    return suffix in VIDEO_EXTENSIONS


def is_known_video_url(value: str) -> bool:
    host = url_host(value)
    return any(host == known or host.endswith(f".{known}") for known in KNOWN_VIDEO_HOSTS)


def youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        return parsed.path.strip("/").split("/", maxsplit=1)[0] or None
    if host.endswith("youtube.com"):
        query_ids = parse_qs(parsed.query).get("v")
        if query_ids:
            return query_ids[0]
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None


def url_record_id(url: str) -> str:
    youtube_id = youtube_video_id(url)
    if youtube_id:
        return youtube_id

    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    for marker in ("videos", "status"):
        if marker in parts:
            marker_index = parts.index(marker)
            if marker_index + 1 < len(parts):
                return parts[marker_index + 1]
    return parts[-1] if parts else ""


def candidate_record_id(candidate: MediaCandidate) -> str:
    if candidate.kind == "hf_file":
        try:
            return Path(parse_hf_dataset_file(candidate.source).filename).stem
        except ValueError:
            return Path(candidate.source).stem
    if candidate.kind == "url":
        return url_record_id(candidate.source)
    if candidate.kind == "file":
        return Path(candidate.source).stem
    return Path(candidate.source).stem


def path_get(row: dict[str, Any], dotted_path: str) -> Any:
    current: Any = row
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(dotted_path)
    return current


def iter_paths(value: Any, prefix: str) -> Iterable[tuple[str, Any]]:
    yield prefix, value
    if isinstance(value, dict):
        for key, nested in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_paths(nested, nested_prefix)
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            nested_prefix = f"{prefix}[{index}]"
            yield from iter_paths(nested, nested_prefix)


def candidate_from_value(path: str, value: Any, *, strict_video: bool) -> MediaCandidate | None:
    if isinstance(value, bytes):
        return MediaCandidate(kind="bytes", source=path, column_path=path, bytes_value=value)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if is_hf_dataset_file(stripped):
            if not strict_video or has_video_extension(stripped):
                return MediaCandidate(kind="hf_file", source=stripped, column_path=path)
        if is_url(stripped):
            if not strict_video or has_video_extension(stripped) or is_known_video_url(stripped):
                return MediaCandidate(kind="url", source=stripped, column_path=path)
        else:
            local_path = Path(stripped).expanduser()
            if local_path.exists() and local_path.is_file():
                return MediaCandidate(kind="file", source=str(local_path), column_path=path)
        return None

    if isinstance(value, dict):
        bytes_value = value.get("bytes")
        path_value = value.get("path")
        if isinstance(bytes_value, bytes):
            source = str(path_value) if path_value else path
            return MediaCandidate(
                kind="bytes",
                source=source,
                column_path=path,
                bytes_value=bytes_value,
            )
        if isinstance(path_value, str):
            nested = candidate_from_value(f"{path}.path", path_value, strict_video=strict_video)
            if nested:
                return nested
        url_value = value.get("url")
        if isinstance(url_value, str):
            nested = candidate_from_value(f"{path}.url", url_value, strict_video=strict_video)
            if nested:
                return nested

    return None


def likely_media_root(path: str) -> bool:
    lowered = path.lower()
    return any(hint in lowered for hint in URL_FIELD_HINTS)


def extract_media_candidates(
    row: dict[str, Any],
    video_columns: list[str],
) -> list[MediaCandidate]:
    def collect(paths: list[tuple[str, Any]], *, strict_video: bool) -> list[MediaCandidate]:
        candidates: list[MediaCandidate] = []
        seen = set()
        for path, value in paths:
            candidate = candidate_from_value(path, value, strict_video=strict_video)
            if not candidate:
                continue
            identity = (candidate.kind, candidate.source)
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(candidate)
        return candidates

    paths: list[tuple[str, Any]] = []
    if video_columns:
        for column in video_columns:
            try:
                paths.append((column, path_get(row, column)))
            except KeyError:
                continue
        return collect(paths, strict_video=False)

    hinted_paths: list[tuple[str, Any]] = []
    all_paths: list[tuple[str, Any]] = []
    for key, value in row.items():
        nested_paths = list(iter_paths(value, str(key)))
        all_paths.extend(nested_paths)
        if likely_media_root(str(key)):
            hinted_paths.extend(nested_paths)

    candidates = collect(hinted_paths, strict_video=True)
    if candidates:
        return candidates

    return collect(all_paths, strict_video=True)


def first_present(row: dict[str, Any], candidates: tuple[str, ...]) -> Any | None:
    lower_to_key = {str(key).lower(): key for key in row}
    for candidate in candidates:
        key = lower_to_key.get(candidate.lower())
        if key is not None:
            return row[key]
    return None


def label_for_row(row: dict[str, Any]) -> str:
    value = first_present(row, LABEL_COLUMNS)
    return slug(value, "unknown") if value is not None else "unknown"


def id_for_row(row: dict[str, Any], row_index: int) -> str:
    value = first_present(row, ID_COLUMNS)
    return slug(value, f"row{row_index:06d}") if value is not None else f"row{row_index:06d}"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def hf_metadata_file(dataset_id: str, filename: str, args: argparse.Namespace) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "Missing dependency: huggingface_hub\n"
            "Install dependencies with: python -m pip install -r requirements.txt"
        )

    return Path(
        hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=filename,
            token=hf_token_arg(args),
        )
    )


def load_jsonl_metadata_map(
    dataset_id: str,
    filename: str,
    label_field: str,
    args: argparse.Namespace,
) -> dict[str, MetadataRecord]:
    path = hf_metadata_file(dataset_id, filename, args)
    records: dict[str, MetadataRecord] = {}
    for row in read_jsonl(path):
        video_id = str(row.get("video_id", "")).strip()
        if not video_id:
            continue
        raw_label = str(row.get(label_field, "")).strip()
        records[video_id] = MetadataRecord(
            record_id=video_id,
            label=normalize_label(raw_label),
            raw_label=raw_label,
            metadata_source=filename,
            metadata_class=str(row.get("class", "")),
            subject=str(row.get("event") or row.get("keywords") or ""),
        )
    return records


def load_json_metadata_map(
    dataset_id: str,
    filename: str,
    label_field: str,
    args: argparse.Namespace,
) -> dict[str, MetadataRecord]:
    path = hf_metadata_file(dataset_id, filename, args)
    rows = json.loads(path.read_text(encoding="utf-8"))
    records: dict[str, MetadataRecord] = {}
    for row in rows:
        video_id = str(row.get("video_id", "")).strip()
        if not video_id:
            continue
        raw_label = str(row.get(label_field, "")).strip()
        records[video_id] = MetadataRecord(
            record_id=video_id,
            label=normalize_label(raw_label),
            raw_label=raw_label,
            metadata_source=filename,
            metadata_class=str(row.get("class", "")),
            subject=str(row.get("subject") or row.get("event") or ""),
        )
    return records


def load_fvc_metadata_map(args: argparse.Namespace) -> dict[str, MetadataRecord]:
    records: dict[str, MetadataRecord] = {}
    for path, source in ((args.fvc_csv, "FVC.csv"), (args.fvc_dup_csv, "FVC_dup.csv")):
        if not path or not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                video_id = url_record_id(row.get("video_url", ""))
                if not video_id:
                    continue
                raw_label = str(row.get("label", "")).strip()
                records[video_id] = MetadataRecord(
                    record_id=video_id,
                    label=normalize_label(raw_label),
                    raw_label=raw_label,
                    metadata_source=source,
                    cascade_id=str(row.get("cascade_id", "")),
                )
    return records


def load_rumor04_metadata_map(dataset_id: str, args: argparse.Namespace) -> dict[str, MetadataRecord]:
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        sys.exit(
            "Missing dependency: huggingface_hub\n"
            "Install dependencies with: python -m pip install -r requirements.txt"
        )

    records: dict[str, MetadataRecord] = {}
    for filename in list_repo_files(dataset_id, repo_type="dataset", token=hf_token_arg(args)):
        if not filename.endswith(".json"):
            continue
        path = hf_metadata_file(dataset_id, filename, args)
        row = json.loads(path.read_text(encoding="utf-8"))
        video_id = Path(filename).stem
        raw_label = str(row.get("rating", "")).strip()
        records[video_id] = MetadataRecord(
            record_id=video_id,
            label=normalize_label(raw_label),
            raw_label=raw_label,
            metadata_source=filename,
            subject=str(row.get("claim") or ""),
        )
    return records


def load_metadata_map(dataset_id: str, args: argparse.Namespace) -> dict[str, MetadataRecord]:
    dataset_name = dataset_id.rsplit("/", maxsplit=1)[-1]
    try:
        if dataset_name == "rumor-00":
            return load_jsonl_metadata_map(dataset_id, "data_complete.jsonl", "annotation", args)
        if dataset_name == "rumor-01":
            return load_jsonl_metadata_map(dataset_id, "data_complete.jsonl", "annotation", args)
        if dataset_name == "rumor-02":
            return load_fvc_metadata_map(args)
        if dataset_name == "rumor-03":
            return load_json_metadata_map(dataset_id, "data.json", "label", args)
        if dataset_name == "rumor-04":
            return load_rumor04_metadata_map(dataset_id, args)
    except Exception as exc:
        print(f"[warn] Could not load metadata labels for {dataset_id}: {exc}", file=sys.stderr)
    return {}


def output_stem(item: DownloadItem) -> str:
    dataset_name = item.dataset_id.rsplit("/", maxsplit=1)[-1]
    suffix = f"__m{item.media_index}" if item.media_index else ""
    return slug(f"{dataset_name}__{item.split}__{item.record_id}{suffix}", f"row{item.row_index:06d}")


def output_dir_for(item: DownloadItem, root: Path) -> Path:
    dataset_name = slug(item.dataset_id.rsplit("/", maxsplit=1)[-1], "dataset")
    return root / dataset_name / slug(item.split, "split") / item.label


def find_existing_file(directory: Path, stem: str) -> Path | None:
    if not directory.exists():
        return None
    for path in sorted(directory.glob(f"{stem}.*")):
        if path.is_file() and not path.name.endswith((".part", ".ytdl")):
            return path
    return None


def infer_extension(source: str, default: str = ".mp4") -> str:
    suffix = Path(unquote(urlparse(source).path)).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return suffix
    guessed = mimetypes.guess_extension(mimetypes.guess_type(source)[0] or "")
    if guessed and guessed.lower() in VIDEO_EXTENSIONS:
        return guessed.lower()
    return default


def write_bytes(item: DownloadItem, root: Path) -> Path:
    directory = output_dir_for(item, root)
    directory.mkdir(parents=True, exist_ok=True)
    stem = output_stem(item)
    extension = infer_extension(item.candidate.source)
    output_path = directory / f"{stem}{extension}"
    if item.candidate.bytes_value is None:
        raise ValueError("bytes candidate did not contain bytes")
    output_path.write_bytes(item.candidate.bytes_value)
    return output_path


def copy_file(item: DownloadItem, root: Path) -> Path:
    source = Path(item.candidate.source)
    directory = output_dir_for(item, root)
    directory.mkdir(parents=True, exist_ok=True)
    stem = output_stem(item)
    extension = source.suffix if source.suffix.lower() in VIDEO_EXTENSIONS else ".mp4"
    output_path = directory / f"{stem}{extension}"
    shutil.copy2(source, output_path)
    return output_path


def download_hf_file(item: DownloadItem, root: Path, args: argparse.Namespace) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "Missing dependency: huggingface_hub\n"
            "Install dependencies with: python -m pip install -r requirements.txt"
        )

    hf_file = parse_hf_dataset_file(item.candidate.source)
    cached_path = hf_hub_download(
        repo_id=hf_file.repo_id,
        repo_type="dataset",
        revision=hf_file.revision,
        filename=hf_file.filename,
        token=hf_token_arg(args),
    )

    directory = output_dir_for(item, root)
    directory.mkdir(parents=True, exist_ok=True)
    stem = output_stem(item)
    suffix = Path(hf_file.filename).suffix.lower()
    extension = suffix if suffix in VIDEO_EXTENSIONS else ".mp4"
    output_path = directory / f"{stem}{extension}"
    shutil.copy2(cached_path, output_path)
    return output_path


def download_url(item: DownloadItem, root: Path, args: argparse.Namespace) -> Path:
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        sys.exit(
            "Missing dependency: yt-dlp\n"
            "Install dependencies with: python -m pip install -r requirements.txt"
        )

    directory = output_dir_for(item, root)
    directory.mkdir(parents=True, exist_ok=True)
    stem = output_stem(item)
    outtmpl = str(directory / f"{stem}.%(ext)s")

    ydl_opts: dict[str, Any] = {
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

    with YoutubeDL(ydl_opts) as ydl:
        return_code = ydl.download([item.candidate.source])
    if return_code != 0:
        raise RuntimeError(f"yt-dlp returned {return_code}")

    existing = find_existing_file(directory, stem)
    if existing:
        return existing
    return directory / f"{stem}.unknown"


def download_item(item: DownloadItem, root: Path, args: argparse.Namespace) -> tuple[str, str]:
    directory = output_dir_for(item, root)
    stem = output_stem(item)
    existing = find_existing_file(directory, stem)
    if existing and not args.no_skip_existing:
        return "skipped", str(existing)

    if args.dry_run:
        return "dry_run", ""

    if item.candidate.kind == "url":
        output_path = download_url(item, root, args)
    elif item.candidate.kind == "hf_file":
        output_path = download_hf_file(item, root, args)
    elif item.candidate.kind == "file":
        output_path = copy_file(item, root)
    elif item.candidate.kind == "bytes":
        output_path = write_bytes(item, root)
    else:
        raise ValueError(f"unsupported media candidate kind: {item.candidate.kind}")

    return "ok", str(output_path)


def discover_splits(dataset_id: str, args: argparse.Namespace) -> list[str]:
    if args.split:
        return args.split
    _, get_dataset_split_names, _ = maybe_import_datasets()
    token = hf_token_arg(args)
    try:
        return list(get_dataset_split_names(dataset_id, token=token))
    except Exception as exc:
        print(
            f"[warn] Could not discover splits for {dataset_id}: {exc}. "
            "Trying split='train'.",
            file=sys.stderr,
        )
        return ["train"]


def iter_dataset_rows(dataset_id: str, split: str, args: argparse.Namespace):
    Video, _, load_dataset = maybe_import_datasets()
    token = hf_token_arg(args)
    dataset = load_dataset(
        dataset_id,
        split=split,
        streaming=args.streaming,
        token=token,
        trust_remote_code=args.trust_remote_code,
    )
    for column_name, feature in (dataset.features or {}).items():
        if isinstance(feature, Video) and feature.decode:
            dataset = dataset.cast_column(column_name, Video(decode=False))
    return dataset


def write_report_header(report_writer: csv.DictWriter) -> None:
    report_writer.writeheader()


def inspect_row(dataset_id: str, split: str, row: dict[str, Any], args: argparse.Namespace) -> None:
    print(f"\n== {dataset_id} / {split} sample ==")
    print("columns:", ", ".join(row.keys()))
    print(f"detected label: {label_for_row(row)}")
    print(f"detected record id: {id_for_row(row, 0)}")
    candidates = extract_media_candidates(row, args.video_column)
    print(f"detected media candidates: {len(candidates)}")
    for index, candidate in enumerate(candidates):
        source = candidate.source
        if len(source) > 180:
            source = source[:180] + "..."
        print(f"  [{index}] {candidate.kind} {candidate.column_path}: {source}")
    for key, value in row.items():
        text = repr(value)
        if len(text) > 220:
            text = text[:220] + "..."
        print(f"  {key} = {text}")


def run(args: argparse.Namespace) -> int:
    if args.cookies_from_browser:
        try:
            args.cookies_from_browser = parse_cookies_from_browser(args.cookies_from_browser)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
    if args.interval_min < 0 or args.interval_max < 0:
        print("Sleep intervals must be non-negative.", file=sys.stderr)
        return 1
    if args.interval_min > args.interval_max:
        print("--interval-min cannot be greater than --interval-max.", file=sys.stderr)
        return 1

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    if args.download_archive is None:
        args.download_archive = str(output_root / "downloaded.txt")

    report_path = output_root / "download_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.quiet:
        print(f"Output directory: {output_root}")
        print(f"Existing downloads are checked under: {output_root}")

    tqdm = maybe_import_tqdm() if not args.no_progress else None
    failures = 0
    downloaded = 0
    skipped = 0
    dry_runs = 0
    no_media = 0

    with report_path.open("w", newline="", encoding="utf-8") as report_handle:
        report_writer = csv.DictWriter(
            report_handle,
            fieldnames=[
                "dataset_id",
                "split",
                "row_index",
                "record_id",
                "label",
                "raw_label",
                "metadata_source",
                "metadata_class",
                "subject",
                "cascade_id",
                "media_index",
                "candidate_kind",
                "candidate_column",
                "source",
                "status",
                "output_path",
                "error",
            ],
        )
        write_report_header(report_writer)

        for dataset_id in args.dataset:
            metadata_map = load_metadata_map(dataset_id, args)
            for split in discover_splits(dataset_id, args):
                try:
                    rows = iter_dataset_rows(dataset_id, split, args)
                except Exception as exc:
                    failures += 1
                    print(f"[error] Could not load {dataset_id}/{split}: {exc}", file=sys.stderr)
                    continue

                iterator = enumerate(rows)
                if tqdm:
                    iterator = tqdm(iterator, desc=f"{dataset_id}/{split}", unit="row")

                inspected = False
                split_processed = 0
                for row_index, row in iterator:
                    if not isinstance(row, dict):
                        failures += 1
                        print(
                            f"[error] {dataset_id}/{split} row {row_index}: expected dict row",
                            file=sys.stderr,
                        )
                        continue

                    if args.inspect and not inspected:
                        inspect_row(dataset_id, split, row, args)
                        inspected = True
                        if args.inspect_only:
                            break

                    candidates = extract_media_candidates(row, args.video_column)
                    candidate_id = candidate_record_id(candidates[0]) if candidates else ""
                    metadata = metadata_map.get(candidate_id)
                    record_id = (
                        metadata.record_id
                        if metadata
                        else candidate_id or id_for_row(row, row_index)
                    )
                    label = metadata.label if metadata else label_for_row(row)
                    raw_label = metadata.raw_label if metadata else ""
                    metadata_source = metadata.metadata_source if metadata else ""
                    metadata_class = metadata.metadata_class if metadata else ""
                    subject = metadata.subject if metadata else ""
                    cascade_id = metadata.cascade_id if metadata else ""

                    if args.first_media_only and candidates:
                        candidates = candidates[:1]

                    if not candidates:
                        no_media += 1
                        report_writer.writerow(
                            {
                                "dataset_id": dataset_id,
                                "split": split,
                                "row_index": row_index,
                                "record_id": record_id,
                                "label": label,
                                "raw_label": raw_label,
                                "metadata_source": metadata_source,
                                "metadata_class": metadata_class,
                                "subject": subject,
                                "cascade_id": cascade_id,
                                "media_index": "",
                                "candidate_kind": "",
                                "candidate_column": "",
                                "source": "",
                                "status": "no_media",
                                "output_path": "",
                                "error": "",
                            }
                        )
                    else:
                        for media_index, candidate in enumerate(candidates):
                            item = DownloadItem(
                                dataset_id=dataset_id,
                                split=split,
                                row_index=row_index,
                                record_id=record_id,
                                label=label,
                                media_index=media_index,
                                candidate=candidate,
                            )
                            status = "failed"
                            output_path = ""
                            error = ""
                            try:
                                status, output_path = download_item(item, output_root, args)
                                downloaded += int(status == "ok")
                                skipped += int(status == "skipped")
                                dry_runs += int(status == "dry_run")
                            except Exception as exc:
                                failures += 1
                                error = str(exc)
                                print(
                                    f"[error] {dataset_id}/{split} row {row_index}: {error}",
                                    file=sys.stderr,
                                    flush=True,
                                )

                            report_writer.writerow(
                                {
                                    "dataset_id": dataset_id,
                                    "split": split,
                                    "row_index": row_index,
                                    "record_id": record_id,
                                    "label": label,
                                    "raw_label": raw_label,
                                    "metadata_source": metadata_source,
                                    "metadata_class": metadata_class,
                                    "subject": subject,
                                    "cascade_id": cascade_id,
                                    "media_index": media_index,
                                    "candidate_kind": candidate.kind,
                                    "candidate_column": candidate.column_path,
                                    "source": candidate.source,
                                    "status": status,
                                    "output_path": output_path,
                                    "error": error,
                                }
                            )

                    report_handle.flush()
                    split_processed += 1
                    if args.limit_per_split is not None and split_processed >= args.limit_per_split:
                        break
                    if args.interval_max and not args.dry_run:
                        time.sleep(random.uniform(args.interval_min, args.interval_max))

    print(
        f"Wrote report: {report_path} "
        f"({downloaded} downloaded, {skipped} skipped, {dry_runs} dry-run, "
        f"{no_media} no-media rows, {failures} failed)"
    )
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download videos referenced by JianLab/rumor-00 through rumor-04 "
            f"from Hugging Face into {DEFAULT_OUTPUT_DIR}."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset id to process. Repeat for multiple ids. Defaults to all JianLab/rumor-00..04.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help="Split to process. Repeat for multiple splits. Defaults to auto-discovery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Download destination. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--video-column",
        action="append",
        default=[],
        help=(
            "Column or dotted path containing the video URL/blob/path. Repeat for multiple "
            "columns. If omitted, the script auto-detects likely media fields."
        ),
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        help="Maximum rows to process per dataset split.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Use regular load_dataset instead of streaming iteration.",
    )
    parser.set_defaults(streaming=True)
    parser.add_argument(
        "--token",
        help="Hugging Face token. Prefer HF_TOKEN env var instead of putting tokens in shell history.",
    )
    parser.add_argument(
        "--use-login-token",
        action="store_true",
        help="Use the token saved by huggingface-cli login.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom dataset loading code if the dataset requires it.",
    )
    parser.add_argument(
        "--fvc-csv",
        type=Path,
        default=Path("FVC.csv"),
        help="Official FVC seed CSV used to label matching rumor-02 videos.",
    )
    parser.add_argument(
        "--fvc-dup-csv",
        type=Path,
        default=Path("FVC_dup.csv"),
        help="Official FVC duplicate CSV used to label matching rumor-02 videos.",
    )
    parser.add_argument(
        "--format",
        default="best[ext=mp4]/best",
        help="yt-dlp format selector for URL downloads.",
    )
    parser.add_argument(
        "--download-archive",
        help="yt-dlp archive file. Defaults to <output-dir>/downloaded.txt.",
    )
    parser.add_argument("--cookies", help="Netscape cookies.txt file for gated video platforms.")
    parser.add_argument(
        "--cookies-from-browser",
        help=(
            "Load cookies from a signed-in browser, e.g. 'chrome+GNOMEKEYRING'. "
            "Uses yt-dlp's BROWSER[+KEYRING][:PROFILE][::CONTAINER] syntax."
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
        help="Minimum random sleep between rows in seconds.",
    )
    parser.add_argument(
        "--interval-max",
        type=float,
        default=0.0,
        help="Maximum random sleep between rows in seconds.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip files that already exist in the output directory.",
    )
    parser.add_argument(
        "--first-media-only",
        action="store_true",
        help="If a row exposes multiple media candidates, download only the first.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output.")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print detected columns and media candidates for the first row of each split.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect the first row of each split; do not download.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report selected media without downloading.")
    args = parser.parse_args()
    if not args.dataset:
        args.dataset = DEFAULT_DATASETS
    return args


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
