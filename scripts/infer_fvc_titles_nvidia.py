#!/usr/bin/env python3
"""Run title-only FVC real/fake inference through NVIDIA's chat API."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_API_URL = "https://inference-api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "openai/openai/gpt-5.1"
DEFAULT_OUTPUT = Path("outputs/fvc_title_llm_predictions.json")
DEFAULT_PROMPT_FILE = Path(__file__).with_name("fvc_language_prompt.txt")
FRENCH_ACCENT_RE = re.compile(r"[àâçéèêëîïôûùüÿæœáãíóõúñ]")
FRENCH_CUE_WORDS = {
    "aérienne",
    "après",
    "attaque",
    "attaques",
    "avec",
    "avion",
    "baie",
    "blessés",
    "caméra",
    "catastrophe",
    "contre",
    "dans",
    "déclenche",
    "détruit",
    "effondrement",
    "enfant",
    "femme",
    "feu",
    "fille",
    "frappe",
    "frappé",
    "grève",
    "homme",
    "incendie",
    "les",
    "mer",
    "mettre",
    "monde",
    "morts",
    "ouragan",
    "passagers",
    "pluie",
    "pour",
    "près",
    "raid",
    "requin",
    "requins",
    "russie",
    "sur",
    "tigre",
    "tornade",
    "volcan",
    "voit",
    "vue",
}


@dataclass(frozen=True)
class FvcTitleRow:
    line_number: int
    cascade_id: str
    video_url: str
    label: str
    event_title: str
    malformed_field_count: int


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def looks_like_french_field(value: str) -> bool:
    text = value.strip().lower()
    if not text:
        return False

    score = 0
    if FRENCH_ACCENT_RE.search(text):
        score += 2
    if re.search(r"\b(?:l|d|qu)['’]", text):
        score += 2
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿæœáãíóõúñ]+", text)
    score += sum(1 for word in words if word in FRENCH_CUE_WORDS)
    return score >= 1


def starts_like_existing_title(candidate: str, title_parts: list[str]) -> bool:
    if not title_parts:
        return False
    title_words = re.findall(r"[a-z0-9]+", " ".join(title_parts).lower())
    candidate_words = re.findall(r"[a-z0-9]+", candidate.lower())
    if len(title_words) < 2 or len(candidate_words) < 2:
        return False
    return candidate_words[:2] == title_words[:2]


def recover_event_title(raw_row: list[str], expected_fields: int) -> str:
    title_parts = [raw_row[3].strip()]
    if len(raw_row) <= expected_fields:
        return title_parts[0]

    for field in raw_row[4:]:
        value = field.strip()
        if looks_like_french_field(value):
            break
        if len(title_parts) > 1 and starts_like_existing_title(value, title_parts):
            break
        if re.search(r"[\u0400-\u04ff\u0600-\u06ff]", value):
            break
        title_parts.append(value)

    title = title_parts[0]
    for part in title_parts[1:]:
        separator = "," if re.search(r"[$£€]\d+$", title) else ", "
        title += separator + part
    return title


def load_fvc_titles(path: Path) -> list[FvcTitleRow]:
    rows: list[FvcTitleRow] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        expected_fields = len(header)
        for line_number, raw_row in enumerate(reader, start=2):
            if len(raw_row) < 4:
                continue
            rows.append(
                FvcTitleRow(
                    line_number=line_number,
                    cascade_id=raw_row[0].strip(),
                    video_url=raw_row[1].strip(),
                    label=raw_row[2].strip().lower(),
                    event_title=recover_event_title(raw_row, expected_fields),
                    malformed_field_count=0
                    if len(raw_row) == expected_fields
                    else len(raw_row),
                )
            )
    return rows


def analyze_title_patterns(rows: list[FvcTitleRow]) -> dict[str, Any]:
    labels = Counter(row.label for row in rows)
    fake_titles = [row.event_title.lower() for row in rows if row.label == "fake"]
    real_titles = [row.event_title.lower() for row in rows if row.label == "real"]
    hint_terms = [
        "alien",
        "ufo",
        "ghost",
        "mermaid",
        "creature",
        "teleportation",
        "invisibility",
        "magic",
        "trick",
        "giant",
        "strange",
        "shark",
        "volcano",
        "killer whale",
        "isis",
        "airstrike",
        "strike",
        "ammunition depot",
    ]
    term_stats = {}
    for term in hint_terms:
        term_stats[term] = {
            "fake": sum(term in title for title in fake_titles),
            "real": sum(term in title for title in real_titles),
        }
    return {
        "row_count": len(rows),
        "labels": dict(labels),
        "hint_term_stats": term_stats,
        "malformed_rows": sum(1 for row in rows if row.malformed_field_count),
    }


def load_system_prompt(path: Path) -> str:
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"empty prompt file: {path}")
    return prompt


def build_user_prompt(row: FvcTitleRow) -> str:
    return (
        "Classify this event title using title text only.\n\n"
        f"Event title: {row.event_title}\n\n"
        "Return JSON only."
    )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_prediction(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"real", "true", "verified"}:
        return "real"
    if text in {"fake", "false", "debunked"}:
        return "fake"
    if "fake" in text or "false" in text:
        return "fake"
    if "real" in text or "true" in text:
        return "real"
    raise ValueError(f"invalid binary prediction: {value!r}")


def call_nvidia_chat(
    *,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    api_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1.0,
    }

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not content:
                raise RuntimeError(f"empty response content: {data}")
            return content
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(2**attempt)
            else:
                raise RuntimeError(f"API failed after {max_retries + 1} attempts: {exc}") from exc

    raise RuntimeError(f"unreachable API failure: {last_error}")


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    results = data.get("results", []) if isinstance(data, dict) else []
    existing = {}
    for item in results:
        cascade_id = item.get("cascade_id")
        if (
            cascade_id
            and item.get("prediction") in {"real", "fake"}
            and not item.get("error")
        ):
            existing[cascade_id] = item
    return existing


def write_output(
    path: Path,
    *,
    args: argparse.Namespace,
    pattern_analysis: dict[str, Any],
    system_prompt: str,
    results: list[dict[str, Any]],
) -> None:
    correct = sum(1 for item in results if item.get("correct") is True)
    evaluated = sum(1 for item in results if item.get("correct") is not None)
    summary = {
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": correct / evaluated if evaluated else None,
        "by_label": {},
    }
    for label in sorted({item.get("label") for item in results}):
        subset = [item for item in results if item.get("label") == label]
        label_eval = [item for item in subset if item.get("correct") is not None]
        label_correct = sum(1 for item in label_eval if item.get("correct") is True)
        summary["by_label"][label] = {
            "count": len(label_eval),
            "correct": label_correct,
            "accuracy": label_correct / len(label_eval) if label_eval else None,
        }

    output = {
        "metadata": {
            "input_csv": str(args.csv),
            "prompt_file": str(args.prompt_file),
            "model": args.model,
            "api_url": args.api_url,
            "temperature": args.temperature,
            "title_only": True,
            "prediction_space": ["real", "fake"],
        },
        "pattern_analysis": pattern_analysis,
        "system_prompt": system_prompt,
        "summary": summary,
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify FVC event titles as real/fake using NVIDIA's chat API."
    )
    parser.add_argument("--csv", type=Path, default=Path("FVC_text_queries.csv"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help=f"System prompt file. Defaults to {DEFAULT_PROMPT_FILE}.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key", help="API key. Prefer .env or environment variables.")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("NVIDIA_API_URL", DEFAULT_API_URL),
        help=f"Chat completions endpoint. Defaults to {DEFAULT_API_URL}.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("NVIDIA_MODEL", DEFAULT_MODEL),
        help=f"Model name. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Limit number of rows for a smoke test.")
    parser.add_argument(
        "--id",
        dest="cascade_ids",
        action="append",
        default=[],
        help="Only process this cascade_id. Repeat for multiple IDs.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse existing output rows.")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts and do not call API.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)
    if args.api_url == DEFAULT_API_URL:
        args.api_url = os.environ.get("NVIDIA_API_URL", args.api_url)
    if args.model == DEFAULT_MODEL:
        args.model = os.environ.get("NVIDIA_MODEL", args.model)

    api_key = (
        args.api_key
        or os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key and not args.dry_run:
        print(
            "Missing API key. Set NVIDIA_API_KEY or OPENAI_API_KEY in .env or environment.",
            file=sys.stderr,
        )
        return 1

    rows = load_fvc_titles(args.csv)
    if args.cascade_ids:
        wanted = set(args.cascade_ids)
        rows = [row for row in rows if row.cascade_id in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print("No rows selected.", file=sys.stderr)
        return 1

    pattern_analysis = analyze_title_patterns(load_fvc_titles(args.csv))
    system_prompt = load_system_prompt(args.prompt_file)
    existing = load_existing_results(args.output) if args.resume else {}
    results: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        if args.resume and row.cascade_id in existing:
            results.append(existing[row.cascade_id])
            continue

        user_prompt = build_user_prompt(row)
        print(f"[{index}/{len(rows)}] {row.cascade_id} label={row.label} title={row.event_title}")
        if args.dry_run:
            print("--- system prompt ---")
            print(system_prompt)
            print("--- user prompt ---")
            print(user_prompt)
            raw_response = ""
            prediction = None
            reasoning = ""
            error = None
        else:
            try:
                raw_response = call_nvidia_chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    api_key=api_key or "",
                    api_url=args.api_url,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                )
                parsed = extract_json_object(raw_response)
                prediction = normalize_prediction(parsed.get("prediction"))
                reasoning = str(parsed.get("reasoning", "")).strip()
                error = None
            except Exception as exc:
                raw_response = ""
                prediction = None
                reasoning = ""
                error = str(exc)
                print(f"  error: {error}", file=sys.stderr)

        correct = prediction == row.label if prediction in {"real", "fake"} else None
        if prediction:
            print(f"  prediction={prediction} correct={correct}")
        result = {
            **asdict(row),
            "prediction": prediction,
            "reasoning": reasoning,
            "raw_response": raw_response,
            "correct": correct,
            "error": error,
        }
        results.append(result)
        write_output(
            args.output,
            args=args,
            pattern_analysis=pattern_analysis,
            system_prompt=system_prompt,
            results=results,
        )

    write_output(
        args.output,
        args=args,
        pattern_analysis=pattern_analysis,
        system_prompt=system_prompt,
        results=results,
    )
    print(f"Wrote {args.output}")
    return 1 if any(item.get("error") for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
