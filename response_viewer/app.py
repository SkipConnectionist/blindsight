#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FILENAME_RE = re.compile(r"^intersect_([YN])_distractor_([YN])_(.+)\.png$", re.IGNORECASE)
YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


@dataclass
class Record:
    image_name: str
    image_url: str
    response_text: str
    expected_intersects: bool
    predicted_intersects: Optional[bool]
    correct: bool


@dataclass
class AppState:
    image_dir: Path
    records: list[Record]
    template_html: str
    css: str
    js: str


def compute_global_metrics(records: list[Record]) -> dict[str, float | int]:
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for record in records:
        predicted = record.predicted_intersects
        expected = record.expected_intersects
        if predicted is True and expected is True:
            true_positives += 1
        elif predicted is False and expected is False:
            true_negatives += 1
        elif predicted is True and expected is False:
            false_positives += 1
        elif predicted is False and expected is True:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0

    return {
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dataset response viewer web server.")
    parser.add_argument(
        "--response-cache",
        required=True,
        type=Path,
        help="Path to response cache directory. Can be a prompt-hash leaf directory or a parent containing prompt-hash subdirectories.",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        type=Path,
        help="Path to source image directory containing PNG files.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind (default: 8000)")
    return parser.parse_args()


def parse_ground_truth(image_name: str) -> bool:
    match = FILENAME_RE.match(image_name)
    if not match:
        raise ValueError(f"Image filename does not match expected format: {image_name}")
    intersects_flag, _distractor_flag, _suffix = match.groups()
    return intersects_flag.upper() == "Y"


def parse_yes_no(text: str) -> Optional[bool]:
    match = YES_NO_RE.search(text)
    if not match:
        return None
    return match.group(1).lower() == "yes"


def load_template_assets(base_dir: Path) -> tuple[str, str, str]:
    template_html = (base_dir / "templates" / "index.html").read_text(encoding="utf-8")
    css = (base_dir / "static" / "styles.css").read_text(encoding="utf-8")
    js = (base_dir / "static" / "app.js").read_text(encoding="utf-8")
    return template_html, css, js


def discover_response_files(response_cache_path: Path) -> list[Path]:
    if not response_cache_path.exists() or not response_cache_path.is_dir():
        raise SystemExit(f"Response cache path must be an existing directory: {response_cache_path}")

    direct_txt = sorted(p for p in response_cache_path.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
    if direct_txt:
        return direct_txt

    nested_txt = sorted(
        p
        for p in response_cache_path.rglob("*.txt")
        if p.is_file() and p.parent != response_cache_path
    )
    if nested_txt:
        return nested_txt

    raise SystemExit(f"No .txt response files found under: {response_cache_path}")


def build_records(response_cache_path: Path, image_dir: Path) -> list[Record]:
    if not image_dir.exists() or not image_dir.is_dir():
        raise SystemExit(f"Image directory must be an existing directory: {image_dir}")

    records: list[Record] = []
    response_files = discover_response_files(response_cache_path)

    for txt_file in response_files:
        image_name = txt_file.name[:-4] if txt_file.name.lower().endswith(".txt") else txt_file.name
        image_path = image_dir / image_name
        if not image_path.exists() or not image_path.is_file():
            continue

        try:
            expected = parse_ground_truth(image_name)
        except ValueError:
            continue
        response_text = txt_file.read_text(encoding="utf-8").strip()
        predicted = parse_yes_no(response_text)
        correct = predicted is not None and predicted == expected

        records.append(
            Record(
                image_name=image_name,
                image_url=f"/image/{quote(image_name)}",
                response_text=response_text,
                expected_intersects=expected,
                predicted_intersects=predicted,
                correct=correct,
            )
        )

    records.sort(key=lambda r: r.image_name)
    if not records:
        raise SystemExit(
            "No records found with matching image and response files. "
            "Ensure response text files are named like '<image_name>.txt'."
        )
    return records


def render_index(state: AppState) -> bytes:
    payload = {
        "records": [
            {
                "image_name": r.image_name,
                "image_url": r.image_url,
                "response_text": r.response_text,
                "expected_intersects": r.expected_intersects,
                "predicted_intersects": r.predicted_intersects,
                "correct": r.correct,
            }
            for r in state.records
        ],
        "global_metrics": compute_global_metrics(state.records),
    }

    safe_json = json.dumps(payload).replace("</", "<\\/")

    page = state.template_html
    page = page.replace("__INLINE_CSS__", state.css)
    page = page.replace("__INLINE_JS__", state.js)
    page = page.replace("__DATA_JSON__", safe_json)
    return page.encode("utf-8")


def build_handler(state: AppState):
    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                body = render_index(state)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path.startswith("/image/"):
                image_name = self.path[len("/image/"):]
                if not image_name:
                    self.send_error(404)
                    return

                from urllib.parse import unquote

                image_name = unquote(image_name)
                image_path = (state.image_dir / image_name).resolve()
                image_root = state.image_dir.resolve()
                if image_root not in image_path.parents or not image_path.is_file():
                    self.send_error(404)
                    return

                try:
                    data = image_path.read_bytes()
                except OSError:
                    self.send_error(500)
                    return

                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            self.send_error(404)

        def log_message(self, _format: str, *_args) -> None:
            return

    return ViewerHandler


def main() -> None:
    args = parse_args()
    if args.port < 1 or args.port > 65535:
        raise SystemExit("--port must be in range 1-65535")

    base_dir = Path(__file__).resolve().parent
    template_html, css, js = load_template_assets(base_dir)
    records = build_records(args.response_cache, args.image_dir)

    state = AppState(
        image_dir=args.image_dir,
        records=records,
        template_html=template_html,
        css=css,
        js=js,
    )

    handler = build_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {len(records)} items at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
