#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from functools import cache

from tqdm import tqdm
from PIL import Image

FILENAME_RE = re.compile(r"^intersect_([YN])_distractor_([YN])_(.+)\.png$", re.IGNORECASE)
YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


@dataclass
class ImageDatum:
    path: Path
    intersects: bool
    has_distractor: bool


@dataclass
class Predictor:
    model: object
    processor: object
    formatted_prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute intersection detection accuracy."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing PNG files."
    )
    parser.add_argument(
        "--api-key-file",
        default=Path("apikey.txt"),
        type=Path,
        help="File containing Anthropic API key (default: apikey.txt).",
    )
    parser.add_argument(
        "--prompt-file",
        default=Path("prompt.txt"),
        type=Path,
        help="Prompt file used for each request (default: prompt.txt).",
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        help="Number of worker threads (default: 4).",
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model name.",
    )
    parser.add_argument(
        "--max-tokens",
        default=2048,
        type=int,
        help="Max tokens for each Claude response (default: 2048).",
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="Use the local model predictor instead of Anthropic API.",
    )
    return parser.parse_args()


DEFAULT_MODEL_CHECKPOINT = ''


@cache
def get_local_predictor(prompt: str, checkpoint: str = DEFAULT_MODEL_CHECKPOINT) -> Predictor:
    from transformers import AutoModelForCausalLM, AutoProcessor
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        'microsoft/Phi-3.5-vision-instruct',
        device_map='cuda',
        trust_remote_code=True,
        torch_dtype='auto',
        _attn_implementation='flash_attention_2'
    )
    processor = AutoProcessor.from_pretrained(
        'microsoft/Phi-3.5-vision-instruct',
        trust_remote_code=True,
        num_crops=1
    )
    formatted_prompt = processor.tokenizer.apply_chat_template(
        [
            {
                'role': 'user',
                'content': f'<|image_1|>\n{prompt}'
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    return Predictor(
        model=model,
        processor=processor,
        formatted_prompt=formatted_prompt
    )


def predict_using_local_model(image_path: Path, prompt: str) -> str:
    import torch

    predictor = get_local_predictor(prompt)

    inputs = predictor.processor(
        images=[Image.open(image_path)],
        text=predictor.formatted_prompt,
        return_tensors='pt'
    ).to(predictor.model.device)

    with torch.no_grad():
        outputs = predictor.model(**inputs)

    return predictor.processor.tokenizer.decode(outputs.logits[0, -1].argmax()).strip()


def read_text_file(path: Path, desc: str) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise SystemExit(f"{desc} not found: {path}") from exc
    if not text:
        raise SystemExit(f"{desc} is empty: {path}")
    return text


def parse_image_datum(path: Path) -> ImageDatum:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Filename does not match expected format: {path.name}")
    intersects_flag, distractor_flag, _suffix = match.groups()
    return ImageDatum(
        path=path,
        intersects=intersects_flag.upper() == "Y",
        has_distractor=distractor_flag.upper() == "Y",
    )


def discover_images(input_dir: Path) -> list[ImageDatum]:
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise SystemExit(f"Input path is not a directory: {input_dir}")

    data: list[ImageDatum] = []
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".png":
            data.append(parse_image_datum(path))
    if not data:
        raise SystemExit(f"No PNG files found in: {input_dir}")
    return data


def extract_text_from_anthropic_response(payload: dict) -> str:
    blocks = payload.get("content", [])
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def parse_yes_no(text: str) -> Optional[bool]:
    match = YES_NO_RE.search(text)
    if not match:
        return None
    return match.group(1).lower() == "yes"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def cache_file_for(
        base: Path,
        prompt_sha: str,
        image_path: Path,
        model: str
) -> Path:
    return base / "response_cache" / image_path.parent.name / model / prompt_sha / f"{image_path.name}.txt"


def call_anthropic(
    api_key: str,
    model: str,
    prompt: str,
    image_path: Path,
    max_tokens: int,
) -> str:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    }

    req = urllib.request.Request(
        url="https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic HTTP {exc.code} for {image_path.name}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error for {image_path.name}: {exc}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from Anthropic for {image_path.name}") from exc

    text = extract_text_from_anthropic_response(payload)
    if not text:
        raise RuntimeError(f"No text content in Anthropic response for {image_path.name}")
    return text


def query_with_cache(
    datum: ImageDatum,
    *,
    cwd: Path,
    prompt_sha: str,
    prompt: str,
    api_key: Optional[str],
    model: str,
    max_tokens: int,
    use_local: bool,
    lock: threading.Lock,
) -> str:
    cache_file = cache_file_for(cwd, prompt_sha, datum.path, model)
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8").strip()

    if use_local:
        response_text = predict_using_local_model(datum.path, prompt)
    else:
        if not api_key:
            raise RuntimeError("API key is required when not using --use-local")
        response_text = call_anthropic(
            api_key=api_key,
            model=model,
            prompt=prompt,
            image_path=datum.path,
            max_tokens=max_tokens,
        )

    with lock:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Double-check in case another worker wrote it while this request was in flight.
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8").strip()
        cache_file.write_text(response_text, encoding="utf-8")
    return response_text


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be >= 1")
    workers = 1 if args.use_local else args.workers

    cwd = Path.cwd()
    api_key = None if args.use_local else read_text_file(args.api_key_file, "API key file")
    prompt = read_text_file(args.prompt_file, "Prompt file")
    prompt_sha = prompt_hash(prompt)
    data = discover_images(args.input_dir)

    lock = threading.Lock()
    results: list[tuple[ImageDatum, str]] = []
    errors: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_datum = {
            executor.submit(
                query_with_cache,
                datum,
                cwd=cwd,
                prompt_sha=prompt_sha,
                prompt=prompt,
                api_key=api_key,
                model=args.model if not args.use_local else "local",
                max_tokens=args.max_tokens,
                use_local=args.use_local,
                lock=lock,
            ): datum
            for datum in data
        }
        progress = tqdm(
            concurrent.futures.as_completed(future_to_datum),
            total=len(future_to_datum),
            desc="Processing images",
            unit="img",
        )
        for future in progress:
            datum = future_to_datum[future]
            try:
                response_text = future.result()
                results.append((datum, response_text))
            except Exception as exc:
                errors.append(f"{datum.path.name}: {exc}")

    correct = 0
    total = len(data)
    unparsable = 0

    for datum, response_text in results:
        predicted = parse_yes_no(response_text)
        if predicted is None:
            unparsable += 1
            continue
        if predicted == datum.intersects:
            correct += 1

    accuracy = (correct / total) if total else 0.0
    print(f"Total images: {total}")
    print(f"Successful responses: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Unparsable Yes/No: {unparsable}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")

    if errors:
        print("\nRequest/processing errors:")
        for err in errors:
            print(f"- {err}")


if __name__ == "__main__":
    main()
