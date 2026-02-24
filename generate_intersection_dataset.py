#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import uuid
from pathlib import Path
import matplotlib.pyplot as plt

import intersection


def _yn(value):
    return "Y" if bool(value) else "N"


def _get_flag(config, key):
    if isinstance(config, dict):
        return bool(config.get(key, False))
    return bool(getattr(config, key, False))


def _generate_one(output_dir):
    params = intersection.gen_random_plot_params()
    fig = intersection.render_plot(params, show=False)

    is_intersect = _get_flag(params, "intersect")
    has_distractor = _get_flag(params, "distractor")
    suffix = uuid.uuid4().hex[:10]

    filename = (
        f"intersect_{_yn(is_intersect)}_distractor_{_yn(has_distractor)}_{suffix}.png"
    )
    out_path = output_dir / filename

    fig.savefig(out_path, format="png", dpi=72, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate intersection dataset.")
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where generated PNG files will be written.",
    )
    parser.add_argument(
        "--count",
        required=True,
        type=int,
        help="Number of PNG files to generate.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes to use (default: 4).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.count < 1:
        raise ValueError("count must be >= 1")
    if args.workers < 1:
        raise ValueError("workers must be >= 1")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with mp.Pool(processes=args.workers) as pool:
        for _ in pool.imap_unordered(
            _generate_one, [output_dir] * args.count, chunksize=1
        ):
            pass


if __name__ == "__main__":
    main()
