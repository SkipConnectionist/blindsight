from dataclasses import dataclass
from typing import TypedDict
from pathlib import Path
from PIL import Image

import re


@dataclass
class ItersectionItem:
    path: Path
    intersects: bool
    has_distractor: bool
    image: Image.Image | None = None


class IntersectionDataDict(TypedDict):
    image: Image.Image
    intersects: bool


FILENAME_RE = re.compile(r"^intersect_([YN])_distractor_([YN])_(.+)\.png$", re.IGNORECASE)


def parse_intersection_item(path: Path) -> ItersectionItem:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Filename does not match expected format: {path.name}")
    intersects_flag, distractor_flag, _suffix = match.groups()
    return ItersectionItem(
        path=path,
        intersects=intersects_flag.upper() == "Y",
        has_distractor=distractor_flag.upper() == "Y",
    )


class IntersectionDataset:

    def __init__(self, image_dir: Path):
        self.image_dir = image_dir
        self.data = [
            parse_intersection_item(p)
            for p in self.image_dir.glob("*.png")
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> IntersectionDataDict:
        item = self.data[idx]
        if item.image is None:
            item.image = Image.open(item.path)
        return {
            "image": item.image,
            "intersects": item.intersects,
            "answer": "Yes" if item.intersects else "No"
        }
