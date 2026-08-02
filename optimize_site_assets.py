#!/usr/bin/env python3
"""Create right-sized WebP display assets and rewrite site references."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None


ROOT = Path(__file__).resolve().parents[0]
IMAGE_ROOT = ROOT / "images"
MIN_SOURCE_BYTES = 64 * 1024
MIN_SAVINGS_RATIO = 0.95
TEXT_ROOTS = [
    ROOT / "index.html",
    ROOT / "blog.html",
    ROOT / "blog",
    ROOT / "assets/css",
    ROOT / "assets/js",
]
IMG_TAG = re.compile(r"<img\b[^>]*>", re.IGNORECASE | re.DOTALL)
SRC_ATTRIBUTE = re.compile(r'\bsrc=["\']([^"\']+)["\']', re.IGNORECASE)
MAGICK = shutil.which("magick")


def target_size(path: Path, width: int, height: int) -> tuple[int, int]:
    relative = path.relative_to(IMAGE_ROOT).as_posix()
    if relative.startswith("demo/tennis-serve/frames/"):
        max_dimension = 1280
    elif "/" not in relative:
        max_dimension = 1600
    else:
        max_dimension = 1200
    scale = min(1.0, max_dimension / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def image_size(path: Path) -> tuple[int, int]:
    if Image is not None:
        with Image.open(path) as image:
            return image.width, image.height
    if not MAGICK:
        raise RuntimeError("Install Pillow or ImageMagick to optimize images.")
    output = subprocess.check_output(
        [MAGICK, "identify", "-format", "%w %h", str(path)],
        text=True,
    )
    width, height = output.split()
    return int(width), int(height)


def save_optimized(source: Path, output: Path) -> None:
    width, height = image_size(source)
    target_width, target_height = target_size(source, width, height)
    if Image is not None:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)  # type: ignore
            image.thumbnail(
                (target_width, target_height),
                Image.Resampling.LANCZOS,
            )
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGBA" if "transparency" in image.info else "RGB")
            image.save(
                output,
                "WEBP",
                quality=86,
                method=6,
                exact=image.mode == "RGBA",
            )
        return

    subprocess.run(  # type: ignore
        [
            MAGICK,  # type: ignore
            str(source),
            "-auto-orient",
            "-resize",
            f"{target_width}x{target_height}>",
            "-define",
            "webp:method=6",
            "-define",
            "webp:exact=true",
            "-quality",
            "86",
            str(output),
        ],
        check=True,
    )


def text_files() -> list[Path]:
    files: list[Path] = []
    for entry in TEXT_ROOTS:
        if entry.is_file():
            files.append(entry)
        elif entry.is_dir():
            files.extend(
                path
                for path in entry.rglob("*")
                if path.suffix.lower() in {".html", ".css", ".js", ".json"}
            )
    return files


def add_image_loading_hints(path: Path, text: str) -> str:
    def update(match: re.Match[str]) -> str:
        tag = match.group(0)
        source_match = SRC_ATTRIBUTE.search(tag)
        if not source_match:
            return tag
        source = source_match.group(1)
        if source.startswith(("http://", "https://", "data:")):
            return tag

        eager = "post-cover" in tag or source.endswith("images/profile.jpg")
        additions: list[str] = []
        if " loading=" not in tag:
            additions.append('loading="eager"' if eager else 'loading="lazy"')
        if " decoding=" not in tag:
            additions.append('decoding="async"')
        if eager and " fetchpriority=" not in tag:
            additions.append('fetchpriority="high"')

        local = (
            ROOT / source.lstrip("/")
            if source.startswith("/")
            else path.parent / source
        )
        if local.is_file() and " width=" not in tag and " height=" not in tag:
            try:
                width, height = image_size(local)
                additions.extend([f'width="{width}"', f'height="{height}"'])
            except (OSError, RuntimeError, subprocess.SubprocessError):
                pass
        if not additions:
            return tag
        return tag.replace("<img", "<img " + " ".join(additions), 1)

    return IMG_TAG.sub(update, text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prune-originals",
        action="store_true",
        help="Remove converted originals after confirming they are unreferenced.",
    )
    args = parser.parse_args()
    replacements: dict[str, str] = {}
    converted_sources: list[Path] = []
    before = 0
    after = 0
    converted = 0

    sources = sorted(
        path
        for path in IMAGE_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and path.stat().st_size >= MIN_SOURCE_BYTES
    )
    for source in sources:
        output = source.with_suffix(".webp")
        save_optimized(source, output)

        source_size = source.stat().st_size
        output_size = output.stat().st_size
        if output_size >= source_size * MIN_SAVINGS_RATIO:
            output.unlink()
            continue

        relative_source = source.relative_to(ROOT).as_posix()
        relative_output = output.relative_to(ROOT).as_posix()
        replacements[relative_source] = relative_output
        replacements[source.name] = output.name
        before += source_size
        after += output_size
        converted += 1
        converted_sources.append(source)

    changed_files = 0
    for path in text_files():
        with path.open("r", encoding="utf-8", newline="") as stream:
            original = stream.read()
        updated = original
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if path.suffix.lower() == ".html":
            updated = add_image_loading_hints(path, updated)
        if updated != original:
            with path.open("w", encoding="utf-8", newline="") as stream:
                stream.write(updated)
            changed_files += 1

    pruned = 0
    if args.prune_originals:
        corpus_parts: list[str] = []
        for path in text_files():
            with path.open("r", encoding="utf-8", newline="") as stream:
                corpus_parts.append(stream.read())
        corpus = "\n".join(corpus_parts)
        for source in converted_sources:
            relative = source.relative_to(ROOT).as_posix()
            if any(
                reference in corpus
                for reference in (relative, f"/{relative}", source.name)
            ):
                continue
            source.unlink()
            pruned += 1

    message = (
        f"Converted {converted} assets and updated {changed_files} files: "
        f"{before / 1024 / 1024:.1f} MB -> {after / 1024 / 1024:.1f} MB."
    )
    if args.prune_originals:
        message += f" Removed {pruned} unreferenced originals."
    print(message)


if __name__ == "__main__":
    main()
