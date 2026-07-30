#!/usr/bin/env python3
"""Build the compact audio and spectrogram assets used by the UW reflection demo."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


ROOT = Path(__file__).resolve().parents[3]
KOEL_STATIC = Path("/Users/alex/Desktop/CS/Startups/Koel/app/server/src/static")
TIMESTAMPS_PATH = Path(
    "/Users/alex/Desktop/CS/Startups/Koel/app/server/src/"
    "phraseReferenceTimestamps.json"
)
OUTPUT_ROOT = ROOT / "images/demo/spectrogram-tutorial"
PRACTICE_DIR = OUTPUT_ROOT / "spectrograms"
PHONEME_DIR = OUTPUT_ROOT / "phonemes"
SOUND_IMAGE_DIR = OUTPUT_ROOT / "sound-spectrograms"
COMPARISON_DIR = OUTPUT_ROOT / "comparison"
DATA_PATH = ROOT / "assets/js/spectrogram-tutorial-data.js"

SAMPLE_RATE = 16_000
HOP_SIZE = 128
MEL_BINS = 96
TARGET_LUFS = -24

EXERCISES = [
    ("Can I have a bag, please?", "bag"),
    ("Can I have water, please?", "water"),
    ("Can I have a quiet room, please?", "quiet"),
    ("Can we check the pronunciation?", "pronunciation"),
    ("Can you help me with my schedule?", "schedule"),
    ("Can we compare the two options?", "options"),
    ("Can we listen one more time?", "listen"),
    ("Can we read it aloud?", "aloud"),
    ("I think the context changes the meaning.", "context"),
    ("Can we break it into parts?", "parts"),
    ("Can we focus on the sound?", "sound"),
    ("Can we check the stress?", "stress"),
    ("Can we make it more natural?", "natural"),
    ("Can I have a receipt, please?", "receipt"),
    ("Can I have the address, please?", "address"),
    ("Can I have coffee, please?", "coffee"),
    ("As a next step, I would recommend a clearer next step.", "next"),
    (
        "Happy Cinco de Mayo! I hope today brings good food and good company.",
        "good",
    ),
]

STOP_WORDS = {
    "a", "an", "and", "as", "at", "can", "do", "have", "i", "in", "is",
    "it", "me", "more", "my", "of", "on", "please", "the", "to", "we",
    "with", "would", "you",
}


def hz_to_mel(frequency: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def decode_audio(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-ac", "1",
            "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype="<f4")


def frame_audio(
    audio: np.ndarray,
    window_size: int,
    hop_size: int,
) -> np.ndarray:
    if len(audio) < window_size:
        audio = np.pad(audio, (0, window_size - len(audio)))
    frame_count = 1 + (len(audio) - window_size) // hop_size
    return np.lib.stride_tricks.as_strided(
        audio,
        shape=(frame_count, window_size),
        strides=(audio.strides[0] * hop_size, audio.strides[0]),
    ).copy()


def stft_power(
    audio: np.ndarray,
    window_size: int,
    hop_size: int,
    fft_size: int,
) -> np.ndarray:
    frames = frame_audio(audio, window_size, hop_size)
    frames *= np.hanning(window_size).astype(np.float32)
    spectrum = np.fft.rfft(frames, n=fft_size, axis=1)
    return (np.abs(spectrum) ** 2).T


def mel_filterbank(fft_size: int) -> np.ndarray:
    low_mel = hz_to_mel(50.0)
    high_mel = hz_to_mel(SAMPLE_RATE / 2)
    mel_points = np.linspace(low_mel, high_mel, MEL_BINS + 2)
    bins = np.floor(
        (fft_size + 1) * mel_to_hz(mel_points) / SAMPLE_RATE
    ).astype(int)
    filters = np.zeros((MEL_BINS, fft_size // 2 + 1), dtype=np.float32)
    for index in range(MEL_BINS):
        left, center, right = bins[index : index + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for frequency_bin in range(left, min(center, filters.shape[1])):
            filters[index, frequency_bin] = (
                frequency_bin - left
            ) / (center - left)
        for frequency_bin in range(center, min(right, filters.shape[1])):
            filters[index, frequency_bin] = (
                right - frequency_bin
            ) / (right - center)
    return filters


def normalize_db(power: np.ndarray, dynamic_range: float = 76.0) -> np.ndarray:
    decibels = 10.0 * np.log10(np.maximum(power, 1e-10))
    upper = np.percentile(decibels, 99.6)
    lower = max(np.percentile(decibels, 5.0), upper - dynamic_range)
    return np.clip((decibels - lower) / max(upper - lower, 1e-6), 0.0, 1.0)


def log_mel(audio: np.ndarray) -> np.ndarray:
    fft_size = 512
    power = stft_power(audio, window_size=400, hop_size=160, fft_size=fft_size)
    return normalize_db(mel_filterbank(fft_size) @ power, dynamic_range=72.0)


def linear_spectrogram(
    audio: np.ndarray,
    window_size: int,
    fft_size: int,
    dynamic_range: float = 72.0,
) -> np.ndarray:
    power = stft_power(
        audio,
        window_size=window_size,
        hop_size=HOP_SIZE,
        fft_size=fft_size,
    )
    return normalize_db(power, dynamic_range=dynamic_range)


def save_grayscale(
    values: np.ndarray,
    output_path: Path,
    size: tuple[int, int],
) -> None:
    darkness = np.power(np.clip(values, 0.0, 1.0), 0.72)
    grayscale = np.flipud(np.round((1.0 - darkness) * 255.0).astype(np.uint8))
    image = Image.fromarray(grayscale, mode="L")
    image = image.resize(size, Image.Resampling.BICUBIC)
    image = ImageOps.autocontrast(image, cutoff=(0.4, 0.4))
    image = ImageEnhance.Contrast(image).enhance(1.08)
    image.save(output_path, "WEBP", lossless=True, method=6)


def save_log_mel(audio_path: Path, output_path: Path) -> float:
    audio = decode_audio(audio_path)
    save_grayscale(log_mel(audio), output_path, (960, 300))
    return len(audio) / SAMPLE_RATE


def save_linear(
    audio_path: Path,
    output_path: Path,
    window_size: int = 160,
    fft_size: int = 1024,
    size: tuple[int, int] = (760, 300),
) -> None:
    audio = decode_audio(audio_path)
    save_grayscale(
        linear_spectrogram(audio, window_size, fft_size),
        output_path,
        size,
    )


def slugify(text: str) -> str:
    slug = "".join(
        character.lower() if character.isalnum() else "-" for character in text
    )
    return "-".join(part for part in slug.split("-") if part)


def normalize_word(word: str) -> str:
    return word.lower().strip(".,!?;:'\"")


def candidate_regions(
    timestamps: list[dict[str, float | str]],
    target: str,
    duration: float,
) -> list[dict[str, float | bool]]:
    normalized_target = normalize_word(target)
    targets = [
        (index, word)
        for index, word in enumerate(timestamps)
        if normalize_word(str(word["word"])) == normalized_target
    ]
    decoys = [
        (index, word)
        for index, word in enumerate(timestamps)
        if normalize_word(str(word["word"])) != normalized_target
        and normalize_word(str(word["word"])) not in STOP_WORDS
        and len(normalize_word(str(word["word"]))) >= 3
    ]
    if len(decoys) < 3:
        decoys = [
            (index, word)
            for index, word in enumerate(timestamps)
            if normalize_word(str(word["word"])) != normalized_target
            and len(normalize_word(str(word["word"]))) >= 2
        ]
    desired_decoys = max(2, 5 - len(targets))
    if len(decoys) > desired_decoys:
        picks = np.linspace(0, len(decoys) - 1, desired_decoys).round().astype(int)
        decoys = [decoys[index] for index in picks]
    selected = targets + decoys[:desired_decoys]
    selected.sort(key=lambda item: float(item[1]["start"]))
    regions = [
        {
            "start": max(0.0, float(word["start"]) / duration),
            "end": min(1.0, float(word["end"]) / duration),
            "correct": normalize_word(str(word["word"])) == normalized_target,
        }
        for _, word in selected
    ]
    for previous, current in zip(regions, regions[1:]):
        if previous["end"] > current["start"]:
            boundary = (float(previous["end"]) + float(current["start"])) / 2
            previous["end"] = boundary
            current["start"] = boundary
    return [
        {
            "start": round(float(region["start"]), 5),
            "end": round(float(region["end"]), 5),
            "correct": bool(region["correct"]),
        }
        for region in regions
    ]


def measure_lufs(path: Path) -> float:
    measurement = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-filter_complex",
            "ebur128",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    matches = re.findall(
        r"I:\s+(-?(?:\d+(?:\.\d+)?|inf)) LUFS",
        measurement.stderr,
    )
    if not matches or matches[-1] == "-inf":
        raise RuntimeError(f"Could not measure loudness for {path}")
    return float(matches[-1])


def normalize_audio(
    source: Path,
    output: Path,
    input_options: list[str] | None = None,
) -> None:
    intermediate = output.with_name(output.stem + ".loudnorm.wav")
    codec_options = (
        ["-c:a", "libmp3lame", "-b:a", "128k"]
        if output.suffix.lower() == ".mp3"
        else ["-c:a", "aac", "-b:a", "96k"]
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            *(input_options or []),
            "-i",
            str(source),
            "-af",
            f"loudnorm=I={TARGET_LUFS}:LRA=7:TP=-2",
            "-c:a",
            "pcm_s16le",
            str(intermediate),
        ],
        check=True,
    )
    gain = TARGET_LUFS - measure_lufs(intermediate)
    best_output = output.with_name(output.stem + ".best" + output.suffix)
    best_error = float("inf")
    for _ in range(8):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(intermediate),
                "-af",
                f"volume={gain:.2f}dB",
                *codec_options,
                str(output),
            ],
            check=True,
        )
        correction = TARGET_LUFS - measure_lufs(output)
        if abs(correction) < best_error:
            best_error = abs(correction)
            shutil.copy2(output, best_output)
        if best_error <= 0.05:
            break
        gain += correction * 0.5
    best_output.replace(output)
    intermediate.unlink()


def build_audio_examples() -> None:
    for source in sorted((KOEL_STATIC / "phoneme-audios").glob("*.m4a")):
        normalize_audio(source, PHONEME_DIR / source.name)

    word_examples = {
        "foot.mp3": "word-reference-audios/yiwepj-foot.mp3",
        "about.mp3": "word-reference-audios/2t22p4-about.mp3",
    }
    for output_name, relative_source in word_examples.items():
        normalize_audio(
            KOEL_STATIC / relative_source,
            PHONEME_DIR / output_name,
        )

    tap_source = (
        KOEL_STATIC / "reference-audios/0075_he_fell_into_the_hot_water.mp3"
    )
    normalize_audio(
        tap_source,
        PHONEME_DIR / "tap.m4a",
        ["-ss", "4.02", "-t", "1.34"],
    )

    for source in sorted(PHONEME_DIR.glob("*.*")):
        if source.suffix.lower() not in {".m4a", ".mp3"}:
            continue
        save_linear(
            source,
            SOUND_IMAGE_DIR / f"{source.stem}.webp",
            window_size=160,
            fft_size=1024,
        )


def build_comparison(timestamp_data: dict[str, dict]) -> None:
    sentence = "Can we check the pronunciation?"
    relative_audio = next(
        path for path, entry in timestamp_data.items() if entry["text"] == sentence
    )
    source = KOEL_STATIC / relative_audio
    save_linear(
        source,
        COMPARISON_DIR / "pronunciation-narrowband.webp",
        window_size=800,
        fft_size=2048,
        size=(820, 330),
    )
    save_linear(
        source,
        COMPARISON_DIR / "pronunciation-wideband.webp",
        window_size=160,
        fft_size=1024,
        size=(820, 330),
    )


def main() -> None:
    for directory in (
        PRACTICE_DIR,
        PHONEME_DIR,
        SOUND_IMAGE_DIR,
        COMPARISON_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    build_audio_examples()
    timestamp_data = json.loads(TIMESTAMPS_PATH.read_text())
    build_comparison(timestamp_data)
    by_text = {
        entry["text"]: (path, entry) for path, entry in timestamp_data.items()
    }
    output = []

    for sentence, target in EXERCISES:
        if sentence not in by_text:
            raise KeyError(f"Missing timestamp entry: {sentence}")
        relative_audio, entry = by_text[sentence]
        source = KOEL_STATIC / relative_audio
        image_name = f"{slugify(target)}-{len(output) + 1:02d}.webp"
        duration = save_log_mel(source, PRACTICE_DIR / image_name)
        highlights = [
            {
                "start": round(float(word["start"]) / duration, 5),
                "end": round(float(word["end"]) / duration, 5),
            }
            for word in entry["timestamps"]
            if normalize_word(str(word["word"])) == normalize_word(target)
        ]
        output.append(
            {
                "sentence": sentence,
                "answer": target,
                "image": (
                    "/images/demo/spectrogram-tutorial/spectrograms/"
                    f"{image_name}"
                ),
                "highlights": highlights,
                "candidates": candidate_regions(
                    entry["timestamps"],
                    target,
                    duration,
                ),
            }
        )

    payload = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
    DATA_PATH.write_text(
        "window.SPECTROGRAM_EXERCISES=" + payload + ";\n",
        encoding="utf-8",
    )
    image_size = sum(
        path.stat().st_size
        for directory in (PRACTICE_DIR, SOUND_IMAGE_DIR, COMPARISON_DIR)
        for path in directory.glob("*.webp")
    )
    print(
        f"Built {len(output)} exercises, "
        f"{len(list(PHONEME_DIR.glob('*.*')))} audio clips, "
        f"{len(list(SOUND_IMAGE_DIR.glob('*.webp')))} sound spectrograms, "
        f"and {image_size // 1024} KB of images."
    )


if __name__ == "__main__":
    main()
