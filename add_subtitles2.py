"""Generate subtitles for MP4 files and optionally burn them into the videos."""

import argparse
import concurrent.futures
from pathlib import Path
import subprocess
import whisper


def format_timestamp(seconds: float) -> str:
    """Return an SRT timestamp (``HH:MM:SS,mmm``) for ``seconds``."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}".replace(".", ",")


def _escape_for_ffmpeg(path: Path) -> str:
    """Escape ``path`` for ffmpeg's subtitles filter."""
    return path.as_posix().replace(":", r"\:")

def transcribe(video_path: Path, model_name: str = "small") -> Path:
    model = whisper.load_model(model_name)
    result = model.transcribe(str(video_path), verbose=False)
    srt_path = video_path.with_suffix(".srt")
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for idx, segment in enumerate(result["segments"], start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            srt_file.write(f"{idx}\n{start} --> {end}\n{text}\n\n")
    return srt_path

def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path) -> None:
    subtitle_filter = f"subtitles={_escape_for_ffmpeg(srt_path)}"
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            subtitle_filter,
            "-c:a",
            "copy",
            str(output_path),
        ],
        check=True,
    )


def get_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate subtitles for MP4 files and optionally burn them into the videos."
    )
    parser.add_argument(
        "input", help="Path to an MP4 file or a directory containing MP4 files"
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model size (tiny, base, small, medium, large)",
    )
    parser.add_argument(
        "--burn",
        action="store_true",
        help="Burn subtitles into the video using ffmpeg",
    )
    args = parser.parse_args()

    target = Path(args.input).expanduser()
    if target.is_dir():
        videos = sorted(target.glob("*.mp4"), key=get_duration)
    else:
        videos = [target]
        target = target.parent

    if not videos:
        print("No MP4 files found.")
        return

    output_dir = target / "字幕付き2"
    output_dir.mkdir(exist_ok=True)

    def process(video_path: Path) -> None:
        srt_path = transcribe(video_path, args.model)
        if args.burn:
            output_path = output_dir / video_path.name
            burn_subtitles(video_path, srt_path, output_path)
            print(f"Subtitled video saved to {output_path}")
        else:
            dest = output_dir / srt_path.name
            srt_path.replace(dest)
            print(f"SRT subtitles saved to {dest}")

    first, *rest = videos
    try:
        process(first)
    except Exception as exc:
        print(f"Error processing {first}: {exc}")
        return

    if rest:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process, v) for v in rest]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"Error: {exc}")

if __name__ == '__main__':
    main()
