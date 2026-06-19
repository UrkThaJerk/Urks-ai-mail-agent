import json
import logging
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


SUPPORTED_VIDEO_FORMATS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
ECLIPSE_PROTOTYPE_REFERENCE = "eclipse.gg"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)


class VideoEditingError(Exception):
    pass


@dataclass
class VideoAsset:
    path: str
    format: str
    duration_seconds: float = 0.0
    width: int | None = None
    height: int | None = None
    video_codec: str | None = None
    audio_codec: str | None = None


@dataclass
class TimelineSegment:
    start_seconds: float
    end_seconds: float
    label: str
    transition: str | None = None
    key_moment: bool = False


@dataclass
class Keyframe:
    time_seconds: float
    property_name: str
    value: str


@dataclass
class EditOperation:
    operation: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoProject:
    source: VideoAsset
    instructions: str = ""
    scenes: list[TimelineSegment] = field(default_factory=list)
    operations: list[EditOperation] = field(default_factory=list)
    keyframes: list[Keyframe] = field(default_factory=list)
    export_path: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "prototype_reference": ECLIPSE_PROTOTYPE_REFERENCE,
            "source": asdict(self.source),
            "instructions": self.instructions,
            "scene_count": len(self.scenes),
            "scenes": [asdict(scene) for scene in self.scenes],
            "operations": [asdict(operation) for operation in self.operations],
            "keyframes": [asdict(keyframe) for keyframe in self.keyframes],
            "export_path": self.export_path,
        }


def _run_media_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise VideoEditingError(f"Required media tool is unavailable: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or "Media command failed."
        raise VideoEditingError(message) from exc


def download_video(url: str, output_dir: str | None = None) -> str:
    """Download a video from a URL (including Twitch VODs) using yt-dlp.

    Returns the local path to the downloaded file.
    """
    import yt_dlp  # imported here so the rest of the module works without yt-dlp installed

    dest_dir = Path(output_dir).expanduser().resolve() if output_dir else Path.cwd()
    dest_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts: dict[str, Any] = {
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_path = ydl.prepare_filename(info)
    except Exception as exc:
        raise VideoEditingError(f"Failed to download video from '{url}': {exc}") from exc

    path = Path(downloaded_path)
    if not path.exists():
        # yt-dlp may merge containers and change extension
        for candidate in sorted(path.parent.glob(f"{path.stem}.*")):
            if candidate.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
                return str(candidate)
        raise VideoEditingError(f"Downloaded file not found at expected path: {path}")

    return str(path)


def extract_clip(source_path: str, start_seconds: float, end_seconds: float, output_path: str) -> str:
    """Extract a time-bounded clip from *source_path* into *output_path* using ffmpeg.

    Returns the output path on success.
    """
    if end_seconds <= start_seconds:
        raise VideoEditingError(
            f"end_seconds ({end_seconds}) must be greater than start_seconds ({start_seconds})"
        )

    duration = end_seconds - start_seconds
    dest = Path(output_path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    _run_media_command(
        [
            "ffmpeg",
            "-y",
            "-ss", str(start_seconds),
            "-i", source_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(dest),
        ]
    )
    return str(dest)


def probe_video(video_path: str) -> VideoAsset:
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        raise VideoEditingError(
            f"Unsupported video format '{path.suffix}'. Supported formats: {', '.join(sorted(SUPPORTED_VIDEO_FORMATS))}"
        )

    try:
        result = _run_media_command(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ]
        )
        payload = json.loads(result.stdout or "{}")
    except (VideoEditingError, json.JSONDecodeError):
        LOGGER.warning("Falling back to file-only metadata for %s", path)
        return VideoAsset(path=str(path), format=path.suffix.lower().lstrip("."))

    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
    format_info = payload.get("format", {})

    duration_value = format_info.get("duration") or video_stream.get("duration") or 0.0
    return VideoAsset(
        path=str(path),
        format=path.suffix.lower().lstrip("."),
        duration_seconds=float(duration_value or 0.0),
        width=video_stream.get("width"),
        height=video_stream.get("height"),
        video_codec=video_stream.get("codec_name"),
        audio_codec=audio_stream.get("codec_name"),
    )


def detect_scene_segments(video: VideoAsset, threshold: float = 0.35) -> list[TimelineSegment]:
    if video.duration_seconds <= 0:
        return [TimelineSegment(start_seconds=0.0, end_seconds=0.0, label="scene_1", key_moment=True)]

    try:
        result = _run_media_command(
            [
                "ffmpeg",
                "-hide_banner",
                "-i",
                video.path,
                "-filter:v",
                f"select='gt(scene,{threshold})',showinfo",
                "-an",
                "-f",
                "null",
                "-",
            ]
        )
        raw_output = result.stderr or result.stdout
    except VideoEditingError:
        raw_output = ""

    boundaries = [0.0]
    for match in re.finditer(r"pts_time:(\d+(?:\.\d+)?)", raw_output):
        boundaries.append(float(match.group(1)))
    boundaries.append(video.duration_seconds)
    boundaries = sorted({round(boundary, 3) for boundary in boundaries})

    scenes: list[TimelineSegment] = []
    for index in range(len(boundaries) - 1):
        start_seconds = boundaries[index]
        end_seconds = boundaries[index + 1]
        if end_seconds <= start_seconds:
            continue
        scenes.append(
            TimelineSegment(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                label=f"scene_{index + 1}",
                transition="cut" if index else None,
                key_moment=index == 0 or index == len(boundaries) - 2 or (end_seconds - start_seconds) >= 5,
            )
        )

    return scenes or [TimelineSegment(start_seconds=0.0, end_seconds=video.duration_seconds, label="scene_1", key_moment=True)]


def parse_edit_instructions(instructions: str) -> list[EditOperation]:
    lowered = instructions.lower()
    operations: list[EditOperation] = []

    if any(phrase in lowered for phrase in ("detect scenes", "scene detection", "auto-segment", "auto segment")):
        operations.append(EditOperation("detect_scenes"))

    if any(phrase in lowered for phrase in ("make clips", "extract clips", "create clips", "split clips")):
        operations.append(EditOperation("extract_clips"))

    effect_keywords = {
        "black and white": "grayscale",
        "grayscale": "grayscale",
        "sepia": "sepia",
        "vintage": "sepia",
        "cinematic": "cinematic",
        "color correction": "cinematic",
    }
    for keyword, effect_name in effect_keywords.items():
        if keyword in lowered:
            operations.append(EditOperation("apply_effect", {"name": effect_name}))

    transition_match = re.search(r"(fade|dissolve|wipe|cut)\s+transition", lowered)
    if transition_match:
        operations.append(EditOperation("apply_transition", {"name": transition_match.group(1)}))

    text_match = re.search(r'(?:text overlay|add text)\s+["\']([^"\']+)["\']', instructions, re.IGNORECASE)
    if text_match:
        operations.append(EditOperation("add_text_overlay", {"text": text_match.group(1), **_extract_timing(instructions)}))

    watermark_match = re.search(r'(?:watermark(?: with)?)\s+["\']?([^"\']+?)["\']?(?:\s+at\s+\w+(?:-\w+)?)?$', instructions, re.IGNORECASE)
    if watermark_match:
        operations.append(
            EditOperation(
                "add_watermark",
                {
                    "text": watermark_match.group(1).strip(),
                    "position": _extract_position(instructions, default="bottom-right"),
                    **_extract_timing(instructions),
                },
            )
        )

    if "sync" in lowered and "audio" in lowered:
        operations.append(EditOperation("sync_audio_video"))

    if any(phrase in lowered for phrase in ("audio mix", "mix audio", "audio mixing")):
        operations.append(EditOperation("mix_audio", {"mode": "mix"}))

    if "equaliz" in lowered:
        operations.append(EditOperation("mix_audio", {"mode": "equalize"}))

    volume_match = re.search(r"(increase|decrease)\s+audio(?: volume)?\s+by\s+(\d+)%", lowered)
    if volume_match:
        direction = 1 if volume_match.group(1) == "increase" else -1
        delta = int(volume_match.group(2)) / 100
        operations.append(EditOperation("mix_audio", {"volume": round(1 + direction * delta, 2)}))

    export_match = re.search(r"(?:export|save)(?: [\w\s]+)? to\s+([^\s]+)", instructions, re.IGNORECASE)
    if export_match:
        operations.append(EditOperation("set_export_path", {"output_path": export_match.group(1)}))

    return operations


def _extract_timing(instructions: str) -> dict[str, float]:
    range_match = re.search(
        r"(?:from|between)\s+(\d+(?:\.\d+)?)\s*(?:s|seconds?)?\s+(?:to|and)\s+(\d+(?:\.\d+)?)\s*(?:s|seconds?)?",
        instructions,
        re.IGNORECASE,
    )
    if range_match:
        return {"start_seconds": float(range_match.group(1)), "end_seconds": float(range_match.group(2))}

    single_match = re.search(r"at\s+(\d+(?:\.\d+)?)\s*(?:s|seconds?)", instructions, re.IGNORECASE)
    if single_match:
        point = float(single_match.group(1))
        return {"start_seconds": point, "end_seconds": point + 2}

    return {}


def _extract_position(instructions: str, default: str = "bottom-center") -> str:
    position_match = re.search(r"\b(top-left|top-right|bottom-left|bottom-right|center)\b", instructions, re.IGNORECASE)
    return position_match.group(1).lower() if position_match else default


def _drawtext_filter(text: str, position: str, start_seconds: float | None = None, end_seconds: float | None = None) -> str:
    escaped_text = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")
    coordinates = {
        "top-left": "x=20:y=20",
        "top-right": "x=w-text_w-20:y=20",
        "bottom-left": "x=20:y=h-text_h-20",
        "bottom-right": "x=w-text_w-20:y=h-text_h-20",
        "center": "x=(w-text_w)/2:y=(h-text_h)/2",
        "bottom-center": "x=(w-text_w)/2:y=h-text_h-30",
    }
    enable = ""
    if start_seconds is not None and end_seconds is not None:
        enable = f":enable='between(t,{start_seconds},{end_seconds})'"
    return (
        "drawtext="
        f"text='{escaped_text}':fontsize=24:fontcolor=white:box=1:boxcolor=black@0.45:"
        f"{coordinates.get(position, coordinates['bottom-center'])}{enable}"
    )


def build_ffmpeg_command(project: VideoProject, output_path: str) -> list[str]:
    video_filters: list[str] = []
    audio_filters: list[str] = []

    effect_filters = {
        "grayscale": "hue=s=0",
        "sepia": "colorchannelmixer=.393:.769:.189:.349:.686:.168:.272:.534:.131",
        "cinematic": "eq=contrast=1.1:saturation=1.15:brightness=0.02",
    }

    for operation in project.operations:
        if operation.operation == "apply_effect":
            effect_name = operation.parameters.get("name")
            if effect_name in effect_filters:
                video_filters.append(effect_filters[effect_name])
        elif operation.operation == "apply_transition":
            if operation.parameters.get("name") == "fade" and project.source.duration_seconds > 0:
                fade_out_start = max(project.source.duration_seconds - 0.5, 0)
                video_filters.extend([f"fade=t=in:st=0:d=0.5", f"fade=t=out:st={fade_out_start}:d=0.5"])
        elif operation.operation == "add_text_overlay":
            video_filters.append(
                _drawtext_filter(
                    operation.parameters["text"],
                    _extract_position(project.instructions),
                    operation.parameters.get("start_seconds"),
                    operation.parameters.get("end_seconds"),
                )
            )
        elif operation.operation == "add_watermark":
            video_filters.append(
                _drawtext_filter(
                    operation.parameters["text"],
                    operation.parameters.get("position", "bottom-right"),
                    operation.parameters.get("start_seconds"),
                    operation.parameters.get("end_seconds"),
                )
            )
        elif operation.operation == "sync_audio_video":
            audio_filters.append("aresample=async=1:first_pts=0")
        elif operation.operation == "mix_audio":
            if operation.parameters.get("mode") == "equalize":
                audio_filters.append("highpass=f=120,lowpass=f=8000")
            if operation.parameters.get("volume"):
                audio_filters.append(f"volume={operation.parameters['volume']}")

    command = ["ffmpeg", "-y", "-i", project.source.path]
    if video_filters:
        command.extend(["-vf", ",".join(video_filters)])
    if audio_filters:
        command.extend(["-af", ",".join(audio_filters)])
    command.extend(["-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", output_path])
    return command


class VideoEditingAgent:
    def __init__(self) -> None:
        self.tools: dict[str, Callable[..., Any]] = {
            "detect_scenes": self.detect_scenes,
            "apply_effect": self.apply_effect,
            "apply_transition": self.apply_transition,
            "add_text_overlay": self.add_text_overlay,
            "add_watermark": self.add_watermark,
            "mix_audio": self.mix_audio,
            "sync_audio_video": self.sync_audio_video,
            "set_export_path": self.set_export_path,
            "extract_clips": self.extract_clips_from_scenes,
        }

    def load_video(self, video_path: str, instructions: str = "") -> VideoProject:
        project = VideoProject(source=probe_video(video_path), instructions=instructions)
        if instructions:
            self.apply_instructions(project, instructions)
        if not project.scenes:
            project.scenes = detect_scene_segments(project.source)
        return project

    def apply_instructions(self, project: VideoProject, instructions: str) -> None:
        for operation in parse_edit_instructions(instructions):
            handler = self.tools.get(operation.operation)
            if handler is None:
                continue
            handler(project, **operation.parameters)

    def detect_scenes(self, project: VideoProject) -> list[TimelineSegment]:
        project.scenes = detect_scene_segments(project.source)
        return project.scenes

    def apply_effect(self, project: VideoProject, name: str) -> None:
        project.operations.append(EditOperation("apply_effect", {"name": name}))

    def apply_transition(self, project: VideoProject, name: str) -> None:
        project.operations.append(EditOperation("apply_transition", {"name": name}))

    def add_text_overlay(
        self,
        project: VideoProject,
        text: str,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> None:
        project.operations.append(
            EditOperation(
                "add_text_overlay",
                {"text": text, "start_seconds": start_seconds, "end_seconds": end_seconds},
            )
        )
        if start_seconds is not None:
            project.keyframes.append(Keyframe(time_seconds=start_seconds, property_name="text_overlay", value=text))

    def add_watermark(
        self,
        project: VideoProject,
        text: str,
        position: str = "bottom-right",
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> None:
        project.operations.append(
            EditOperation(
                "add_watermark",
                {
                    "text": text,
                    "position": position,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                },
            )
        )

    def mix_audio(self, project: VideoProject, mode: str | None = None, volume: float | None = None) -> None:
        parameters: dict[str, Any] = {}
        if mode:
            parameters["mode"] = mode
        if volume is not None:
            parameters["volume"] = volume
        project.operations.append(EditOperation("mix_audio", parameters))

    def sync_audio_video(self, project: VideoProject) -> None:
        project.operations.append(EditOperation("sync_audio_video"))

    def set_export_path(self, project: VideoProject, output_path: str) -> None:
        project.export_path = output_path

    def load_from_url(self, url: str, instructions: str = "", download_dir: str | None = None) -> VideoProject:
        """Download a video from *url* then load it as a :class:`VideoProject`."""
        local_path = download_video(url, output_dir=download_dir)
        return self.load_video(local_path, instructions)

    def extract_clips_from_scenes(self, project: VideoProject, output_dir: str | None = None) -> list[str]:
        """Extract each scene in *project* as a separate clip file.

        Returns a list of output file paths.
        """
        if not project.scenes:
            project.scenes = detect_scene_segments(project.source)

        dest_dir = Path(output_dir).expanduser().resolve() if output_dir else Path(project.source.path).parent / "clips"
        dest_dir.mkdir(parents=True, exist_ok=True)

        source_stem = Path(project.source.path).stem
        clip_paths: list[str] = []
        for scene in project.scenes:
            clip_filename = f"{source_stem}_{scene.label}.mp4"
            clip_path = extract_clip(
                project.source.path,
                scene.start_seconds,
                scene.end_seconds,
                str(dest_dir / clip_filename),
            )
            clip_paths.append(clip_path)
            LOGGER.info("Extracted clip: %s", clip_path)

        return clip_paths

    def export_video(self, project: VideoProject, output_path: str | None = None) -> str:
        resolved_output_path = Path(output_path or project.export_path or "edited_output.mp4").expanduser().resolve()
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        _run_media_command(build_ffmpeg_command(project, str(resolved_output_path)))
        project.export_path = str(resolved_output_path)
        return project.export_path

    def process_batch(self, requests: list[dict[str, str]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for request in requests:
            project = self.load_video(request["video_path"], request.get("instructions", ""))
            if request.get("output_path") or project.export_path:
                self.export_video(project, request.get("output_path"))
            results.append(project.summary())
        return results


def process_video_jobs() -> None:
    agent = VideoEditingAgent()
    batch_spec_path = os.getenv("VIDEO_BATCH_SPEC")

    if batch_spec_path:
        with open(batch_spec_path, "r", encoding="utf-8") as batch_spec_file:
            batch_requests = json.load(batch_spec_file)
        print(json.dumps(agent.process_batch(batch_requests), indent=2))
        return

    video_path = os.getenv("VIDEO_INPUT_PATH")
    if not video_path:
        raise VideoEditingError("Set VIDEO_INPUT_PATH or VIDEO_BATCH_SPEC to process video edits.")

    instructions = os.getenv("VIDEO_EDIT_INSTRUCTIONS", "detect scenes and export to edited_output.mp4")
    project = agent.load_video(video_path, instructions)

    output_path = os.getenv("VIDEO_OUTPUT_PATH")
    if output_path or project.export_path:
        agent.export_video(project, output_path)

    print(json.dumps(project.summary(), indent=2))


if __name__ == "__main__":
    process_video_jobs()
