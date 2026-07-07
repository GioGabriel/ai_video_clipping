from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from src.core.command_runner import CommandError, CommandRunner
from src.core.config import AppSettings
from src.core.models import ClipArtifact, EditPlan, OutputAspectRatio, SpeakerFocusTrack, SubtitleBurnInResult

logger = logging.getLogger(__name__)


class ClipGenerator:
    _MAX_FOCUS_TRACK_POINTS = 24
    _FOCUS_TRACK_MIN_DELTA_PIXELS = 6.0
    _FOCUS_TRACK_MIN_TIME_GAP_SECONDS = 0.9

    def __init__(self, settings: AppSettings, command_runner: CommandRunner) -> None:
        self.settings = settings
        self.command_runner = command_runner
        self._subtitle_filter_supported: bool | None = None

    def generate(
        self,
        job_id: str,
        video_id: str,
        video_path: Path,
        plans: list[EditPlan],
        output_aspect_ratio: OutputAspectRatio | str = OutputAspectRatio.VERTICAL_9_16.value,
        focus_tracks: dict[int, SpeakerFocusTrack] | None = None,
        progress_callback: Callable[[str, int | None, int | None], None] | None = None,
    ) -> list[ClipArtifact]:
        clip_directory = self.settings.clips_dir / video_id
        clip_directory.mkdir(parents=True, exist_ok=True)

        clips: list[ClipArtifact] = []
        total_plans = len(plans)
        for sequence_number, plan in enumerate(plans, start=1):
            output_path = clip_directory / f"clip_{sequence_number:03d}.mp4"
            duration = max(plan.duration_seconds, 1.0)
            focus_track = (focus_tracks or {}).get(sequence_number)
            if progress_callback is not None:
                progress_callback(
                    f"Rendering clip {sequence_number}/{total_plans} "
                    f"({self._format_clock(duration)}) "
                    f"for '{plan.hook}'.",
                    sequence_number,
                    total_plans,
                )
            command = [
                self.settings.ffmpeg_binary,
                "-y",
                "-ss",
                f"{plan.start_seconds:.3f}",
                "-i",
                str(video_path),
                "-progress",
                "pipe:2",
                "-nostats",
                "-t",
                f"{duration:.3f}",
                "-filter_complex",
                self._platform_filter_graph(output_aspect_ratio, focus_track),
                "-map",
                "[outv]",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-b:v",
                self.settings.export_video_bitrate,
                "-maxrate",
                self.settings.export_video_maxrate,
                "-bufsize",
                self.settings.export_video_bufsize,
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                self.settings.export_audio_bitrate,
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            self.command_runner.run(
                command,
                on_output=self._build_ffmpeg_progress_handler(
                    clip_number=sequence_number,
                    clip_total=total_plans,
                    duration_seconds=duration,
                    progress_callback=progress_callback,
                ),
            )

            clips.append(
                ClipArtifact(
                    job_id=job_id,
                    video_id=video_id,
                    sequence_number=sequence_number,
                    file_path=output_path,
                    start_seconds=plan.start_seconds,
                    end_seconds=plan.end_seconds,
                    hook=plan.hook,
                    reason=plan.reason,
                    score=plan.score,
                )
            )
            logger.info("Generated clip %s for video %s", output_path.name, video_id)
            if progress_callback is not None:
                progress_callback(
                    f"Saved clip {sequence_number}/{total_plans}: {output_path.name}.",
                    sequence_number,
                    total_plans,
                )

        return clips

    def burn_subtitles(self, clips: list[ClipArtifact], subtitle_paths: dict[int, Path]) -> SubtitleBurnInResult:
        burned = 0
        if not clips:
            return SubtitleBurnInResult(burned_count=0)

        if not self._supports_subtitle_filter():
            warning_message = (
                "FFmpeg subtitle burn-in was skipped because the installed FFmpeg binary "
                "does not include the subtitles filter. Styled .ass and .srt files were still generated."
            )
            logger.warning(warning_message)
            return SubtitleBurnInResult(burned_count=0, warning_message=warning_message)

        for clip in clips:
            subtitle_path = subtitle_paths.get(clip.sequence_number)
            if subtitle_path is None or not subtitle_path.exists():
                continue

            temp_output_path = clip.file_path.with_name(f"{clip.file_path.stem}.captioned.mp4")
            command = [
                self.settings.ffmpeg_binary,
                "-y",
                "-i",
                str(clip.file_path),
                "-vf",
                f"subtitles=filename={self._escape_filter_path(subtitle_path)}",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-b:v",
                self.settings.export_video_bitrate,
                "-maxrate",
                self.settings.export_video_maxrate,
                "-bufsize",
                self.settings.export_video_bufsize,
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(temp_output_path),
            ]
            try:
                self.command_runner.run(command)
            except CommandError as exc:
                if self._is_missing_subtitle_filter_error(str(exc)):
                    self._subtitle_filter_supported = False
                    warning_message = (
                        "FFmpeg subtitle burn-in was skipped because the installed FFmpeg binary "
                        "does not include the subtitles filter. Styled .ass and .srt files were still generated."
                    )
                    logger.warning(warning_message)
                    return SubtitleBurnInResult(burned_count=burned, warning_message=warning_message)
                raise
            temp_output_path.replace(clip.file_path)
            burned += 1

        return SubtitleBurnInResult(burned_count=burned)

    def _platform_filter_graph(
        self,
        output_aspect_ratio: OutputAspectRatio | str,
        focus_track: SpeakerFocusTrack | None = None,
    ) -> str:
        width, height = self.render_dimensions(output_aspect_ratio)
        if focus_track is None:
            return (
                f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},fps={self.settings.export_fps},setsar=1,"
                f"unsharp=5:5:0.48:5:5:0.0,format=yuv420p[outv]"
            )

        crop_width, crop_height = self._crop_dimensions(
            source_width=focus_track.source_width,
            source_height=focus_track.source_height,
            target_width=width,
            target_height=height,
        )
        if crop_width >= focus_track.source_width and crop_height >= focus_track.source_height:
            return (
                f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},fps={self.settings.export_fps},setsar=1,"
                f"unsharp=5:5:0.48:5:5:0.0,format=yuv420p[outv]"
            )

        x_values = [
            self._clamp(
                (point.center_x * focus_track.source_width) - (crop_width / 2),
                0.0,
                max(focus_track.source_width - crop_width, 0),
            )
            for point in focus_track.points
        ]
        y_values = [
            self._clamp(
                (point.center_y * focus_track.source_height) - (crop_height / 2),
                0.0,
                max(focus_track.source_height - crop_height, 0),
            )
            for point in focus_track.points
        ]
        simplified_times, simplified_x, simplified_y = self._simplify_focus_track(
            [point.time_seconds for point in focus_track.points],
            x_values,
            y_values,
        )
        x_expr = self._interpolated_expression(simplified_times, simplified_x)
        y_expr = self._interpolated_expression(simplified_times, simplified_y)
        return (
            f"[0:v]crop=w={crop_width}:h={crop_height}:x='{x_expr}':y='{y_expr}',"
            f"scale={width}:{height}:flags=lanczos,fps={self.settings.export_fps},setsar=1,"
            f"unsharp=5:5:0.48:5:5:0.0,format=yuv420p[outv]"
        )

    @staticmethod
    def render_dimensions(output_aspect_ratio: OutputAspectRatio | str) -> tuple[int, int]:
        normalized = str(output_aspect_ratio)
        mapping = {
            OutputAspectRatio.VERTICAL_9_16.value: (1080, 1920),
            OutputAspectRatio.LANDSCAPE_16_9.value: (1920, 1080),
            OutputAspectRatio.SQUARE_1_1.value: (1080, 1080),
            OutputAspectRatio.PORTRAIT_4_5.value: (1080, 1350),
        }
        return mapping.get(normalized, mapping[OutputAspectRatio.VERTICAL_9_16.value])

    @staticmethod
    def _crop_dimensions(source_width: int, source_height: int, target_width: int, target_height: int) -> tuple[int, int]:
        source_ratio = source_width / source_height
        target_ratio = target_width / target_height
        if source_ratio >= target_ratio:
            crop_height = source_height
            crop_width = int(round(crop_height * target_ratio))
        else:
            crop_width = source_width
            crop_height = int(round(crop_width / target_ratio))
        return max(crop_width - (crop_width % 2), 2), max(crop_height - (crop_height % 2), 2)

    @staticmethod
    def _interpolated_expression(times: list[float], values: list[float]) -> str:
        if not times or not values:
            return "0"
        if len(times) == 1 or len(values) == 1:
            return f"{values[0]:.3f}"

        expression = f"{values[-1]:.3f}"
        for index in range(len(times) - 2, -1, -1):
            start = times[index]
            end = max(times[index + 1], start + 0.001)
            start_value = values[index]
            end_value = values[index + 1]
            delta = end_value - start_value
            duration = end - start
            if abs(delta) < 0.5:
                segment = f"{start_value:.3f}"
            else:
                segment = f"{start_value:.3f}+({delta:.3f})*((t-{start:.3f})/{duration:.3f})"
            expression = f"if(lt(t,{end:.3f}),{segment},{expression})"
        return expression

    def _simplify_focus_track(
        self,
        times: list[float],
        x_values: list[float],
        y_values: list[float],
    ) -> tuple[list[float], list[float], list[float]]:
        if len(times) <= 2:
            return times, x_values, y_values

        kept_indices = [0]
        last_kept = 0
        for index in range(1, len(times) - 1):
            time_gap = times[index] - times[last_kept]
            x_delta = abs(x_values[index] - x_values[last_kept])
            y_delta = abs(y_values[index] - y_values[last_kept])
            if (
                time_gap >= self._FOCUS_TRACK_MIN_TIME_GAP_SECONDS
                and (x_delta >= self._FOCUS_TRACK_MIN_DELTA_PIXELS or y_delta >= self._FOCUS_TRACK_MIN_DELTA_PIXELS)
            ):
                kept_indices.append(index)
                last_kept = index
        if kept_indices[-1] != len(times) - 1:
            kept_indices.append(len(times) - 1)

        if len(kept_indices) > self._MAX_FOCUS_TRACK_POINTS:
            target = self._MAX_FOCUS_TRACK_POINTS
            sampled = [kept_indices[0]]
            interior_count = target - 2
            if interior_count > 0:
                last_position = len(kept_indices) - 1
                for slot in range(1, interior_count + 1):
                    mapped = round((slot * last_position) / (interior_count + 1))
                    sampled.append(kept_indices[mapped])
            sampled.append(kept_indices[-1])
            kept_indices = sorted(set(sampled))

        simplified_times = [times[index] for index in kept_indices]
        simplified_x = [x_values[index] for index in kept_indices]
        simplified_y = [y_values[index] for index in kept_indices]
        return simplified_times, simplified_x, simplified_y

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(value, maximum))

    @staticmethod
    def _escape_filter_path(path: Path) -> str:
        return (
            str(path)
            .replace("\\", r"\\")
            .replace(" ", r"\ ")
            .replace(":", r"\:")
            .replace("'", r"\'")
            .replace(",", r"\,")
            .replace("[", r"\[")
            .replace("]", r"\]")
        )

    def _supports_subtitle_filter(self) -> bool:
        if self._subtitle_filter_supported is not None:
            return self._subtitle_filter_supported

        try:
            result = subprocess.run(
                [self.settings.ffmpeg_binary, "-hide_banner", "-filters"],
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError:
            logger.warning("Could not inspect FFmpeg filters for subtitle burn-in support.", exc_info=True)
            self._subtitle_filter_supported = True
            return self._subtitle_filter_supported

        if result.returncode != 0:
            logger.warning("FFmpeg filter inspection returned %s; attempting subtitle burn-in anyway.", result.returncode)
            self._subtitle_filter_supported = True
            return self._subtitle_filter_supported

        filter_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        self._subtitle_filter_supported = bool(re.search(r"(?m)^\s*\S+\s+subtitles\s", filter_output))
        return self._subtitle_filter_supported

    @staticmethod
    def _is_missing_subtitle_filter_error(message: str) -> bool:
        lowered = message.lower()
        return "unknown filter 'subtitles'" in lowered or "no such filter: 'subtitles'" in lowered

    @classmethod
    def _build_ffmpeg_progress_handler(
        cls,
        clip_number: int,
        clip_total: int,
        duration_seconds: float,
        progress_callback: Callable[[str, int | None, int | None], None] | None,
    ) -> Callable[[str], None] | None:
        if progress_callback is None:
            return None

        duration_seconds = max(duration_seconds, 0.1)
        last_bucket = {"value": -1}

        def handle(line: str) -> None:
            message = line.strip()
            if not message or not message.startswith("out_time="):
                return
            elapsed_seconds = cls._parse_ffmpeg_clock(message.partition("=")[2])
            bucket = min(int((elapsed_seconds / duration_seconds) * 10), 9)
            if bucket <= last_bucket["value"]:
                return
            last_bucket["value"] = bucket
            progress_callback(
                f"Clip {clip_number}/{clip_total} render progress: "
                f"{cls._format_clock(elapsed_seconds)} / {cls._format_clock(duration_seconds)}.",
                clip_number,
                clip_total,
            )

        return handle

    @staticmethod
    def _parse_ffmpeg_clock(value: str) -> float:
        hours, minutes, seconds = value.strip().split(":")
        return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)

    @staticmethod
    def _format_clock(seconds: float) -> str:
        total = max(int(round(seconds)), 0)
        hours = total // 3600
        minutes = (total % 3600) // 60
        remainder = total % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{remainder:02d}"
        return f"{minutes}:{remainder:02d}"
