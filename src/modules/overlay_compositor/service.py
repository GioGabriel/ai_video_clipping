from __future__ import annotations

import math
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

from src.core.command_runner import CommandRunner
from src.core.config import AppSettings
from src.core.models import (
    CaptionTheme,
    ClipArtifact,
    EditPlan,
    HookOverlay,
    KaraokeCue,
    KaraokeWord,
    OutputAspectRatio,
    SubtitleBurnInResult,
    TranscriptArtifact,
)
from src.modules.clip_generator.service import ClipGenerator
from src.modules.subtitle_generator.service import SubtitleGenerator


class OverlayCompositor:
    def __init__(
        self,
        settings: AppSettings,
        command_runner: CommandRunner,
        subtitle_generator: SubtitleGenerator,
    ) -> None:
        self.settings = settings
        self.command_runner = command_runner
        self.subtitle_generator = subtitle_generator
        self._font_cache: dict[tuple[int, bool], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    def apply(
        self,
        video_id: str,
        transcript: TranscriptArtifact,
        clips: list[ClipArtifact],
        edit_plans: dict[int, EditPlan] | None = None,
        output_aspect_ratio: OutputAspectRatio | str = OutputAspectRatio.VERTICAL_9_16.value,
        caption_theme: CaptionTheme | str = CaptionTheme.TIKTOK.value,
        progress_callback: Callable[[str, int | None, int | None], None] | None = None,
    ) -> SubtitleBurnInResult:
        if not clips:
            return SubtitleBurnInResult(burned_count=0)

        normalized_theme = self._normalize_caption_theme(caption_theme)
        rendered_count = 0
        total_clips = len(clips)
        for index, clip in enumerate(clips, start=1):
            if progress_callback is not None:
                progress_callback(
                    f"Rendering {normalized_theme} overlays for clip {index}/{total_clips}.",
                    index,
                    total_clips,
                )
            plan = (edit_plans or {}).get(clip.sequence_number)
            cues = self.subtitle_generator.build_karaoke_cues(clip, transcript, caption_theme=normalized_theme)
            hook_overlay = plan.hook_overlay if plan else None
            temp_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"overlay_{clip.sequence_number:03d}_",
                    dir=str(self.settings.clips_dir / video_id),
                )
            )
            try:
                concat_path = self._render_overlay_sequence(
                    temp_dir=temp_dir,
                    duration_seconds=clip.duration_seconds,
                    cues=cues,
                    hook_overlay=hook_overlay,
                    output_aspect_ratio=output_aspect_ratio,
                    caption_theme=normalized_theme,
                )
                if concat_path is None:
                    continue

                temp_output_path = clip.file_path.with_name(f"{clip.file_path.stem}.captioned.mp4")
                command = [
                    self.settings.ffmpeg_binary,
                    "-y",
                    "-i",
                    str(clip.file_path),
                    "-progress",
                    "pipe:2",
                    "-nostats",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_path.name),
                    "-filter_complex",
                    "[1:v]format=rgba[overlay];"
                    "[0:v][overlay]overlay=0:0:eof_action=pass:format=auto,format=yuv420p[outv]",
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
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(temp_output_path),
                ]
                self.command_runner.run(
                    command,
                    cwd=temp_dir,
                    on_output=self._build_ffmpeg_progress_handler(
                        clip_number=index,
                        clip_total=total_clips,
                        duration_seconds=clip.duration_seconds,
                        progress_callback=progress_callback,
                    ),
                )
                temp_output_path.replace(clip.file_path)
                rendered_count += 1
                if progress_callback is not None:
                    progress_callback(
                        f"{normalized_theme.capitalize()} overlay rendered for clip {index}/{total_clips}.",
                        index,
                        total_clips,
                    )
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        return SubtitleBurnInResult(burned_count=rendered_count)

    @staticmethod
    def _build_ffmpeg_progress_handler(
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
            elapsed_seconds = OverlayCompositor._parse_ffmpeg_clock(message.partition("=")[2])
            bucket = min(int((elapsed_seconds / duration_seconds) * 10), 9)
            if bucket <= last_bucket["value"]:
                return
            last_bucket["value"] = bucket
            progress_callback(
                "Overlay pass "
                f"{clip_number}/{clip_total}: "
                f"{OverlayCompositor._format_clock(elapsed_seconds)} / {OverlayCompositor._format_clock(duration_seconds)}.",
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

    def _render_overlay_sequence(
        self,
        temp_dir: Path,
        duration_seconds: float,
        cues: list[KaraokeCue],
        hook_overlay: HookOverlay | None,
        output_aspect_ratio: OutputAspectRatio | str,
        caption_theme: CaptionTheme | str,
    ) -> Path | None:
        duration_seconds = max(duration_seconds, 0.1)
        width, height = ClipGenerator.render_dimensions(output_aspect_ratio)
        boundaries = {0.0, round(duration_seconds, 3)}
        for cue in cues:
            boundaries.add(round(max(cue.start_seconds, 0.0), 3))
            boundaries.add(round(min(cue.end_seconds, duration_seconds), 3))
        if self.settings.overlay_hook_card_enabled and hook_overlay is not None:
            boundaries.add(round(max(hook_overlay.start_seconds, 0.0), 3))
            boundaries.add(round(min(hook_overlay.end_seconds, duration_seconds), 3))

        ordered = sorted(point for point in boundaries if 0.0 <= point <= duration_seconds)
        if ordered[-1] < duration_seconds:
            ordered.append(duration_seconds)

        blank_frame_path = self._render_overlay_frame(
            temp_dir=temp_dir,
            file_stem="blank",
            width=width,
            height=height,
            cue=None,
            hook_overlay=None,
            caption_theme=caption_theme,
        )

        segments: list[tuple[Path, float]] = []
        frame_cache: dict[tuple[object, ...], Path] = {}
        for start, end in zip(ordered, ordered[1:]):
            segment_duration = round(end - start, 3)
            if segment_duration <= 0:
                continue

            midpoint = start + (segment_duration / 2)
            cue = self._cue_for_time(cues, midpoint)
            active_hook = hook_overlay if self._is_hook_active(hook_overlay, midpoint) else None

            if cue is None and active_hook is None:
                segments.append((blank_frame_path, segment_duration))
                continue

            cache_key = self._frame_cache_key(cue, active_hook, caption_theme)
            frame_path = frame_cache.get(cache_key)
            if frame_path is None:
                frame_path = self._render_overlay_frame(
                    temp_dir=temp_dir,
                    file_stem=f"frame_{len(frame_cache) + 1:04d}",
                    width=width,
                    height=height,
                    cue=cue,
                    hook_overlay=active_hook,
                    caption_theme=caption_theme,
                )
                frame_cache[cache_key] = frame_path
            segments.append((frame_path, segment_duration))

        if not segments or all(path == blank_frame_path for path, _ in segments):
            return None

        concat_path = temp_dir / "overlay.ffconcat"
        lines = ["ffconcat version 1.0"]
        for frame_path, segment_duration in segments:
            lines.append(f"file '{frame_path.name}'")
            lines.append(f"duration {segment_duration:.3f}")
        lines.append(f"file '{segments[-1][0].name}'")
        concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return concat_path

    def _render_overlay_frame(
        self,
        temp_dir: Path,
        file_stem: str,
        width: int,
        height: int,
        cue: KaraokeCue | None,
        hook_overlay: HookOverlay | None,
        caption_theme: CaptionTheme | str,
    ) -> Path:
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        theme = self._theme_profile(caption_theme)

        self._draw_theme_frame_accents(draw, width, height, theme)
        if hook_overlay is not None and hook_overlay.text.strip():
            self._draw_hook_overlay(draw, width, height, hook_overlay.text, theme)
        if cue is not None:
            self._draw_karaoke_cue(draw, width, height, cue, theme)

        output_path = temp_dir / f"{file_stem}.png"
        image.save(output_path)
        return output_path

    def _draw_theme_frame_accents(self, draw: ImageDraw.ImageDraw, width: int, height: int, theme: dict[str, object]) -> None:
        if theme["name"] == CaptionTheme.CINEMATIC.value:
            bar_height = max(int(height * 0.085), 72)
            draw.rectangle((0, 0, width, bar_height), fill=(0, 0, 0, 118))
            draw.rectangle((0, height - bar_height, width, height), fill=(0, 0, 0, 132))
            accent_y = max(int(height * 0.18), 72)
            draw.line(
                (
                    width * 0.24,
                    accent_y,
                    width * 0.76,
                    accent_y,
                ),
                fill=theme["hook_accent_fill"],
                width=max(width // 540, 2),
            )
            return

        if theme["name"] == CaptionTheme.MOTIVATIONAL.value:
            glow_width = max(int(width * 0.46), 260)
            glow_height = max(int(height * 0.22), 180)
            center_x = width // 2
            center_y = int(height * 0.72)
            draw.ellipse(
                (
                    center_x - glow_width // 2,
                    center_y - glow_height // 2,
                    center_x + glow_width // 2,
                    center_y + glow_height // 2,
                ),
                fill=(255, 184, 88, 34),
            )
            draw.rectangle((0, 0, width, int(height * 0.08)), fill=(8, 8, 12, 28))
            return

        draw.rounded_rectangle(
            (
                width * 0.16,
                height * 0.075,
                width * 0.84,
                height * 0.118,
            ),
            radius=max(width // 80, 18),
            fill=(8, 16, 28, 40),
            outline=(46, 242, 255, 72),
            width=max(width // 360, 2),
        )

    def _draw_hook_overlay(self, draw: ImageDraw.ImageDraw, width: int, height: int, text: str, theme: dict[str, object]) -> None:
        max_width = int(width * float(theme["hook_max_width_ratio"]))
        font_size = int(theme["hook_font_size"])
        font = self._load_font(font_size, heavy=bool(theme["hook_font_heavy"]))
        lines = self._wrap_hook_text(text, draw, font, max_width)
        if not lines:
            return

        line_boxes = [draw.textbbox((0, 0), line, font=font, stroke_width=0) for line in lines]
        content_width = max((box[2] - box[0]) for box in line_boxes)
        while content_width > max_width and font_size > 24:
            font_size -= 2
            font = self._load_font(font_size, heavy=bool(theme["hook_font_heavy"]))
            lines = self._wrap_hook_text(text, draw, font, max_width)
            line_boxes = [draw.textbbox((0, 0), line, font=font, stroke_width=0) for line in lines]
            content_width = max((box[2] - box[0]) for box in line_boxes)

        line_height = max(box[3] - box[1] for box in line_boxes)
        spacing = max(font_size // 6, 8)
        total_height = (line_height * len(lines)) + (spacing * max(len(lines) - 1, 0))
        position = str(theme["hook_position"])
        if position == "center":
            start_y = max(int((height - total_height) * 0.24), 54)
        else:
            start_y = max(int(theme["hook_margin_top"]), 44)

        accent_y = start_y - max(font_size // 3, 14)
        accent_width = max(int(content_width * 0.62), 160)
        accent_x1 = (width - accent_width) / 2
        accent_x2 = accent_x1 + accent_width
        draw.line(
            (accent_x1, accent_y, accent_x2, accent_y),
            fill=theme["hook_accent_fill"],
            width=max(font_size // 14, 2),
        )

        text_y = start_y
        for line, box in zip(lines, line_boxes):
            line_width = box[2] - box[0]
            line_x = (width - line_width) / 2
            shadow_color = theme["hook_shadow_fill"]
            draw.text(
                (line_x, text_y + 4),
                line,
                font=font,
                fill=shadow_color,
            )
            draw.text(
                (line_x, text_y),
                line,
                font=font,
                fill=theme["hook_fill"],
                stroke_width=int(theme["hook_stroke_width"]),
                stroke_fill=theme["hook_stroke_fill"],
            )
            text_y += line_height + spacing

    @staticmethod
    def _wrap_hook_text(
        text: str,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_content_width: int,
    ) -> list[str]:
        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not raw_lines:
            return []

        wrapped_lines: list[str] = []
        for raw_line in raw_lines:
            words = raw_line.split()
            if not words:
                continue
            current_line = words[0]
            for word in words[1:]:
                candidate = f"{current_line} {word}"
                candidate_box = draw.textbbox((0, 0), candidate, font=font, stroke_width=0)
                if candidate_box[2] - candidate_box[0] <= max_content_width:
                    current_line = candidate
                    continue
                wrapped_lines.append(current_line)
                current_line = word
            wrapped_lines.append(current_line)
        return wrapped_lines[:3]

    def _draw_karaoke_cue(self, draw: ImageDraw.ImageDraw, width: int, height: int, cue: KaraokeCue, theme: dict[str, object]) -> None:
        if not cue.lines:
            return

        base_size = int(theme["font_size"])
        horizontal_margin = int(theme["horizontal_margin"])
        max_width = max(width - (horizontal_margin * 2), width // 3)
        metrics = self._measure_lines(draw, cue.lines, base_size, max_width, theme)
        while metrics["max_line_width"] > max_width and base_size > 30:
            base_size -= 4
            metrics = self._measure_lines(draw, cue.lines, base_size, max_width, theme)

        line_gap = int(theme["line_gap"])
        total_height = sum(line["height"] for line in metrics["lines"]) + (line_gap * max(len(metrics["lines"]) - 1, 0))
        y = height - int(theme["bottom_margin"]) - total_height

        panel_padding_x = int(theme["panel_padding_x"])
        panel_padding_y = int(theme["panel_padding_y"])
        if theme["line_panel_fill"] is not None and metrics["lines"]:
            panel_width = max(line["width"] for line in metrics["lines"]) + (panel_padding_x * 2)
            panel_height = total_height + (panel_padding_y * 2)
            panel_x1 = (width - panel_width) / 2
            panel_y1 = y - panel_padding_y
            panel_x2 = panel_x1 + panel_width
            panel_y2 = panel_y1 + panel_height
            draw.rounded_rectangle(
                (panel_x1, panel_y1, panel_x2, panel_y2),
                radius=int(theme["panel_radius"]),
                fill=theme["line_panel_fill"],
                outline=theme["panel_outline_fill"],
                width=int(theme["panel_outline_width"]),
            )

        for line in metrics["lines"]:
            x = (width - line["width"]) / 2
            for word in line["words"]:
                if word["is_active"] and theme["active_chip_fill"] is not None:
                    chip_x1 = x - int(theme["chip_padding_x"])
                    chip_y1 = y - int(theme["chip_padding_y"])
                    chip_x2 = x + word["width"] + int(theme["chip_padding_x"])
                    chip_y2 = y + word["height"] + int(theme["chip_padding_y"])
                    draw.rounded_rectangle(
                        (chip_x1, chip_y1, chip_x2, chip_y2),
                        radius=int(theme["chip_radius"]),
                        fill=theme["active_chip_fill"],
                        outline=theme["chip_outline_fill"],
                        width=int(theme["chip_outline_width"]),
                    )

                draw.text(
                    (x, y + int(theme["shadow_offset_y"])),
                    word["text"],
                    font=word["font"],
                    fill=word["shadow_fill"],
                    stroke_width=0,
                )
                draw.text(
                    (x, y),
                    word["text"],
                    font=word["font"],
                    fill=word["fill"],
                    stroke_width=int(theme["stroke_width"]),
                    stroke_fill=theme["stroke_fill"],
                )
                x += word["width"] + line["space_width"]
            y += line["height"] + line_gap

    def _measure_lines(
        self,
        draw: ImageDraw.ImageDraw,
        lines: list[list[KaraokeWord]],
        base_size: int,
        max_width: int,
        theme: dict[str, object],
    ) -> dict[str, object]:
        active_scale = float(theme["active_scale"])
        active_size = max(int(round(base_size * active_scale)), base_size + 2)
        measured_lines = []
        max_line_width = 0
        for line in lines:
            measured_words = []
            line_height = 0
            line_width = 0
            for word in line:
                font = self._load_font(active_size if word.is_active else base_size, heavy=bool(theme["font_heavy"]))
                bbox = draw.textbbox((0, 0), word.text, font=font, stroke_width=int(theme["stroke_width"]))
                word_width = bbox[2] - bbox[0]
                word_height = bbox[3] - bbox[1]
                measured_words.append(
                    {
                        "text": word.text,
                        "font": font,
                        "width": word_width,
                        "height": word_height,
                        "fill": theme["active_fill"] if word.is_active else theme["inactive_fill"],
                        "shadow_fill": theme["active_shadow_fill"] if word.is_active else theme["shadow_fill"],
                        "is_active": word.is_active,
                    }
                )
                line_height = max(line_height, word_height)
                line_width += word_width

            space_font = self._load_font(base_size, heavy=bool(theme["font_heavy"]))
            space_width = draw.textbbox((0, 0), " ", font=space_font, stroke_width=0)[2]
            if measured_words:
                line_width += space_width * (len(measured_words) - 1)
            line_width = min(line_width, max_width + 240)
            max_line_width = max(max_line_width, line_width)
            measured_lines.append(
                {
                    "words": measured_words,
                    "width": line_width,
                    "height": line_height,
                    "space_width": space_width,
                }
            )

        return {"lines": measured_lines, "max_line_width": max_line_width}

    def _load_font(self, size: int, heavy: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        cache_key = (size, heavy)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        candidates = [
            self.settings.subtitle_font_name,
            f"{self.settings.subtitle_font_name}.ttf",
            "/System/Library/Fonts/Supplemental/Arial Black.ttf" if heavy else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Black.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "Arial Black.ttf",
            "Arial Bold.ttf",
            "DejaVuSans-Bold.ttf" if heavy else "DejaVuSans.ttf",
        ]

        font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        for candidate in candidates:
            if not candidate:
                continue
            try:
                font = ImageFont.truetype(candidate, size=size)
                self._font_cache[cache_key] = font
                return font
            except OSError:
                continue

        font = ImageFont.load_default()
        self._font_cache[cache_key] = font
        return font

    @staticmethod
    def _cue_for_time(cues: list[KaraokeCue], timestamp_seconds: float) -> KaraokeCue | None:
        for cue in cues:
            if cue.start_seconds <= timestamp_seconds < cue.end_seconds:
                return cue
        return cues[-1] if cues and math.isclose(timestamp_seconds, cues[-1].end_seconds, abs_tol=0.03) else None

    @staticmethod
    def _is_hook_active(hook_overlay: HookOverlay | None, timestamp_seconds: float) -> bool:
        if hook_overlay is None:
            return False
        return hook_overlay.start_seconds <= timestamp_seconds < hook_overlay.end_seconds

    @staticmethod
    def _frame_cache_key(
        cue: KaraokeCue | None,
        hook_overlay: HookOverlay | None,
        caption_theme: CaptionTheme | str,
    ) -> tuple[object, ...]:
        cue_key: tuple[object, ...]
        if cue is None:
            cue_key = ("blank",)
        else:
            cue_key = (
                round(cue.start_seconds, 3),
                round(cue.end_seconds, 3),
                tuple(tuple((word.text, word.is_active) for word in line) for line in cue.lines),
            )
        hook_key = (
            hook_overlay.text if hook_overlay else "",
            round(hook_overlay.start_seconds, 3) if hook_overlay else 0.0,
            round(hook_overlay.end_seconds, 3) if hook_overlay else 0.0,
        )
        return cue_key + hook_key + (OverlayCompositor._normalize_caption_theme(caption_theme),)

    @staticmethod
    def _parse_color(value: str) -> tuple[int, int, int, int]:
        red, green, blue = ImageColor.getrgb(value)
        return red, green, blue, 255

    @staticmethod
    def _normalize_caption_theme(value: CaptionTheme | str) -> str:
        if isinstance(value, CaptionTheme):
            return value.value

        normalized = str(value or CaptionTheme.TIKTOK.value).strip().lower()
        allowed = {theme.value for theme in CaptionTheme}
        return normalized if normalized in allowed else CaptionTheme.TIKTOK.value

    def _theme_profile(self, caption_theme: CaptionTheme | str) -> dict[str, object]:
        theme = self._normalize_caption_theme(caption_theme)
        common = {
            "name": theme,
            "stroke_fill": (16, 16, 16, 255),
            "shadow_fill": (0, 0, 0, self.settings.overlay_caption_shadow_opacity),
            "active_shadow_fill": (0, 0, 0, min(self.settings.overlay_caption_shadow_opacity + 28, 220)),
            "shadow_offset_y": 6,
            "font_size": self.settings.subtitle_font_size,
            "font_heavy": True,
            "line_gap": max(self.settings.subtitle_font_size // 4, 12),
            "horizontal_margin": self.settings.subtitle_margin_horizontal,
            "bottom_margin": self.settings.subtitle_margin_bottom,
            "hook_margin_top": self.settings.overlay_card_margin_top,
            "hook_max_width_ratio": 0.8,
            "hook_font_size": max(self.settings.subtitle_font_size - 18, 36),
            "hook_font_heavy": True,
            "hook_position": "top",
            "hook_fill": (255, 255, 255, 255),
            "hook_shadow_fill": (0, 0, 0, 160),
            "hook_stroke_fill": (10, 10, 12, 255),
            "hook_stroke_width": 6,
            "hook_accent_fill": (255, 255, 255, 210),
            "line_panel_fill": None,
            "panel_outline_fill": (255, 255, 255, 0),
            "panel_outline_width": 0,
            "panel_padding_x": 34,
            "panel_padding_y": 22,
            "panel_radius": 36,
            "active_chip_fill": None,
            "chip_outline_fill": (255, 255, 255, 0),
            "chip_outline_width": 0,
            "chip_padding_x": 16,
            "chip_padding_y": 10,
            "chip_radius": 24,
            "stroke_width": 8,
            "active_scale": self.settings.overlay_active_word_scale,
            "inactive_fill": (255, 255, 255, 255),
            "active_fill": self._parse_color(self.settings.overlay_active_word_color),
        }

        if theme == CaptionTheme.CINEMATIC.value:
            return {
                **common,
                "inactive_fill": (248, 241, 216, 255),
                "active_fill": (255, 211, 107, 255),
                "line_panel_fill": None,
                "hook_fill": (248, 241, 216, 255),
                "hook_accent_fill": (255, 211, 107, 230),
                "hook_stroke_width": 5,
                "hook_position": "center",
                "hook_font_heavy": False,
                "font_heavy": False,
                "stroke_width": 7,
                "active_scale": 1.08,
            }

        if theme == CaptionTheme.MOTIVATIONAL.value:
            return {
                **common,
                "inactive_fill": (255, 246, 225, 255),
                "active_fill": (255, 179, 71, 255),
                "line_panel_fill": None,
                "panel_outline_fill": (255, 255, 255, 0),
                "panel_outline_width": 0,
                "active_chip_fill": None,
                "chip_outline_fill": (255, 255, 255, 0),
                "chip_outline_width": 0,
                "hook_fill": (255, 219, 125, 255),
                "hook_accent_fill": (255, 179, 71, 236),
                "hook_stroke_width": 5,
                "active_scale": 1.1,
            }

        return {
            **common,
            "inactive_fill": (255, 255, 255, 255),
            "active_fill": self._parse_color(self.settings.overlay_active_word_color),
            "line_panel_fill": None,
            "panel_outline_fill": (255, 255, 255, 0),
            "panel_outline_width": 0,
            "active_chip_fill": None,
            "chip_outline_fill": (255, 255, 255, 0),
            "chip_outline_width": 0,
            "hook_fill": (255, 255, 255, 255),
            "hook_accent_fill": (46, 242, 255, 236),
            "hook_stroke_width": 6,
            "active_scale": max(self.settings.overlay_active_word_scale, 1.14),
        }
