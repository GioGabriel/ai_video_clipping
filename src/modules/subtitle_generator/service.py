from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.core.config import AppSettings
from src.core.models import (
    CaptionTheme,
    ClipArtifact,
    EditPlan,
    HookOverlay,
    KaraokeCue,
    KaraokeWord,
    OutputAspectRatio,
    SubtitleArtifact,
    TranscriptArtifact,
    TranscriptSegment,
    TranscriptWord,
)
from src.core.timecode import seconds_to_srt_timestamp
from src.modules.clip_generator.service import ClipGenerator

@dataclass(slots=True)
class CaptionCue:
    start_seconds: float
    end_seconds: float
    display_text: str
    sidecar_text: str


class SubtitleGenerator:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def generate(
        self,
        video_id: str,
        transcript: TranscriptArtifact,
        clips: list[ClipArtifact],
        edit_plans: dict[int, EditPlan] | None = None,
        output_aspect_ratio: OutputAspectRatio | str = OutputAspectRatio.VERTICAL_9_16.value,
        caption_theme: CaptionTheme | str = CaptionTheme.TIKTOK.value,
        progress_callback: Callable[[str, int | None, int | None], None] | None = None,
    ) -> dict[int, SubtitleArtifact]:
        subtitle_dir = self.settings.clips_dir / video_id / "subtitles"
        subtitle_dir.mkdir(parents=True, exist_ok=True)
        styled_dir = subtitle_dir / "styled"
        styled_dir.mkdir(parents=True, exist_ok=True)

        normalized_theme = self._normalize_caption_theme(caption_theme)
        output: dict[int, SubtitleArtifact] = {}
        total_clips = len(clips)
        for index, clip in enumerate(clips, start=1):
            if progress_callback is not None:
                progress_callback(
                    f"Generating subtitle files for clip {index}/{total_clips}.",
                    index,
                    total_clips,
                )
            sidecar_path = subtitle_dir / f"clip_{clip.sequence_number:03d}.srt"
            styled_path = styled_dir / f"clip_{clip.sequence_number:03d}.ass"
            plan = (edit_plans or {}).get(clip.sequence_number)
            sidecar_cues = self._build_phrase_cues(clip, transcript)
            styled_cues = self.build_karaoke_cues(clip, transcript, caption_theme=normalized_theme)

            if sidecar_cues or styled_cues:
                sidecar_path.write_text(
                    self._to_srt(sidecar_cues or self._karaoke_cues_to_sidecar_cues(styled_cues)),
                    encoding="utf-8",
                )
                styled_path.write_text(
                    self._to_ass(
                        styled_cues or self._caption_cues_to_karaoke_cues(sidecar_cues),
                        plan.hook_overlay if plan else None,
                        output_aspect_ratio=output_aspect_ratio,
                        caption_theme=normalized_theme,
                    ),
                    encoding="utf-8",
                )
                output[clip.sequence_number] = SubtitleArtifact(
                    sidecar_path=sidecar_path,
                    styled_path=styled_path,
                )
                if progress_callback is not None:
                    progress_callback(
                        f"Subtitle sidecars written for clip {index}/{total_clips}.",
                        index,
                        total_clips,
                    )

        return output

    def build_karaoke_cues(
        self,
        clip: ClipArtifact,
        transcript: TranscriptArtifact,
        caption_theme: CaptionTheme | str = CaptionTheme.TIKTOK.value,
    ) -> list[KaraokeCue]:
        relative_words = self._relative_words_for_clip(clip, transcript)
        if not relative_words:
            return []

        normalized_theme = self._normalize_caption_theme(caption_theme)
        cues: list[KaraokeCue] = []
        for index, word in enumerate(relative_words):
            lines = self._build_kinetic_lines(relative_words, index, normalized_theme)
            cues.append(
                KaraokeCue(
                    start_seconds=word.start_seconds,
                    end_seconds=max(word.end_seconds, word.start_seconds + 0.12),
                    lines=lines,
                    spoken_text=word.text.upper(),
                )
            )
        return cues

    def _build_phrase_cues(self, clip: ClipArtifact, transcript: TranscriptArtifact) -> list[CaptionCue]:
        words = self._relative_words_for_clip(clip, transcript)
        if not words:
            return []

        cues: list[CaptionCue] = []
        current_words: list[TranscriptWord] = []
        for word in words:
            candidate_words = current_words + [word]
            candidate_text = " ".join(item.text.strip() for item in candidate_words if item.text.strip())
            if (
                current_words
                and (
                    len(candidate_words) > self.settings.subtitle_words_per_cue
                    or len(candidate_text) > self.settings.subtitle_max_chars_per_cue
                )
            ):
                cues.append(self._caption_cue_from_words(current_words))
                current_words = [word]
            else:
                current_words = candidate_words

        if current_words:
            cues.append(self._caption_cue_from_words(current_words))

        return self._merge_adjacent_short_cues(cues)

    def _relative_words_for_clip(self, clip: ClipArtifact, transcript: TranscriptArtifact) -> list[TranscriptWord]:
        words: list[TranscriptWord] = []
        for segment in transcript.segments:
            overlap_start = max(segment.start_seconds, clip.start_seconds)
            overlap_end = min(segment.end_seconds, clip.end_seconds)
            if overlap_end <= overlap_start:
                continue

            if segment.words:
                words.extend(self._relative_words_from_words(segment.words, clip, overlap_start, overlap_end))
            else:
                words.extend(self._relative_words_from_segment_text(segment, clip, overlap_start, overlap_end))

        return words

    def _relative_words_from_words(
        self,
        words: list[TranscriptWord],
        clip: ClipArtifact,
        overlap_start: float,
        overlap_end: float,
    ) -> list[TranscriptWord]:
        output: list[TranscriptWord] = []
        for word in words:
            word_start = max(word.start_seconds, overlap_start)
            word_end = min(word.end_seconds, overlap_end)
            if word_end <= word_start:
                continue

            text = word.text.strip()
            if not text:
                continue

            output.append(
                TranscriptWord(
                    start_seconds=word_start - clip.start_seconds,
                    end_seconds=word_end - clip.start_seconds,
                    text=text,
                )
            )
        return output

    def _relative_words_from_segment_text(
        self,
        segment: TranscriptSegment,
        clip: ClipArtifact,
        overlap_start: float,
        overlap_end: float,
    ) -> list[TranscriptWord]:
        tokens = [token.strip() for token in segment.text.split() if token.strip()]
        if not tokens:
            return []

        total_duration = max(overlap_end - overlap_start, 0.35)
        slice_duration = total_duration / len(tokens)
        cursor = overlap_start
        output: list[TranscriptWord] = []
        for index, token in enumerate(tokens):
            token_end = overlap_end if index == len(tokens) - 1 else cursor + slice_duration
            output.append(
                TranscriptWord(
                    start_seconds=cursor - clip.start_seconds,
                    end_seconds=max(token_end - clip.start_seconds, cursor - clip.start_seconds + 0.12),
                    text=token,
                )
            )
            cursor = token_end
        return output

    def _caption_cue_from_words(self, words: list[TranscriptWord]) -> CaptionCue:
        sidecar_words = [word.text.strip() for word in words if word.text.strip()]
        display_text = self._format_display_line(sidecar_words)
        return CaptionCue(
            start_seconds=words[0].start_seconds,
            end_seconds=max(words[-1].end_seconds, words[0].start_seconds + 0.25),
            display_text=display_text.upper(),
            sidecar_text=" ".join(sidecar_words),
        )

    @staticmethod
    def _format_display_line(words: list[str]) -> str:
        if len(words) <= 2:
            return " ".join(words)

        split_index = (len(words) + 1) // 2
        first_line = " ".join(words[:split_index])
        second_line = " ".join(words[split_index:])
        return first_line if not second_line else f"{first_line}\n{second_line}"

    @staticmethod
    def _merge_adjacent_short_cues(cues: list[CaptionCue]) -> list[CaptionCue]:
        if not cues:
            return []

        merged: list[CaptionCue] = [cues[0]]
        for cue in cues[1:]:
            previous = merged[-1]
            if cue.start_seconds - previous.end_seconds <= 0.08 and previous.end_seconds - previous.start_seconds < 0.4:
                merged[-1] = CaptionCue(
                    start_seconds=previous.start_seconds,
                    end_seconds=cue.end_seconds,
                    display_text=f"{previous.display_text}\n{cue.display_text}",
                    sidecar_text=f"{previous.sidecar_text} {cue.sidecar_text}",
                )
                continue
            merged.append(cue)
        return merged

    def _build_kinetic_lines(
        self,
        words: list[TranscriptWord],
        active_index: int,
        caption_theme: CaptionTheme | str,
    ) -> list[list[KaraokeWord]]:
        window_words = self._window_words(words, active_index, caption_theme)
        midpoint = (len(window_words) + 1) // 2
        formatted = [
            KaraokeWord(text=word.text.upper(), is_active=index == active_index)
            for index, word in window_words
        ]
        first_line = formatted[:midpoint]
        second_line = formatted[midpoint:]
        return [line for line in (first_line, second_line) if line]

    def _window_words(
        self,
        words: list[TranscriptWord],
        active_index: int,
        caption_theme: CaptionTheme | str,
    ) -> list[tuple[int, TranscriptWord]]:
        window_size = self._theme_context_words(caption_theme)
        if window_size <= 1:
            return [(active_index, words[active_index])]

        start = max(active_index - 1, 0)
        end = min(start + window_size, len(words))
        if active_index >= end:
            start = max(active_index - window_size + 1, 0)
            end = active_index + 1
        if end - start < window_size and start > 0:
            start = max(end - window_size, 0)
        indices = list(range(start, end))
        max_total_chars = max(self.settings.subtitle_max_chars_per_line * 2, self.settings.subtitle_max_chars_per_line)
        while len(indices) > 1 and self._window_char_count(indices, words) > max_total_chars:
            if active_index - indices[0] >= indices[-1] - active_index:
                indices = indices[1:]
            else:
                indices = indices[:-1]
        return [(index, words[index]) for index in indices]

    @staticmethod
    def _window_char_count(indices: list[int], words: list[TranscriptWord]) -> int:
        if not indices:
            return 0
        return sum(len(words[index].text) for index in indices) + max(len(indices) - 1, 0)

    @staticmethod
    def _karaoke_cues_to_sidecar_cues(cues: list[KaraokeCue]) -> list[CaptionCue]:
        return [
            CaptionCue(
                start_seconds=cue.start_seconds,
                end_seconds=cue.end_seconds,
                display_text=cue.spoken_text,
                sidecar_text=cue.spoken_text,
            )
            for cue in cues
        ]

    @staticmethod
    def _caption_cues_to_karaoke_cues(cues: list[CaptionCue]) -> list[KaraokeCue]:
        return [
            KaraokeCue(
                start_seconds=cue.start_seconds,
                end_seconds=cue.end_seconds,
                lines=[[KaraokeWord(text=word.upper(), is_active=False) for word in cue.display_text.replace("\n", " ").split()]],
                spoken_text=cue.sidecar_text.upper(),
            )
            for cue in cues
        ]

    def _to_srt(self, cues: list[CaptionCue]) -> str:
        entries: list[str] = []
        for index, cue in enumerate(cues, start=1):
            start_timestamp = seconds_to_srt_timestamp(cue.start_seconds)
            end_timestamp = seconds_to_srt_timestamp(cue.end_seconds)
            entries.append(f"{index}\n{start_timestamp} --> {end_timestamp}\n{cue.sidecar_text}\n")
        return "\n".join(entries).strip() + "\n"

    def _to_ass(
        self,
        cues: list[KaraokeCue],
        hook_overlay: HookOverlay | None = None,
        *,
        output_aspect_ratio: OutputAspectRatio | str = OutputAspectRatio.VERTICAL_9_16.value,
        caption_theme: CaptionTheme | str = CaptionTheme.TIKTOK.value,
    ) -> str:
        width, height = ClipGenerator.render_dimensions(output_aspect_ratio)
        theme = self._theme_ass_profile(caption_theme, height)
        header = """[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Viral,{font_name},{font_size},{primary_colour},{primary_colour},{outline_colour},{back_colour},-1,0,0,0,100,100,{spacing},0,1,{outline},{shadow},2,{horizontal_margin},{horizontal_margin},{bottom_margin},1
Style: ViralActive,{font_name},{font_size},{active_colour},{active_colour},{outline_colour},{back_colour},-1,0,0,0,100,100,{spacing},0,1,{active_outline},{active_shadow},2,{horizontal_margin},{horizontal_margin},{bottom_margin},1
Style: HookTop,{font_name},{hook_size},{hook_colour},{hook_colour},{outline_colour},{hook_back_colour},-1,0,0,0,100,100,{hook_spacing},0,1,{hook_outline},{hook_shadow},8,{horizontal_margin},{horizontal_margin},{hook_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(
            play_res_x=width,
            play_res_y=height,
            font_name=self.settings.subtitle_font_name,
            font_size=theme["font_size"],
            hook_size=theme["hook_size"],
            horizontal_margin=theme["horizontal_margin"],
            bottom_margin=theme["bottom_margin"],
            hook_margin=theme["hook_margin"],
            primary_colour=theme["primary_colour"],
            active_colour=theme["active_colour"],
            hook_colour=theme["hook_colour"],
            outline_colour=theme["outline_colour"],
            back_colour=theme["back_colour"],
            hook_back_colour=theme["hook_back_colour"],
            spacing=theme["spacing"],
            hook_spacing=theme["hook_spacing"],
            outline=theme["outline"],
            active_outline=theme["active_outline"],
            hook_outline=theme["hook_outline"],
            shadow=theme["shadow"],
            active_shadow=theme["active_shadow"],
            hook_shadow=theme["hook_shadow"],
        )

        events: list[str] = []
        if hook_overlay is not None and hook_overlay.text.strip():
            events.append(
                "Dialogue: 1,"
                f"{self._to_ass_timestamp(hook_overlay.start_seconds)},"
                f"{self._to_ass_timestamp(hook_overlay.end_seconds)},"
                "HookTop,,0,0,0,,"
                r"{\fad(90,140)}"
                f"{self._escape_ass_text(hook_overlay.text)}"
            )

        events.extend(
            [
                "Dialogue: 0,"
                f"{self._to_ass_timestamp(cue.start_seconds)},"
                f"{self._to_ass_timestamp(cue.end_seconds)},"
                "Viral,,0,0,0,,"
                f"{self._animation_tags(caption_theme)}{self._karaoke_cue_to_ass_text(cue)}"
                for cue in cues
            ]
        )
        return header + "\n".join(events).strip() + "\n"

    def _karaoke_cue_to_ass_text(self, cue: KaraokeCue) -> str:
        lines: list[str] = []
        for line in cue.lines:
            rendered_words = []
            for word in line:
                if word.is_active:
                    rendered_words.append(self._highlight_ass_word(word.text))
                else:
                    rendered_words.append(self._plain_ass_word(word.text))
            lines.append(" ".join(rendered_words).strip())
        return r"\N".join(part for part in lines if part)

    @staticmethod
    def _to_ass_timestamp(value: float) -> str:
        total_centiseconds = round(max(value, 0) * 100)
        hours, remainder = divmod(total_centiseconds, 360000)
        minutes, remainder = divmod(remainder, 6000)
        seconds, centiseconds = divmod(remainder, 100)
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

    @staticmethod
    def _escape_ass_text(value: str) -> str:
        return value.replace("\\", r"\\").replace("\n", r"\N").replace("{", r"\{").replace("}", r"\}")

    @staticmethod
    def _animation_tags(caption_theme: CaptionTheme | str) -> str:
        theme = SubtitleGenerator._normalize_caption_theme(caption_theme)
        if theme == CaptionTheme.CINEMATIC.value:
            return r"{\blur0.4}"
        if theme == CaptionTheme.MOTIVATIONAL.value:
            return r"{\blur0.6}"
        return ""

    @staticmethod
    def _plain_ass_word(text: str) -> str:
        return SubtitleGenerator._escape_ass_text(text)

    @staticmethod
    def _highlight_ass_word(text: str) -> str:
        escaped = SubtitleGenerator._escape_ass_text(text)
        return r"{\rViralActive}" + escaped + r"{\rViral}"

    @staticmethod
    def _normalize_caption_theme(value: CaptionTheme | str) -> str:
        if isinstance(value, CaptionTheme):
            return value.value

        normalized = str(value or CaptionTheme.TIKTOK.value).strip().lower()
        allowed = {theme.value for theme in CaptionTheme}
        return normalized if normalized in allowed else CaptionTheme.TIKTOK.value

    def _theme_context_words(self, caption_theme: CaptionTheme | str) -> int:
        _ = self._normalize_caption_theme(caption_theme)
        return 1

    def _theme_ass_profile(self, caption_theme: CaptionTheme | str, height: int) -> dict[str, int | str]:
        theme = self._normalize_caption_theme(caption_theme)
        scale = max(height / 1920, 0.7)
        font_size = max(int(round(self.settings.subtitle_font_size * scale)), 44)
        hook_size = max(font_size - 18, 40)
        horizontal_margin = max(int(round(self.settings.subtitle_margin_horizontal * scale)), 64)
        bottom_margin = max(int(round(self.settings.subtitle_margin_bottom * scale)), 72)
        hook_margin = max(int(round(self.settings.subtitle_hook_margin_top * scale)), 48)

        profiles = {
            CaptionTheme.TIKTOK.value: {
                "primary_colour": self._hex_to_ass_color("#FFFFFF"),
                "active_colour": self._hex_to_ass_color("#2EF2FF"),
                "hook_colour": self._hex_to_ass_color("#FFFFFF"),
                "outline_colour": self._hex_to_ass_color("#101010"),
                "back_colour": "&H50000000",
                "hook_back_colour": "&H14000000",
                "spacing": 0,
                "hook_spacing": 0,
                "outline": 5,
                "active_outline": 7,
                "hook_outline": 4,
                "shadow": 0,
                "active_shadow": 0,
                "hook_shadow": 0,
            },
            CaptionTheme.CINEMATIC.value: {
                "primary_colour": self._hex_to_ass_color("#F8F1D8"),
                "active_colour": self._hex_to_ass_color("#FFD36B"),
                "hook_colour": self._hex_to_ass_color("#F8F1D8"),
                "outline_colour": self._hex_to_ass_color("#080808"),
                "back_colour": "&H24000000",
                "hook_back_colour": "&H00000000",
                "spacing": 2,
                "hook_spacing": 3,
                "outline": 4,
                "active_outline": 5,
                "hook_outline": 3,
                "shadow": 0,
                "active_shadow": 0,
                "hook_shadow": 0,
            },
            CaptionTheme.MOTIVATIONAL.value: {
                "primary_colour": self._hex_to_ass_color("#FFF6E1"),
                "active_colour": self._hex_to_ass_color("#FFB347"),
                "hook_colour": self._hex_to_ass_color("#FFDB7D"),
                "outline_colour": self._hex_to_ass_color("#0D0D12"),
                "back_colour": "&H30000000",
                "hook_back_colour": "&H00000000",
                "spacing": 1,
                "hook_spacing": 1,
                "outline": 5,
                "active_outline": 6,
                "hook_outline": 3,
                "shadow": 0,
                "active_shadow": 0,
                "hook_shadow": 0,
            },
        }
        profile = profiles[theme]
        return {
            **profile,
            "font_size": font_size,
            "hook_size": hook_size,
            "horizontal_margin": horizontal_margin,
            "bottom_margin": bottom_margin,
            "hook_margin": hook_margin,
        }

    @staticmethod
    def _hex_to_ass_color(value: str) -> str:
        normalized = value.lstrip("#")
        if len(normalized) != 6:
            return "&H00FFFFFF"
        red = normalized[0:2]
        green = normalized[2:4]
        blue = normalized[4:6]
        return f"&H00{blue}{green}{red}"
