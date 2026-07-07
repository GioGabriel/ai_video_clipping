from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStep(str, Enum):
    QUEUED = "queued"
    DOWNLOAD = "download_video"
    EXTRACT_AUDIO = "extract_audio"
    TRANSCRIBE = "transcribe_audio"
    SEGMENT_TRANSCRIPT = "segment_transcript"
    CLASSIFY_VIRAL_PHRASES = "classify_viral_phrases"
    DETECT_MOMENTS = "detect_viral_moments"
    DETECT_HOOKS = "detect_hooks"
    COMPLETE_THOUGHTS = "complete_thoughts"
    OPTIMIZE_CLIPS = "optimize_clips"
    GENERATE_CLIPS = "generate_clips"
    GENERATE_SUBTITLES = "generate_subtitles"
    EXPORT = "export_assets"
    COMPLETED = "completed"
    FAILED = "failed"


class OutputAspectRatio(str, Enum):
    VERTICAL_9_16 = "9:16"
    LANDSCAPE_16_9 = "16:9"
    SQUARE_1_1 = "1:1"
    PORTRAIT_4_5 = "4:5"


class CaptionTheme(str, Enum):
    TIKTOK = "tiktok"
    CINEMATIC = "cinematic"
    MOTIVATIONAL = "motivational"


@dataclass(slots=True)
class TranscriptWord:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(slots=True)
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str
    words: list[TranscriptWord] | None = None


@dataclass(slots=True)
class SentenceBlock:
    start_seconds: float
    end_seconds: float
    text: str
    speaker: str = "speaker_1"
    source_segment_start_index: int | None = None
    source_segment_end_index: int | None = None
    phrase_score: float = 0.0
    detected_triggers: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(slots=True)
class TranscriptArtifact:
    video_id: str
    language: str
    text: str
    transcript_path: Path
    segments: list[TranscriptSegment]
    sentence_blocks: list[SentenceBlock] | None = None


@dataclass(slots=True)
class ViralMoment:
    start_seconds: float
    end_seconds: float
    score: float
    hook: str
    reason: str
    hook_start_seconds: float | None = None
    core_start_seconds: float | None = None
    core_end_seconds: float | None = None
    hook_strength: float = 0.0
    emotion_level: float = 0.0
    statement_strength: float = 0.0
    novelty: float = 0.0
    duration_score: float = 0.0
    phrase_score: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(slots=True)
class SubtitleArtifact:
    sidecar_path: Path
    styled_path: Path


@dataclass(slots=True)
class KaraokeWord:
    text: str
    is_active: bool = False


@dataclass(slots=True)
class KaraokeCue:
    start_seconds: float
    end_seconds: float
    lines: list[list[KaraokeWord]]
    spoken_text: str


@dataclass(slots=True)
class SubtitleBurnInResult:
    burned_count: int
    warning_message: str | None = None


@dataclass(slots=True)
class ZoomEffect:
    start_seconds: float
    end_seconds: float
    peak_scale: float
    anchor_text: str | None = None


@dataclass(slots=True)
class HookOverlay:
    text: str
    start_seconds: float
    end_seconds: float


@dataclass(slots=True)
class SpeakerFocusPoint:
    time_seconds: float
    center_x: float
    center_y: float


@dataclass(slots=True)
class SpeakerFocusTrack:
    source_width: int
    source_height: int
    points: list[SpeakerFocusPoint]


@dataclass(slots=True)
class EditPlan:
    start_seconds: float
    end_seconds: float
    score: float
    hook: str
    reason: str
    zoom_effects: list[ZoomEffect]
    hook_overlay: HookOverlay | None = None

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(slots=True)
class ClipArtifact:
    job_id: str
    video_id: str
    sequence_number: int
    file_path: Path
    start_seconds: float
    end_seconds: float
    hook: str
    reason: str
    score: float
    subtitle_path: Path | None = None
    styled_subtitle_path: Path | None = None

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds
