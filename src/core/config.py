from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        cleaned_value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), cleaned_value)


def _bool_from_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_from_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def _int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


@dataclass(slots=True)
class AppSettings:
    project_root: Path
    app_name: str
    app_host: str
    app_port: int
    log_level: str
    data_dir: Path
    videos_dir: Path
    audio_dir: Path
    transcripts_dir: Path
    clips_dir: Path
    database_path: Path
    yt_dlp_binary: str
    ffmpeg_binary: str
    yt_dlp_skip_certificate_check: bool
    ollama_base_url: str
    ollama_model: str
    ollama_request_timeout_seconds: int
    whisper_model: str
    whisper_device: str
    job_worker_count: int
    max_clip_candidates: int
    max_candidates_per_chunk: int
    min_clip_duration_seconds: float
    max_clip_duration_seconds: float
    target_clip_duration_seconds: float
    viral_chunk_duration_seconds: float
    viral_chunk_overlap_seconds: float
    viral_min_score: float
    subtitle_generation_enabled: bool
    download_format: str = "mp4/bestvideo+bestaudio/best"
    clip_lead_in_seconds: float = 0.8
    clip_trailing_pad_seconds: float = 1.2
    clip_sentence_gap_threshold_seconds: float = 1.6
    clip_sentence_hard_gap_threshold_seconds: float = 3.0
    clip_thought_max_duration_seconds: float = 180.0
    hook_detector_lookback_seconds: float = 15.0
    thought_completion_pause_seconds: float = 0.8
    transcript_sentence_pause_seconds: float = 0.8
    clip_optimizer_split_tolerance_seconds: float = 6.0
    subtitle_burn_in_enabled: bool = True
    subtitle_words_per_cue: int = 4
    subtitle_max_chars_per_cue: int = 26
    subtitle_font_name: str = "Arial Black"
    subtitle_font_size: int = 80
    subtitle_max_chars_per_line: int = 16
    subtitle_margin_horizontal: int = 120
    subtitle_margin_bottom: int = 230
    subtitle_hook_margin_top: int = 160
    edit_plan_zoom_enabled: bool = False
    edit_plan_max_zoom_beats: int = 4
    edit_plan_zoom_pre_roll_seconds: float = 0.08
    edit_plan_zoom_post_roll_seconds: float = 0.45
    edit_plan_zoom_min_scale: float = 1.08
    edit_plan_zoom_max_scale: float = 1.14
    edit_plan_zoom_min_gap_seconds: float = 1.8
    edit_plan_hook_overlay_seconds: float = 1.8
    kinetic_caption_context_words: int = 3
    overlay_hook_card_enabled: bool = True
    overlay_hook_card_position: str = "center"
    overlay_active_word_scale: float = 1.12
    overlay_active_word_color: str = "#2EF2FF"
    overlay_caption_shadow_opacity: int = 132
    overlay_card_max_width_ratio: float = 0.84
    overlay_card_padding_horizontal: int = 38
    overlay_card_padding_vertical: int = 22
    overlay_card_radius: int = 34
    overlay_card_margin_top: int = 72
    overlay_card_center_offset: int = 0
    speaker_tracking_enabled: bool = True
    speaker_tracking_sample_interval_seconds: float = 0.45
    speaker_tracking_smoothing: float = 0.68
    export_fps: int = 30
    export_video_bitrate: str = "6M"
    export_video_maxrate: str = "8M"
    export_video_bufsize: str = "12M"
    export_audio_bitrate: str = "320k"
    default_caption_theme: str = "tiktok"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


def load_settings() -> AppSettings:
    project_root = Path(__file__).resolve().parents[2]
    _load_dotenv(project_root)
    data_dir = project_root / os.getenv("DATA_DIR", "data")
    database_path = project_root / os.getenv("DATABASE_PATH", "data/app.db")

    settings = AppSettings(
        project_root=project_root,
        app_name=os.getenv("APP_NAME", "AI Clipping System"),
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=_int_from_env("APP_PORT", 8000),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        data_dir=data_dir,
        videos_dir=data_dir / "videos",
        audio_dir=data_dir / "audio",
        transcripts_dir=data_dir / "transcripts",
        clips_dir=data_dir / "clips",
        database_path=database_path,
        yt_dlp_binary=os.getenv("YT_DLP_BINARY", "yt-dlp"),
        ffmpeg_binary=os.getenv("FFMPEG_BINARY", "ffmpeg"),
        yt_dlp_skip_certificate_check=_bool_from_env("YT_DLP_SKIP_CERTIFICATE_CHECK", False),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
        ollama_request_timeout_seconds=_int_from_env("OLLAMA_REQUEST_TIMEOUT_SECONDS", 180),
        whisper_model=os.getenv("WHISPER_MODEL", "base"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        job_worker_count=_int_from_env("JOB_WORKER_COUNT", 1),
        max_clip_candidates=_int_from_env("MAX_CLIP_CANDIDATES", 10),
        max_candidates_per_chunk=_int_from_env("MAX_CANDIDATES_PER_CHUNK", 3),
        min_clip_duration_seconds=_float_from_env("MIN_CLIP_DURATION_SECONDS", 18),
        max_clip_duration_seconds=_float_from_env("MAX_CLIP_DURATION_SECONDS", 50),
        target_clip_duration_seconds=_float_from_env("TARGET_CLIP_DURATION_SECONDS", 30),
        viral_chunk_duration_seconds=_float_from_env("VIRAL_CHUNK_DURATION_SECONDS", 420),
        viral_chunk_overlap_seconds=_float_from_env("VIRAL_CHUNK_OVERLAP_SECONDS", 60),
        viral_min_score=_float_from_env("VIRAL_MIN_SCORE", 60),
        subtitle_generation_enabled=_bool_from_env("SUBTITLE_GENERATION_ENABLED", True),
        clip_lead_in_seconds=_float_from_env("CLIP_LEAD_IN_SECONDS", 0.8),
        clip_trailing_pad_seconds=_float_from_env("CLIP_TRAILING_PAD_SECONDS", 1.2),
        clip_sentence_gap_threshold_seconds=_float_from_env("CLIP_SENTENCE_GAP_THRESHOLD_SECONDS", 1.6),
        clip_sentence_hard_gap_threshold_seconds=_float_from_env("CLIP_SENTENCE_HARD_GAP_THRESHOLD_SECONDS", 3.0),
        clip_thought_max_duration_seconds=_float_from_env("CLIP_THOUGHT_MAX_DURATION_SECONDS", 180.0),
        hook_detector_lookback_seconds=_float_from_env("HOOK_DETECTOR_LOOKBACK_SECONDS", 15.0),
        thought_completion_pause_seconds=_float_from_env("THOUGHT_COMPLETION_PAUSE_SECONDS", 0.8),
        transcript_sentence_pause_seconds=_float_from_env("TRANSCRIPT_SENTENCE_PAUSE_SECONDS", 0.8),
        clip_optimizer_split_tolerance_seconds=_float_from_env("CLIP_OPTIMIZER_SPLIT_TOLERANCE_SECONDS", 6.0),
        subtitle_burn_in_enabled=_bool_from_env("SUBTITLE_BURN_IN_ENABLED", True),
        subtitle_words_per_cue=_int_from_env("SUBTITLE_WORDS_PER_CUE", 4),
        subtitle_max_chars_per_cue=_int_from_env("SUBTITLE_MAX_CHARS_PER_CUE", 26),
        subtitle_font_name=os.getenv("SUBTITLE_FONT_NAME", "Arial Black"),
        subtitle_font_size=_int_from_env("SUBTITLE_FONT_SIZE", 80),
        subtitle_max_chars_per_line=_int_from_env("SUBTITLE_MAX_CHARS_PER_LINE", 16),
        subtitle_margin_horizontal=_int_from_env("SUBTITLE_MARGIN_HORIZONTAL", 120),
        subtitle_margin_bottom=_int_from_env("SUBTITLE_MARGIN_BOTTOM", 230),
        subtitle_hook_margin_top=_int_from_env("SUBTITLE_HOOK_MARGIN_TOP", 160),
        edit_plan_zoom_enabled=_bool_from_env("EDIT_PLAN_ZOOM_ENABLED", False),
        edit_plan_max_zoom_beats=_int_from_env("EDIT_PLAN_MAX_ZOOM_BEATS", 4),
        edit_plan_zoom_pre_roll_seconds=_float_from_env("EDIT_PLAN_ZOOM_PRE_ROLL_SECONDS", 0.08),
        edit_plan_zoom_post_roll_seconds=_float_from_env("EDIT_PLAN_ZOOM_POST_ROLL_SECONDS", 0.45),
        edit_plan_zoom_min_scale=_float_from_env("EDIT_PLAN_ZOOM_MIN_SCALE", 1.08),
        edit_plan_zoom_max_scale=_float_from_env("EDIT_PLAN_ZOOM_MAX_SCALE", 1.14),
        edit_plan_zoom_min_gap_seconds=_float_from_env("EDIT_PLAN_ZOOM_MIN_GAP_SECONDS", 1.8),
        edit_plan_hook_overlay_seconds=_float_from_env("EDIT_PLAN_HOOK_OVERLAY_SECONDS", 1.8),
        kinetic_caption_context_words=_int_from_env("KINETIC_CAPTION_CONTEXT_WORDS", 3),
        overlay_hook_card_enabled=_bool_from_env("OVERLAY_HOOK_CARD_ENABLED", True),
        overlay_hook_card_position=os.getenv("OVERLAY_HOOK_CARD_POSITION", "center"),
        overlay_active_word_scale=_float_from_env("OVERLAY_ACTIVE_WORD_SCALE", 1.12),
        overlay_active_word_color=os.getenv("OVERLAY_ACTIVE_WORD_COLOR", "#2EF2FF"),
        overlay_caption_shadow_opacity=_int_from_env("OVERLAY_CAPTION_SHADOW_OPACITY", 132),
        overlay_card_max_width_ratio=_float_from_env("OVERLAY_CARD_MAX_WIDTH_RATIO", 0.84),
        overlay_card_padding_horizontal=_int_from_env("OVERLAY_CARD_PADDING_HORIZONTAL", 38),
        overlay_card_padding_vertical=_int_from_env("OVERLAY_CARD_PADDING_VERTICAL", 22),
        overlay_card_radius=_int_from_env("OVERLAY_CARD_RADIUS", 34),
        overlay_card_margin_top=_int_from_env("OVERLAY_CARD_MARGIN_TOP", 72),
        overlay_card_center_offset=_int_from_env("OVERLAY_CARD_CENTER_OFFSET", 0),
        speaker_tracking_enabled=_bool_from_env("SPEAKER_TRACKING_ENABLED", True),
        speaker_tracking_sample_interval_seconds=_float_from_env("SPEAKER_TRACKING_SAMPLE_INTERVAL_SECONDS", 0.45),
        speaker_tracking_smoothing=_float_from_env("SPEAKER_TRACKING_SMOOTHING", 0.68),
        export_fps=_int_from_env("EXPORT_FPS", 30),
        export_video_bitrate=os.getenv("EXPORT_VIDEO_BITRATE", "6M"),
        export_video_maxrate=os.getenv("EXPORT_VIDEO_MAXRATE", "8M"),
        export_video_bufsize=os.getenv("EXPORT_VIDEO_BUFSIZE", "12M"),
        export_audio_bitrate=os.getenv("EXPORT_AUDIO_BITRATE", "320k"),
        default_caption_theme=os.getenv("DEFAULT_CAPTION_THEME", "tiktok"),
    )
    settings.ensure_directories()
    return settings
