from __future__ import annotations

from pydantic import BaseModel, Field

from src.core.models import CaptionTheme, JobStatus, OutputAspectRatio


class ProcessVideoRequest(BaseModel):
    url: str = Field(..., min_length=1, description="Source video URL to process.")
    output_aspect_ratio: OutputAspectRatio = Field(
        default=OutputAspectRatio.VERTICAL_9_16,
        description="Target output aspect ratio for rendered clips.",
    )
    caption_theme: CaptionTheme = Field(
        default=CaptionTheme.TIKTOK,
        description="Caption and overlay vibe for the rendered clips.",
    )
    ollama_model: str | None = Field(
        default=None,
        min_length=1,
        description="Installed Ollama model name to use for viral moment analysis.",
    )


class ProcessVideoResponse(BaseModel):
    job_id: str
    video_id: str
    status: JobStatus
    output_aspect_ratio: OutputAspectRatio
    caption_theme: CaptionTheme
    ollama_model: str


class JobStatusResponse(BaseModel):
    id: str
    video_id: str
    source_url: str
    output_aspect_ratio: OutputAspectRatio
    caption_theme: CaptionTheme
    ollama_model: str
    status: JobStatus
    current_step: str
    step_progress_current: int = 0
    step_progress_total: int = 0
    active_task: str | None = None
    error_message: str | None
    video_path: str | None
    audio_path: str | None
    transcript_path: str | None
    manifest_path: str | None
    created_at: str
    updated_at: str
    completed_at: str | None
    clip_count: int
    video_media_url: str | None = None
    audio_media_url: str | None = None
    transcript_media_url: str | None = None
    manifest_media_url: str | None = None


class RecentJobResponse(BaseModel):
    id: str
    video_id: str
    source_url: str
    output_aspect_ratio: OutputAspectRatio
    caption_theme: CaptionTheme
    ollama_model: str
    status: JobStatus
    current_step: str
    step_progress_current: int = 0
    step_progress_total: int = 0
    active_task: str | None = None
    error_message: str | None
    created_at: str
    updated_at: str
    completed_at: str | None
    clip_count: int


class RecentJobsResponse(BaseModel):
    jobs: list[RecentJobResponse]


class OllamaModelsResponse(BaseModel):
    models: list[str]
    default_model: str
    available: bool


class DeleteJobResponse(BaseModel):
    job_id: str
    video_id: str
    deleted_paths: list[str]
    deleted_bytes: int


class JobEventResponse(BaseModel):
    id: int
    job_id: str
    step: str
    level: str
    message: str
    created_at: str


class JobEventsResponse(BaseModel):
    job_id: str
    events: list[JobEventResponse]


class ClipResponse(BaseModel):
    id: int
    job_id: str
    video_id: str
    sequence_number: int
    file_path: str
    subtitle_path: str | None
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    hook: str | None
    reason: str | None
    score: float | None
    created_at: str
    media_url: str
    subtitle_url: str | None = None


class ClipListResponse(BaseModel):
    video_id: str
    clips: list[ClipResponse]
