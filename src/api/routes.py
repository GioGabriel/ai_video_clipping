from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request, status

from src.api.schemas import (
    ClipListResponse,
    DeleteJobResponse,
    JobEventsResponse,
    JobStatusResponse,
    OllamaModelsResponse,
    ProcessVideoRequest,
    ProcessVideoResponse,
    RecentJobsResponse,
)
from src.core.container import AppContainer
from src.core.models import JobStatus

router = APIRouter()


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def build_media_url(container: AppContainer, file_path: str | None) -> str | None:
    if not file_path:
        return None

    try:
        relative_path = Path(file_path).resolve().relative_to(container.settings.data_dir.resolve())
    except ValueError:
        return None

    return f"/media/{relative_path.as_posix()}"


@router.post("/process-video", response_model=ProcessVideoResponse, status_code=status.HTTP_202_ACCEPTED)
def process_video(payload: ProcessVideoRequest, request: Request) -> ProcessVideoResponse:
    container = get_container(request)
    job_id = str(uuid4())
    video_id = uuid4().hex[:12]
    selected_model = str(payload.ollama_model or container.settings.ollama_model).strip() or container.settings.ollama_model

    container.job_repository.create(
        job_id=job_id,
        video_id=video_id,
        source_url=payload.url,
        output_aspect_ratio=payload.output_aspect_ratio,
        caption_theme=payload.caption_theme,
        ollama_model=selected_model,
    )
    container.job_event_repository.create(
        job_id=job_id,
        step="queued",
        level="info",
        message=f"Job accepted by API and queued for local processing with Ollama model '{selected_model}'.",
    )
    container.job_queue.submit(job_id)

    return ProcessVideoResponse(
        job_id=job_id,
        video_id=video_id,
        status=JobStatus.QUEUED,
        output_aspect_ratio=payload.output_aspect_ratio,
        caption_theme=payload.caption_theme,
        ollama_model=selected_model,
    )


@router.get("/ollama/models", response_model=OllamaModelsResponse)
def list_ollama_models(request: Request) -> OllamaModelsResponse:
    container = get_container(request)
    catalog = container.ollama_runtime.list_models()
    return OllamaModelsResponse(
        models=catalog.models,
        default_model=catalog.default_model,
        available=catalog.available,
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str, request: Request) -> JobStatusResponse:
    container = get_container(request)
    job = container.job_repository.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    job["video_media_url"] = build_media_url(container, job.get("video_path"))
    job["audio_media_url"] = build_media_url(container, job.get("audio_path"))
    job["transcript_media_url"] = build_media_url(container, job.get("transcript_path"))
    job["manifest_media_url"] = build_media_url(container, job.get("manifest_path"))
    return JobStatusResponse(**job)


@router.get("/jobs", response_model=RecentJobsResponse)
def list_jobs(
    request: Request,
    limit: int = Query(default=10, ge=1, le=50),
) -> RecentJobsResponse:
    container = get_container(request)
    jobs = container.job_repository.list_recent(limit=limit)
    return RecentJobsResponse(jobs=jobs)


@router.delete("/jobs/{job_id}", response_model=DeleteJobResponse)
def delete_job(job_id: str, request: Request) -> DeleteJobResponse:
    container = get_container(request)
    try:
        result = container.job_cleanup.delete_job(job_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return DeleteJobResponse(
        job_id=result.job_id,
        video_id=result.video_id,
        deleted_paths=result.deleted_paths,
        deleted_bytes=result.deleted_bytes,
    )


@router.get("/events/{job_id}", response_model=JobEventsResponse)
def list_job_events(job_id: str, request: Request) -> JobEventsResponse:
    container = get_container(request)
    job = container.job_repository.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    events = container.job_event_repository.list_by_job_id(job_id)
    if not events:
        events = build_fallback_events(job)

    return JobEventsResponse(job_id=job_id, events=events)


@router.get("/clips/{video_id}", response_model=ClipListResponse)
def list_clips(video_id: str, request: Request) -> ClipListResponse:
    container = get_container(request)
    job = container.job_repository.get_by_video_id(video_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found.")

    clips = container.clip_repository.list_by_video_id(video_id)
    for clip in clips:
        clip["media_url"] = build_media_url(container, clip.get("file_path")) or ""
        clip["subtitle_url"] = build_media_url(container, clip.get("subtitle_path"))
    return ClipListResponse(video_id=video_id, clips=clips)


def build_fallback_events(job: dict[str, object]) -> list[dict[str, object]]:
    fallback_events: list[dict[str, object]] = [
        {
            "id": 0,
            "job_id": job["id"],
            "step": "queued",
            "level": "info",
            "message": "Job exists in the database. This job was created before detailed event capture was enabled.",
            "created_at": job["created_at"],
        }
    ]

    artifact_map = [
        ("download_video", "video_path", "Source video is present on disk."),
        ("extract_audio", "audio_path", "Extracted audio file is present on disk."),
        ("transcribe_audio", "transcript_path", "Transcript artifact is present on disk."),
        ("export_assets", "manifest_path", "Manifest artifact is present on disk."),
    ]
    synthetic_id = 1
    for step, field, message in artifact_map:
        if job.get(field):
            fallback_events.append(
                {
                    "id": synthetic_id,
                    "job_id": job["id"],
                    "step": step,
                    "level": "success",
                    "message": message,
                    "created_at": job["updated_at"],
                }
            )
            synthetic_id += 1

    if job.get("error_message"):
        fallback_events.append(
            {
                "id": synthetic_id,
                "job_id": job["id"],
                "step": str(job.get("current_step") or "failed"),
                "level": "error",
                "message": str(job["error_message"]),
                "created_at": job["updated_at"],
            }
        )

    return fallback_events
