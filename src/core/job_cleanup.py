from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from src.core.config import AppSettings
from src.core.models import JobStatus
from src.core.repositories import ClipRepository, JobEventRepository, JobRepository


@dataclass(slots=True)
class JobCleanupResult:
    job_id: str
    video_id: str
    deleted_paths: list[str]
    deleted_bytes: int


class JobCleanupService:
    def __init__(
        self,
        settings: AppSettings,
        job_repository: JobRepository,
        job_event_repository: JobEventRepository,
        clip_repository: ClipRepository,
    ) -> None:
        self.settings = settings
        self.job_repository = job_repository
        self.job_event_repository = job_event_repository
        self.clip_repository = clip_repository

    def delete_job(self, job_id: str) -> JobCleanupResult:
        job = self.job_repository.get(job_id)
        if not job:
            raise KeyError(job_id)

        if job["status"] in {JobStatus.QUEUED.value, JobStatus.RUNNING.value}:
            raise RuntimeError("Active jobs cannot be deleted yet. Wait for them to finish or fail.")

        video_id = str(job["video_id"])
        deleted_paths: list[str] = []
        deleted_bytes = 0

        candidate_paths = self._collect_candidate_paths(job, video_id)
        for path in candidate_paths:
            removed_bytes = self._delete_path(path)
            if removed_bytes <= 0:
                continue
            deleted_paths.append(str(path))
            deleted_bytes += removed_bytes

        self.clip_repository.delete_by_job_id(job_id)
        self.job_event_repository.delete_by_job_id(job_id)
        self.job_repository.delete(job_id)

        return JobCleanupResult(
            job_id=job_id,
            video_id=video_id,
            deleted_paths=deleted_paths,
            deleted_bytes=deleted_bytes,
        )

    def _collect_candidate_paths(self, job: dict[str, object], video_id: str) -> list[Path]:
        candidates: list[Path] = []

        for field in ("video_path", "audio_path", "transcript_path", "manifest_path"):
            value = job.get(field)
            if value:
                candidates.append(Path(str(value)))

        candidates.extend(self.settings.videos_dir.glob(f"{video_id}*"))
        candidates.extend(self.settings.audio_dir.glob(f"{video_id}*"))
        candidates.extend(self.settings.transcripts_dir.glob(f"{video_id}*"))

        clip_directory = self.settings.clips_dir / video_id
        if clip_directory.exists():
            candidates.append(clip_directory)

        legacy_global_manifest = self.settings.clips_dir / "manifest.json"
        if legacy_global_manifest.exists():
            candidates.append(legacy_global_manifest)

        deduplicated: list[Path] = []
        seen: set[Path] = set()
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            if not self._is_within_data_dir(resolved):
                continue
            seen.add(resolved)
            deduplicated.append(resolved)

        return deduplicated

    def _delete_path(self, path: Path) -> int:
        if not path.exists():
            return 0

        if path.is_dir():
            total_bytes = sum(
                child.stat().st_size
                for child in path.rglob("*")
                if child.is_file()
            )
            shutil.rmtree(path)
            return total_bytes

        file_size = path.stat().st_size
        path.unlink()
        return file_size

    def _is_within_data_dir(self, path: Path) -> bool:
        try:
            path.relative_to(self.settings.data_dir.resolve())
        except ValueError:
            return False
        return True
