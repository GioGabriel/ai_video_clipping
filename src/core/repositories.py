from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.core.database import Database
from src.core.models import CaptionTheme, ClipArtifact, JobStatus, OutputAspectRatio, PipelineStep


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(
        self,
        job_id: str,
        video_id: str,
        source_url: str,
        output_aspect_ratio: OutputAspectRatio | str = OutputAspectRatio.VERTICAL_9_16.value,
        caption_theme: CaptionTheme | str = CaptionTheme.TIKTOK.value,
        ollama_model: str | None = None,
    ) -> None:
        now = utcnow_iso()
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, video_id, source_url, output_aspect_ratio, caption_theme, ollama_model, status, current_step,
                    step_progress_current, step_progress_total, active_task, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    video_id,
                    source_url,
                    self._normalize_output_aspect_ratio(output_aspect_ratio),
                    self._normalize_caption_theme(caption_theme),
                    self._normalize_ollama_model(ollama_model),
                    JobStatus.QUEUED.value,
                    PipelineStep.QUEUED.value,
                    0,
                    0,
                    "Queued for local processing.",
                    now,
                    now,
                ),
            )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT jobs.*,
                       (SELECT COUNT(*) FROM clips WHERE clips.job_id = jobs.id) AS clip_count
                FROM jobs
                WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
        return self._normalize_job_row(dict(row)) if row else None

    def get_by_video_id(self, video_id: str) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT jobs.*,
                       (SELECT COUNT(*) FROM clips WHERE clips.job_id = jobs.id) AS clip_count
                FROM jobs
                WHERE video_id = ?
                """,
                (video_id,),
            ).fetchone()
        return self._normalize_job_row(dict(row)) if row else None

    def list_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT jobs.*,
                       (SELECT COUNT(*) FROM clips WHERE clips.job_id = jobs.id) AS clip_count
                FROM jobs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._normalize_job_row(dict(row)) for row in rows]

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return

        fields["updated_at"] = utcnow_iso()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = list(fields.values()) + [job_id]

        with self.database.connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?",
                values,
            )

    def delete(self, job_id: str) -> None:
        with self.database.connect() as connection:
            connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def reconcile_incomplete_jobs(self, error_message: str) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM jobs
                WHERE status IN (?, ?)
                ORDER BY created_at ASC
                """,
                (JobStatus.QUEUED.value, JobStatus.RUNNING.value),
            ).fetchall()

            if not rows:
                return []

            now = utcnow_iso()
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, error_message = ?, completed_at = ?, updated_at = ?
                WHERE status IN (?, ?)
                """,
                (
                    JobStatus.FAILED.value,
                    error_message,
                    now,
                    now,
                    JobStatus.QUEUED.value,
                    JobStatus.RUNNING.value,
                ),
            )

        reconciled: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["status"] = JobStatus.FAILED.value
            item["error_message"] = error_message
            item["completed_at"] = now
            item["updated_at"] = now
            reconciled.append(self._normalize_job_row(item))
        return reconciled

    @staticmethod
    def _normalize_job_row(row: dict[str, Any]) -> dict[str, Any]:
        if "output_aspect_ratio" in row:
            row["output_aspect_ratio"] = JobRepository._normalize_output_aspect_ratio(row["output_aspect_ratio"])
        row["caption_theme"] = JobRepository._normalize_caption_theme(row.get("caption_theme"))
        row["ollama_model"] = JobRepository._normalize_ollama_model(row.get("ollama_model"))
        if "step_progress_current" in row:
            row["step_progress_current"] = int(row.get("step_progress_current") or 0)
        if "step_progress_total" in row:
            row["step_progress_total"] = int(row.get("step_progress_total") or 0)
        if "active_task" in row and row["active_task"] is not None:
            row["active_task"] = str(row["active_task"])
        return row

    @staticmethod
    def _normalize_output_aspect_ratio(value: OutputAspectRatio | str | None) -> str:
        if isinstance(value, OutputAspectRatio):
            return value.value

        normalized = str(value or OutputAspectRatio.VERTICAL_9_16.value).strip()
        mapping = {
            "OutputAspectRatio.VERTICAL_9_16": OutputAspectRatio.VERTICAL_9_16.value,
            "OutputAspectRatio.LANDSCAPE_16_9": OutputAspectRatio.LANDSCAPE_16_9.value,
            "OutputAspectRatio.SQUARE_1_1": OutputAspectRatio.SQUARE_1_1.value,
            "OutputAspectRatio.PORTRAIT_4_5": OutputAspectRatio.PORTRAIT_4_5.value,
        }
        return mapping.get(normalized, normalized)

    @staticmethod
    def _normalize_ollama_model(value: str | None) -> str:
        normalized = str(value or "llama3").strip()
        return normalized or "llama3"

    @staticmethod
    def _normalize_caption_theme(value: CaptionTheme | str | None) -> str:
        if isinstance(value, CaptionTheme):
            return value.value

        normalized = str(value or CaptionTheme.TIKTOK.value).strip().lower()
        allowed = {theme.value for theme in CaptionTheme}
        return normalized if normalized in allowed else CaptionTheme.TIKTOK.value


class ClipRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def replace_for_job(self, job_id: str, video_id: str, clips: list[ClipArtifact]) -> None:
        created_at = utcnow_iso()
        with self.database.connect() as connection:
            connection.execute("DELETE FROM clips WHERE job_id = ?", (job_id,))
            connection.executemany(
                """
                INSERT INTO clips (
                    job_id,
                    video_id,
                    sequence_number,
                    file_path,
                    subtitle_path,
                    start_seconds,
                    end_seconds,
                    duration_seconds,
                    hook,
                    reason,
                    score,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        job_id,
                        video_id,
                        clip.sequence_number,
                        str(clip.file_path),
                        str(clip.subtitle_path) if clip.subtitle_path else None,
                        clip.start_seconds,
                        clip.end_seconds,
                        clip.duration_seconds,
                        clip.hook,
                        clip.reason,
                        clip.score,
                        created_at,
                    )
                    for clip in clips
                ],
            )

    def list_by_video_id(self, video_id: str) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM clips
                WHERE video_id = ?
                ORDER BY sequence_number ASC
                """,
                (video_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_by_job_id(self, job_id: str) -> None:
        with self.database.connect() as connection:
            connection.execute("DELETE FROM clips WHERE job_id = ?", (job_id,))


class JobEventRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, job_id: str, step: str, level: str, message: str, created_at: str | None = None) -> None:
        timestamp = created_at or utcnow_iso()
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO job_events (job_id, step, level, message, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, step, level, message, timestamp),
            )

    def list_by_job_id(self, job_id: str) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM job_events
                WHERE job_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (job_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_by_job_id(self, job_id: str) -> None:
        with self.database.connect() as connection:
            connection.execute("DELETE FROM job_events WHERE job_id = ?", (job_id,))
