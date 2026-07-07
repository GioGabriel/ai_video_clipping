from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL UNIQUE,
    source_url TEXT NOT NULL,
    output_aspect_ratio TEXT NOT NULL DEFAULT '9:16',
    caption_theme TEXT NOT NULL DEFAULT 'tiktok',
    ollama_model TEXT NOT NULL DEFAULT 'llama3',
    status TEXT NOT NULL,
    current_step TEXT NOT NULL,
    step_progress_current INTEGER NOT NULL DEFAULT 0,
    step_progress_total INTEGER NOT NULL DEFAULT 0,
    active_task TEXT,
    error_message TEXT,
    video_path TEXT,
    audio_path TEXT,
    transcript_path TEXT,
    manifest_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    subtitle_path TEXT,
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    hook TEXT,
    reason TEXT,
    score REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id)
);

CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    step TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_clips_video_id ON clips(video_id);
CREATE INDEX IF NOT EXISTS idx_clips_job_id ON clips(job_id);
CREATE INDEX IF NOT EXISTS idx_job_events_job_id_created_at ON job_events(job_id, created_at);
"""


class Database:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL;")
            connection.execute("PRAGMA synchronous=NORMAL;")
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            self._ensure_column(
                connection,
                table_name="jobs",
                column_name="output_aspect_ratio",
                column_definition="TEXT NOT NULL DEFAULT '9:16'",
            )
            self._ensure_column(
                connection,
                table_name="jobs",
                column_name="step_progress_current",
                column_definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                connection,
                table_name="jobs",
                column_name="ollama_model",
                column_definition="TEXT NOT NULL DEFAULT 'llama3'",
            )
            self._ensure_column(
                connection,
                table_name="jobs",
                column_name="caption_theme",
                column_definition="TEXT NOT NULL DEFAULT 'tiktok'",
            )
            self._ensure_column(
                connection,
                table_name="jobs",
                column_name="step_progress_total",
                column_definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                connection,
                table_name="jobs",
                column_name="active_task",
                column_definition="TEXT",
            )
            connection.executescript(
                """
                UPDATE jobs SET output_aspect_ratio = '9:16'
                WHERE output_aspect_ratio = 'OutputAspectRatio.VERTICAL_9_16';

                UPDATE jobs SET output_aspect_ratio = '16:9'
                WHERE output_aspect_ratio = 'OutputAspectRatio.LANDSCAPE_16_9';

                UPDATE jobs SET output_aspect_ratio = '1:1'
                WHERE output_aspect_ratio = 'OutputAspectRatio.SQUARE_1_1';

                UPDATE jobs SET output_aspect_ratio = '4:5'
                WHERE output_aspect_ratio = 'OutputAspectRatio.PORTRAIT_4_5';

                UPDATE jobs SET ollama_model = 'llama3'
                WHERE ollama_model IS NULL OR TRIM(ollama_model) = '';

                UPDATE jobs SET caption_theme = 'tiktok'
                WHERE caption_theme IS NULL OR TRIM(caption_theme) = '';
                """
            )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_definition: str,
    ) -> None:
        columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name in columns:
            return
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        )
