from pathlib import Path

import pytest

from src.core.config import AppSettings
from src.core.database import Database
from src.core.models import (
    ClipArtifact,
    EditPlan,
    HookOverlay,
    SentenceBlock,
    SubtitleArtifact,
    SubtitleBurnInResult,
    TranscriptArtifact,
    TranscriptSegment,
    TranscriptWord,
    ViralMoment,
    ZoomEffect,
)
from src.core.repositories import ClipRepository, JobEventRepository, JobRepository
from src.modules.edit_planner.service import EditPlanner
from src.modules.moment_refiner.service import MomentRefiner
from src.pipeline.process_video import VideoProcessingPipeline


def build_settings(tmp_path: Path) -> AppSettings:
    data_dir = tmp_path / "data"
    return AppSettings(
        project_root=tmp_path,
        app_name="AI Clipping System",
        app_host="127.0.0.1",
        app_port=8000,
        log_level="INFO",
        data_dir=data_dir,
        videos_dir=data_dir / "videos",
        audio_dir=data_dir / "audio",
        transcripts_dir=data_dir / "transcripts",
        clips_dir=data_dir / "clips",
        database_path=data_dir / "app.db",
        yt_dlp_binary="yt-dlp",
        ffmpeg_binary="ffmpeg",
        yt_dlp_skip_certificate_check=False,
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="llama3",
        ollama_request_timeout_seconds=180,
        whisper_model="base",
        whisper_device="cpu",
        job_worker_count=1,
        max_clip_candidates=10,
        max_candidates_per_chunk=3,
        min_clip_duration_seconds=20,
        max_clip_duration_seconds=60,
        target_clip_duration_seconds=30,
        viral_chunk_duration_seconds=420,
        viral_chunk_overlap_seconds=60,
        viral_min_score=60,
        subtitle_generation_enabled=True,
    )


class FailingDownloader:
    def download(self, url: str, video_id: str, progress_callback=None) -> Path:  # noqa: ANN001
        raise RuntimeError("download failed")


class UnusedModule:
    def __getattr__(self, name: str):
        raise AssertionError(f"Unexpected access to {name}")


class SuccessfulDownloader:
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path

    def download(self, url: str, video_id: str, progress_callback=None) -> Path:  # noqa: ANN001
        if progress_callback is not None:
            progress_callback("Downloader test progress.", 1, 1)
        return self.video_path


class SuccessfulAudioExtractor:
    def __init__(self, audio_path: Path) -> None:
        self.audio_path = audio_path

    def extract(self, video_path: Path, video_id: str, progress_callback=None) -> Path:  # noqa: ANN001
        return self.audio_path


class SuccessfulTranscriber:
    def __init__(self, transcript: TranscriptArtifact) -> None:
        self.transcript = transcript

    def transcribe(self, audio_path: Path, video_id: str, progress_callback=None) -> TranscriptArtifact:  # noqa: ANN001
        return self.transcript


class SuccessfulTranscriptSegmenter:
    def __init__(self, blocks: list[SentenceBlock]) -> None:
        self.blocks = blocks

    def segment(self, transcript: TranscriptArtifact) -> list[SentenceBlock]:
        transcript.sentence_blocks = self.blocks
        return self.blocks


class SuccessfulViralPhraseClassifier:
    def classify(self, sentence_blocks: list[SentenceBlock]) -> list[SentenceBlock]:
        return sentence_blocks


class SuccessfulViralDetector:
    def __init__(self, moments: list[ViralMoment]) -> None:
        self.moments = moments
        self.received_model_name: str | None = None

    def detect(
        self,
        segments: list[TranscriptSegment],
        model_name: str | None = None,
        progress_callback=None,
    ) -> list[ViralMoment]:  # noqa: ANN001
        self.received_model_name = model_name
        return self.moments


class SuccessfulHookDetector:
    def detect(self, sentence_blocks: list[SentenceBlock], moments: list[ViralMoment]) -> list[ViralMoment]:
        return moments


class SuccessfulThoughtCompletion:
    def complete(self, sentence_blocks: list[SentenceBlock], moments: list[ViralMoment]) -> list[ViralMoment]:
        return moments


class SuccessfulClipOptimizerModule:
    def optimize(self, sentence_blocks: list[SentenceBlock], moments: list[ViralMoment]) -> list[ViralMoment]:
        return moments


class PassthroughMomentRefiner:
    def refine(self, segments: list[TranscriptSegment], moments: list[ViralMoment]) -> list[ViralMoment]:
        return moments


class SuccessfulEditPlanner:
    def __init__(self, plans: list[EditPlan]) -> None:
        self.plans = plans

    def plan(self, transcript: TranscriptArtifact, moments: list[ViralMoment]) -> list[EditPlan]:
        return self.plans


class SuccessfulClipGenerator:
    def __init__(self, clips: list[ClipArtifact], burn_in_result: SubtitleBurnInResult) -> None:
        self.clips = clips
        self.burn_in_result = burn_in_result

    def generate(
        self,
        job_id: str,
        video_id: str,
        video_path: Path,
        plans: list[EditPlan],
        output_aspect_ratio: str = "9:16",
        focus_tracks: dict[int, object] | None = None,
        progress_callback=None,  # noqa: ANN001
    ) -> list[ClipArtifact]:
        return self.clips

    def burn_subtitles(self, clips: list[ClipArtifact], subtitle_paths: dict[int, Path]) -> SubtitleBurnInResult:
        return self.burn_in_result


class SuccessfulSubtitleGenerator:
    def __init__(self, artifacts: dict[int, SubtitleArtifact]) -> None:
        self.artifacts = artifacts

    def generate(
        self,
        video_id: str,
        transcript: TranscriptArtifact,
        clips: list[ClipArtifact],
        edit_plans: dict[int, EditPlan] | None = None,
        output_aspect_ratio: str = "9:16",
        caption_theme: str = "tiktok",
        progress_callback=None,  # noqa: ANN001
    ) -> dict[int, SubtitleArtifact]:
        return self.artifacts


class SuccessfulSpeakerTracker:
    def __init__(self, tracks: dict[int, object] | None = None) -> None:
        self.tracks = tracks or {}

    def track(
        self,
        video_path: Path,
        plans: list[EditPlan],
        output_aspect_ratio: str = "9:16",
        progress_callback=None,  # noqa: ANN001
    ) -> dict[int, object]:
        return self.tracks


class SuccessfulOverlayCompositor:
    def __init__(self, result: SubtitleBurnInResult) -> None:
        self.result = result

    def apply(
        self,
        video_id: str,
        transcript: TranscriptArtifact,
        clips: list[ClipArtifact],
        edit_plans: dict[int, EditPlan] | None = None,
        output_aspect_ratio: str = "9:16",
        caption_theme: str = "tiktok",
        progress_callback=None,  # noqa: ANN001
    ) -> SubtitleBurnInResult:
        return self.result


class SuccessfulExportManager:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path

    def build_manifest(self, **kwargs: object) -> Path:
        self.manifest_path.write_text("{}", encoding="utf-8")
        return self.manifest_path


def test_pipeline_preserves_failing_step_on_job_failure(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.ensure_directories()

    database = Database(settings.database_path)
    database.initialize()
    job_repository = JobRepository(database)
    job_event_repository = JobEventRepository(database)
    clip_repository = ClipRepository(database)

    job_id = "job-1"
    video_id = "video-1"
    job_repository.create(job_id=job_id, video_id=video_id, source_url="https://example.com/video")

    pipeline = VideoProcessingPipeline(
        settings=settings,
        job_repository=job_repository,
        job_event_repository=job_event_repository,
        clip_repository=clip_repository,
        downloader=FailingDownloader(),
        audio_extractor=UnusedModule(),
        transcriber=UnusedModule(),
        transcript_segmenter=UnusedModule(),
        viral_phrase_classifier=UnusedModule(),
        viral_detector=UnusedModule(),
        hook_detector=UnusedModule(),
        thought_completion=UnusedModule(),
        clip_optimizer=UnusedModule(),
        moment_refiner=MomentRefiner(settings),
        edit_planner=EditPlanner(settings),
        speaker_tracker=SuccessfulSpeakerTracker(),
        clip_generator=UnusedModule(),
        subtitle_generator=UnusedModule(),
        overlay_compositor=UnusedModule(),
        export_manager=UnusedModule(),
    )

    with pytest.raises(RuntimeError, match="download failed"):
        pipeline.run(job_id)

    job = job_repository.get(job_id)
    assert job is not None
    assert job["status"] == "failed"
    assert job["current_step"] == "download_video"
    assert job["error_message"] == "download failed"

    events = job_event_repository.list_by_job_id(job_id)
    assert any(event["step"] == "download_video" and event["level"] == "error" for event in events)


def test_pipeline_completes_when_subtitle_burn_in_is_unavailable(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.ensure_directories()

    database = Database(settings.database_path)
    database.initialize()
    job_repository = JobRepository(database)
    job_event_repository = JobEventRepository(database)
    clip_repository = ClipRepository(database)

    job_id = "job-2"
    video_id = "video-2"
    job_repository.create(job_id=job_id, video_id=video_id, source_url="https://example.com/video")
    job_repository.update(job_id, ollama_model="deepseek-r1:8b")

    video_path = settings.videos_dir / f"{video_id}.mp4"
    audio_path = settings.audio_dir / f"{video_id}.wav"
    transcript_path = settings.transcripts_dir / f"{video_id}.json"
    clip_path = settings.clips_dir / video_id / "clip_001.mp4"
    subtitle_sidecar = settings.clips_dir / video_id / "subtitles" / "clip_001.srt"
    subtitle_styled = settings.clips_dir / video_id / "subtitles" / "styled" / "clip_001.ass"
    manifest_path = settings.clips_dir / video_id / "manifest.json"

    for path in [video_path, audio_path, transcript_path, clip_path, subtitle_sidecar, subtitle_styled]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact", encoding="utf-8")

    transcript = TranscriptArtifact(
        video_id=video_id,
        language="en",
        text="Keep going.",
        transcript_path=transcript_path,
        segments=[
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=3.4,
                text="Keep going.",
                words=[
                    TranscriptWord(start_seconds=0.0, end_seconds=0.8, text="Keep"),
                    TranscriptWord(start_seconds=0.8, end_seconds=1.6, text="going."),
                ],
            )
        ],
    )
    moments = [
        ViralMoment(
            start_seconds=0.0,
            end_seconds=3.4,
            score=0.95,
            hook="KEEP GOING",
            reason="Motivational payoff",
        )
    ]
    plans = [
        EditPlan(
            start_seconds=0.0,
            end_seconds=4.0,
            score=0.95,
            hook="KEEP GOING",
            reason="Motivational payoff",
            zoom_effects=[ZoomEffect(start_seconds=0.2, end_seconds=0.9, peak_scale=1.1, anchor_text="KEEP")],
            hook_overlay=HookOverlay(text="KEEP GOING", start_seconds=0.0, end_seconds=1.2),
        )
    ]
    clips = [
        ClipArtifact(
            job_id=job_id,
            video_id=video_id,
            sequence_number=1,
            file_path=clip_path,
            start_seconds=0.0,
            end_seconds=4.0,
            hook="KEEP GOING",
            reason="Motivational payoff",
            score=0.95,
        )
    ]
    subtitle_artifacts = {
        1: SubtitleArtifact(
            sidecar_path=subtitle_sidecar,
            styled_path=subtitle_styled,
        )
    }
    sentence_blocks = [
        SentenceBlock(
            start_seconds=0.0,
            end_seconds=3.4,
            text="Keep going.",
            speaker="speaker_1",
            source_segment_start_index=0,
            source_segment_end_index=0,
        )
    ]
    viral_detector = SuccessfulViralDetector(moments)

    pipeline = VideoProcessingPipeline(
        settings=settings,
        job_repository=job_repository,
        job_event_repository=job_event_repository,
        clip_repository=clip_repository,
        downloader=SuccessfulDownloader(video_path),
        audio_extractor=SuccessfulAudioExtractor(audio_path),
        transcriber=SuccessfulTranscriber(transcript),
        transcript_segmenter=SuccessfulTranscriptSegmenter(sentence_blocks),
        viral_phrase_classifier=SuccessfulViralPhraseClassifier(),
        viral_detector=viral_detector,
        hook_detector=SuccessfulHookDetector(),
        thought_completion=SuccessfulThoughtCompletion(),
        clip_optimizer=SuccessfulClipOptimizerModule(),
        moment_refiner=PassthroughMomentRefiner(),
        edit_planner=SuccessfulEditPlanner(plans),
        speaker_tracker=SuccessfulSpeakerTracker(),
        clip_generator=SuccessfulClipGenerator(
            clips=clips,
            burn_in_result=SubtitleBurnInResult(
                burned_count=0,
                warning_message="FFmpeg subtitle burn-in was skipped because the installed FFmpeg binary does not include the subtitles filter.",
            ),
        ),
        subtitle_generator=SuccessfulSubtitleGenerator(subtitle_artifacts),
        overlay_compositor=SuccessfulOverlayCompositor(
            SubtitleBurnInResult(
                burned_count=1,
                warning_message=None,
            )
        ),
        export_manager=SuccessfulExportManager(manifest_path),
    )

    pipeline.run(job_id)

    job = job_repository.get(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert job["current_step"] == "completed"

    events = job_event_repository.list_by_job_id(job_id)
    assert viral_detector.received_model_name == "deepseek-r1:8b"
    assert any(
        event["step"] == "generate_subtitles"
        and "rendered karaoke overlays into 1 clip(s)" in event["message"]
        for event in events
    )
    assert any(event["step"] == "classify_viral_phrases" and event["level"] == "success" for event in events)
    assert any(event["step"] == "completed" and event["level"] == "success" for event in events)
