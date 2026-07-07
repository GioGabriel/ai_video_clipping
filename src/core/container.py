from __future__ import annotations

from dataclasses import dataclass

from src.core.command_runner import CommandRunner
from src.core.config import AppSettings, load_settings
from src.core.database import Database
from src.core.job_cleanup import JobCleanupService
from src.core.logging import configure_logging
from src.core.queue import ProcessingJobQueue
from src.core.repositories import ClipRepository, JobEventRepository, JobRepository
from src.modules.audio_extractor.service import AudioExtractor
from src.modules.clip_generator.service import ClipGenerator
from src.modules.clip_optimizer.service import ClipOptimizer
from src.modules.downloader.service import VideoDownloader
from src.modules.edit_planner.service import EditPlanner
from src.modules.export_manager.service import ExportManager
from src.modules.hook_detector.service import HookDetector
from src.modules.moment_refiner.service import MomentRefiner
from src.modules.ollama_runtime.service import OllamaRuntimeService
from src.modules.overlay_compositor.service import OverlayCompositor
from src.modules.speaker_tracker.service import SpeakerTracker
from src.modules.subtitle_generator.service import SubtitleGenerator
from src.modules.transcription.service import WhisperTranscriber
from src.modules.thought_completion.service import ThoughtCompletion
from src.modules.transcript_segmenter.service import TranscriptSegmenter
from src.modules.viral_detector.service import ViralMomentDetector
from src.modules.viral_phrase_classifier.service import ViralPhraseClassifier
from src.pipeline.process_video import VideoProcessingPipeline


@dataclass(slots=True)
class AppContainer:
    settings: AppSettings
    database: Database
    job_repository: JobRepository
    job_event_repository: JobEventRepository
    clip_repository: ClipRepository
    job_cleanup: JobCleanupService
    downloader: VideoDownloader
    audio_extractor: AudioExtractor
    transcriber: WhisperTranscriber
    transcript_segmenter: TranscriptSegmenter
    viral_phrase_classifier: ViralPhraseClassifier
    ollama_runtime: OllamaRuntimeService
    viral_detector: ViralMomentDetector
    hook_detector: HookDetector
    thought_completion: ThoughtCompletion
    clip_optimizer: ClipOptimizer
    moment_refiner: MomentRefiner
    edit_planner: EditPlanner
    speaker_tracker: SpeakerTracker
    clip_generator: ClipGenerator
    subtitle_generator: SubtitleGenerator
    overlay_compositor: OverlayCompositor
    export_manager: ExportManager
    pipeline: VideoProcessingPipeline
    job_queue: ProcessingJobQueue


def build_container() -> AppContainer:
    return build_container_with_settings(load_settings())


def build_container_with_settings(settings: AppSettings) -> AppContainer:
    configure_logging(settings.log_level)

    database = Database(settings.database_path)
    job_repository = JobRepository(database)
    job_event_repository = JobEventRepository(database)
    clip_repository = ClipRepository(database)
    job_cleanup = JobCleanupService(settings, job_repository, job_event_repository, clip_repository)
    command_runner = CommandRunner()

    downloader = VideoDownloader(settings, command_runner)
    audio_extractor = AudioExtractor(settings, command_runner)
    transcriber = WhisperTranscriber(settings)
    transcript_segmenter = TranscriptSegmenter(settings)
    viral_phrase_classifier = ViralPhraseClassifier()
    ollama_runtime = OllamaRuntimeService(settings)
    viral_detector = ViralMomentDetector(settings)
    hook_detector = HookDetector(settings)
    thought_completion = ThoughtCompletion(settings)
    clip_optimizer = ClipOptimizer(settings)
    moment_refiner = MomentRefiner(settings)
    edit_planner = EditPlanner(settings)
    speaker_tracker = SpeakerTracker(settings)
    clip_generator = ClipGenerator(settings, command_runner)
    subtitle_generator = SubtitleGenerator(settings)
    overlay_compositor = OverlayCompositor(settings, command_runner, subtitle_generator)
    export_manager = ExportManager(settings)

    pipeline = VideoProcessingPipeline(
        settings=settings,
        job_repository=job_repository,
        job_event_repository=job_event_repository,
        clip_repository=clip_repository,
        downloader=downloader,
        audio_extractor=audio_extractor,
        transcriber=transcriber,
        transcript_segmenter=transcript_segmenter,
        viral_phrase_classifier=viral_phrase_classifier,
        viral_detector=viral_detector,
        hook_detector=hook_detector,
        thought_completion=thought_completion,
        clip_optimizer=clip_optimizer,
        moment_refiner=moment_refiner,
        edit_planner=edit_planner,
        speaker_tracker=speaker_tracker,
        clip_generator=clip_generator,
        subtitle_generator=subtitle_generator,
        overlay_compositor=overlay_compositor,
        export_manager=export_manager,
    )
    job_queue = ProcessingJobQueue(settings.job_worker_count, pipeline.run)

    return AppContainer(
        settings=settings,
        database=database,
        job_repository=job_repository,
        job_event_repository=job_event_repository,
        clip_repository=clip_repository,
        job_cleanup=job_cleanup,
        downloader=downloader,
        audio_extractor=audio_extractor,
        transcriber=transcriber,
        transcript_segmenter=transcript_segmenter,
        viral_phrase_classifier=viral_phrase_classifier,
        ollama_runtime=ollama_runtime,
        viral_detector=viral_detector,
        hook_detector=hook_detector,
        thought_completion=thought_completion,
        clip_optimizer=clip_optimizer,
        moment_refiner=moment_refiner,
        edit_planner=edit_planner,
        speaker_tracker=speaker_tracker,
        clip_generator=clip_generator,
        subtitle_generator=subtitle_generator,
        overlay_compositor=overlay_compositor,
        export_manager=export_manager,
        pipeline=pipeline,
        job_queue=job_queue,
    )
