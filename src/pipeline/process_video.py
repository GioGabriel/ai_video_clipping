from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.config import AppSettings
from src.core.models import EditPlan, JobStatus, PipelineStep, TranscriptArtifact, ViralMoment
from src.core.repositories import ClipRepository, JobEventRepository, JobRepository, utcnow_iso

if TYPE_CHECKING:
    from src.modules.audio_extractor.service import AudioExtractor
    from src.modules.clip_generator.service import ClipGenerator
    from src.modules.clip_optimizer.service import ClipOptimizer
    from src.modules.downloader.service import VideoDownloader
    from src.modules.edit_planner.service import EditPlanner
    from src.modules.export_manager.service import ExportManager
    from src.modules.hook_detector.service import HookDetector
    from src.modules.moment_refiner.service import MomentRefiner
    from src.modules.overlay_compositor.service import OverlayCompositor
    from src.modules.speaker_tracker.service import SpeakerTracker
    from src.modules.subtitle_generator.service import SubtitleGenerator
    from src.modules.thought_completion.service import ThoughtCompletion
    from src.modules.transcript_segmenter.service import TranscriptSegmenter
    from src.modules.transcription.service import WhisperTranscriber
    from src.modules.viral_detector.service import ViralMomentDetector
    from src.modules.viral_phrase_classifier.service import ViralPhraseClassifier

logger = logging.getLogger(__name__)


class VideoProcessingPipeline:
    def __init__(
        self,
        settings: AppSettings,
        job_repository: JobRepository,
        job_event_repository: JobEventRepository,
        clip_repository: ClipRepository,
        downloader: VideoDownloader,
        audio_extractor: AudioExtractor,
        transcriber: WhisperTranscriber,
        transcript_segmenter: TranscriptSegmenter,
        viral_phrase_classifier: ViralPhraseClassifier,
        viral_detector: ViralMomentDetector,
        hook_detector: HookDetector,
        thought_completion: ThoughtCompletion,
        clip_optimizer: ClipOptimizer,
        moment_refiner: MomentRefiner,
        edit_planner: EditPlanner,
        speaker_tracker: SpeakerTracker,
        clip_generator: ClipGenerator,
        subtitle_generator: SubtitleGenerator,
        overlay_compositor: OverlayCompositor,
        export_manager: ExportManager,
    ) -> None:
        self.settings = settings
        self.job_repository = job_repository
        self.job_event_repository = job_event_repository
        self.clip_repository = clip_repository
        self.downloader = downloader
        self.audio_extractor = audio_extractor
        self.transcriber = transcriber
        self.transcript_segmenter = transcript_segmenter
        self.viral_phrase_classifier = viral_phrase_classifier
        self.viral_detector = viral_detector
        self.hook_detector = hook_detector
        self.thought_completion = thought_completion
        self.clip_optimizer = clip_optimizer
        self.moment_refiner = moment_refiner
        self.edit_planner = edit_planner
        self.speaker_tracker = speaker_tracker
        self.clip_generator = clip_generator
        self.subtitle_generator = subtitle_generator
        self.overlay_compositor = overlay_compositor
        self.export_manager = export_manager

    def run(self, job_id: str) -> None:
        job = self.job_repository.get(job_id)
        if not job:
            raise RuntimeError(f"Job {job_id} does not exist.")

        source_url = job["source_url"]
        video_id = job["video_id"]
        output_aspect_ratio = str(job.get("output_aspect_ratio") or "9:16")
        caption_theme = str(job.get("caption_theme") or self.settings.default_caption_theme)
        ollama_model = str(job.get("ollama_model") or self.settings.ollama_model)
        transcript: TranscriptArtifact | None = None
        moments: list[ViralMoment] = []
        edit_plans: list[EditPlan] = []
        sentence_blocks = []

        try:
            self.job_repository.update(
                job_id,
                status=JobStatus.RUNNING.value,
                error_message=None,
                completed_at=None,
                active_task="Worker picked up the job.",
                step_progress_current=0,
                step_progress_total=0,
            )
            self._set_step(job_id, PipelineStep.DOWNLOAD, "Started local worker execution.")
            video_path = self.downloader.download(
                source_url,
                video_id,
                progress_callback=self._progress_callback(job_id, PipelineStep.DOWNLOAD),
            )
            self.job_repository.update(
                job_id,
                video_path=str(video_path),
                active_task=f"Source video ready at {video_path.name}.",
                step_progress_current=1,
                step_progress_total=1,
            )
            self._record_event(job_id, PipelineStep.DOWNLOAD, "success", f"Source video ready at {video_path}.")

            self._set_step(job_id, PipelineStep.EXTRACT_AUDIO, "Extracting mono 16 kHz WAV audio with FFmpeg.")
            audio_path = self.audio_extractor.extract(
                video_path,
                video_id,
                progress_callback=self._progress_callback(job_id, PipelineStep.EXTRACT_AUDIO),
            )
            self.job_repository.update(
                job_id,
                audio_path=str(audio_path),
                active_task=f"Audio extracted to {audio_path.name}.",
                step_progress_current=1,
                step_progress_total=1,
            )
            self._record_event(job_id, PipelineStep.EXTRACT_AUDIO, "success", f"Audio extracted to {audio_path}.")

            self._set_step(job_id, PipelineStep.TRANSCRIBE, "Transcribing audio locally with Whisper.")
            transcript = self.transcriber.transcribe(
                audio_path,
                video_id,
                progress_callback=self._progress_callback(job_id, PipelineStep.TRANSCRIBE),
            )
            self.job_repository.update(
                job_id,
                transcript_path=str(transcript.transcript_path),
                active_task=f"Transcript saved to {transcript.transcript_path.name}.",
                step_progress_current=len(transcript.segments),
                step_progress_total=len(transcript.segments),
            )
            self._record_event(
                job_id,
                PipelineStep.TRANSCRIBE,
                "success",
                f"Transcript saved to {transcript.transcript_path} with {len(transcript.segments)} timestamped segments.",
            )

            self._set_step(job_id, PipelineStep.SEGMENT_TRANSCRIPT, "Segmenting transcript into sentence-aware blocks.")
            sentence_blocks = self.transcript_segmenter.segment(transcript)
            self.job_repository.update(
                job_id,
                active_task=f"Built {len(sentence_blocks)} sentence block(s) for scoring.",
                step_progress_current=len(sentence_blocks),
                step_progress_total=len(sentence_blocks),
            )
            self._record_event(
                job_id,
                PipelineStep.SEGMENT_TRANSCRIPT,
                "success",
                f"Segmented transcript into {len(sentence_blocks)} sentence-aware block(s).",
            )

            self._set_step(job_id, PipelineStep.CLASSIFY_VIRAL_PHRASES, "Classifying sentence blocks for viral phrase triggers.")
            sentence_blocks = self.viral_phrase_classifier.classify(sentence_blocks)
            transcript.sentence_blocks = sentence_blocks
            triggered_blocks = sum(1 for block in sentence_blocks if block.phrase_score > 0)
            self.job_repository.update(
                job_id,
                active_task=f"Detected viral phrase triggers in {triggered_blocks} of {len(sentence_blocks)} sentence block(s).",
                step_progress_current=triggered_blocks,
                step_progress_total=len(sentence_blocks),
            )
            self._record_event(
                job_id,
                PipelineStep.CLASSIFY_VIRAL_PHRASES,
                "success",
                f"Viral phrase classifier tagged {triggered_blocks} sentence block(s) with curiosity, advice, contrarian, list, or emotional triggers.",
            )

            self._set_step(
                job_id,
                PipelineStep.DETECT_MOMENTS,
                f"Analyzing transcript chunks with Ollama model '{ollama_model}'.",
            )
            moments = self.viral_detector.detect(
                sentence_blocks,
                model_name=ollama_model,
                progress_callback=self._progress_callback(job_id, PipelineStep.DETECT_MOMENTS),
            )
            if not moments:
                raise RuntimeError("No viral moments were detected for this video.")
            self.job_repository.update(
                job_id,
                active_task=f"Detected {len(moments)} scored viral candidate clip(s).",
                step_progress_current=len(moments),
                step_progress_total=len(moments),
            )
            self._record_event(
                job_id,
                PipelineStep.DETECT_MOMENTS,
                "success",
                (
                    f"Detected {len(moments)} scored viral clip candidate(s) above "
                    f"{self.settings.viral_min_score:.0f} using model '{ollama_model}'."
                ),
            )

            self._set_step(job_id, PipelineStep.DETECT_HOOKS, "Finding the strongest hook before each candidate moment.")
            moments = self.hook_detector.detect(sentence_blocks, moments)
            self.job_repository.update(
                job_id,
                active_task=f"Aligned {len(moments)} clip candidate(s) to stronger hook starts.",
                step_progress_current=len(moments),
                step_progress_total=len(moments),
            )
            self._record_event(
                job_id,
                PipelineStep.DETECT_HOOKS,
                "success",
                f"Hook detector rewound {len(moments)} candidate clip(s) to stronger opening lines.",
            )

            self._set_step(job_id, PipelineStep.COMPLETE_THOUGHTS, "Extending candidates so each thought lands cleanly.")
            moments = self.moment_refiner.refine(transcript.segments, moments)
            self._record_event(
                job_id,
                PipelineStep.COMPLETE_THOUGHTS,
                "info",
                f"Moment refiner aligned {len(moments)} candidate clip(s) to stronger transcript thought blocks.",
            )
            moments = self.thought_completion.complete(sentence_blocks, moments)
            self.job_repository.update(
                job_id,
                active_task=f"Thought completion updated {len(moments)} clip candidate(s).",
                step_progress_current=len(moments),
                step_progress_total=len(moments),
            )
            self._record_event(
                job_id,
                PipelineStep.COMPLETE_THOUGHTS,
                "success",
                f"Thought completion ensured {len(moments)} clip candidate(s) end on sentence boundaries.",
            )

            self._set_step(job_id, PipelineStep.OPTIMIZE_CLIPS, "Optimizing clip lengths for 18-50 second short-form delivery.")
            moments = self.clip_optimizer.optimize(sentence_blocks, moments)
            self.job_repository.update(
                job_id,
                active_task=f"Clip optimizer produced {len(moments)} upload-ready candidate clip(s).",
                step_progress_current=len(moments),
                step_progress_total=len(moments),
            )
            self._record_event(
                job_id,
                PipelineStep.OPTIMIZE_CLIPS,
                "success",
                f"Clip optimizer produced {len(moments)} sentence-safe candidate clip(s) targeted for 25-35 seconds.",
            )

            self._set_step(job_id, PipelineStep.GENERATE_CLIPS, "Planning clip edits and render tasks.")
            edit_plans = self.edit_planner.plan(transcript, moments)
            self.job_repository.update(
                job_id,
                active_task=f"Built {len(edit_plans)} edit plan(s).",
                step_progress_current=0,
                step_progress_total=len(edit_plans),
            )
            self._record_event(
                job_id,
                PipelineStep.GENERATE_CLIPS,
                "info",
                f"Built {len(edit_plans)} edit plan(s) with hook overlays and caption timing beats.",
            )
            self._emit_runtime(
                job_id,
                PipelineStep.GENERATE_CLIPS,
                "Rendering platform-formatted clips with FFmpeg.",
                current=0,
                total=len(edit_plans),
                record_event=True,
            )
            focus_tracks = self.speaker_tracker.track(
                video_path,
                edit_plans,
                output_aspect_ratio,
                progress_callback=self._progress_callback(job_id, PipelineStep.GENERATE_CLIPS),
            )
            if focus_tracks:
                self._record_event(
                    job_id,
                    PipelineStep.GENERATE_CLIPS,
                    "info",
                    f"Tracked face-based speaker framing for {len(focus_tracks)} clip candidate(s).",
                )
            clips = self.clip_generator.generate(
                job_id,
                video_id,
                video_path,
                edit_plans,
                output_aspect_ratio,
                focus_tracks=focus_tracks,
                progress_callback=self._progress_callback(job_id, PipelineStep.GENERATE_CLIPS),
            )
            self.job_repository.update(
                job_id,
                active_task=f"Generated {len(clips)} clip export(s).",
                step_progress_current=len(clips),
                step_progress_total=len(edit_plans),
            )
            self._record_event(
                job_id,
                PipelineStep.GENERATE_CLIPS,
                "success",
                f"Generated {len(clips)} clip export(s) at {output_aspect_ratio}.",
            )

            if self.settings.subtitle_generation_enabled:
                self._set_step(job_id, PipelineStep.GENERATE_SUBTITLES, "Writing subtitle sidecars for rendered clips.")
                subtitles = self.subtitle_generator.generate(
                    video_id,
                    transcript,
                    clips,
                    {index: plan for index, plan in enumerate(edit_plans, start=1)},
                    output_aspect_ratio=output_aspect_ratio,
                    caption_theme=caption_theme,
                    progress_callback=self._progress_callback(job_id, PipelineStep.GENERATE_SUBTITLES),
                )
                for clip in clips:
                    subtitle_artifact = subtitles.get(clip.sequence_number)
                    if subtitle_artifact is None:
                        continue
                    clip.subtitle_path = subtitle_artifact.sidecar_path
                    clip.styled_subtitle_path = subtitle_artifact.styled_path

                burned_clips = 0
                if self.settings.subtitle_burn_in_enabled:
                    burn_in_result = self.overlay_compositor.apply(
                        video_id=video_id,
                        transcript=transcript,
                        clips=clips,
                        edit_plans={index: plan for index, plan in enumerate(edit_plans, start=1)},
                        output_aspect_ratio=output_aspect_ratio,
                        caption_theme=caption_theme,
                        progress_callback=self._progress_callback(job_id, PipelineStep.GENERATE_SUBTITLES),
                    )
                    burned_clips = burn_in_result.burned_count
                    if burn_in_result.warning_message:
                        self._record_event(
                            job_id,
                            PipelineStep.GENERATE_SUBTITLES,
                            "warning",
                            burn_in_result.warning_message,
                        )
                self.job_repository.update(
                    job_id,
                    active_task=(
                        f"Generated {len(subtitles)} subtitle file(s) and rendered overlays into "
                        f"{burned_clips} clip(s)."
                    ),
                    step_progress_current=len(clips),
                    step_progress_total=len(clips),
                )
                self._record_event(
                    job_id,
                    PipelineStep.GENERATE_SUBTITLES,
                    "success",
                    f"Generated {len(subtitles)} subtitle file(s) and rendered karaoke overlays into {burned_clips} clip(s).",
                )

            self.clip_repository.replace_for_job(job_id, video_id, clips)

            self._set_step(job_id, PipelineStep.EXPORT, "Writing manifest and final output metadata.")
            manifest_path = self.export_manager.build_manifest(
                job_id=job_id,
                video_id=video_id,
                source_url=source_url,
                output_aspect_ratio=output_aspect_ratio,
                caption_theme=caption_theme,
                transcript=transcript,
                moments=moments,
                clips=clips,
                edit_plans=edit_plans,
            )
            self.job_repository.update(
                job_id,
                active_task=f"Manifest written to {manifest_path.name}.",
                step_progress_current=1,
                step_progress_total=1,
            )
            self._record_event(job_id, PipelineStep.EXPORT, "success", f"Manifest written to {manifest_path}.")

            self.job_repository.update(
                job_id,
                status=JobStatus.COMPLETED.value,
                current_step=PipelineStep.COMPLETED.value,
                manifest_path=str(manifest_path),
                completed_at=utcnow_iso(),
                error_message=None,
                active_task=f"Job completed with {len(clips)} clip(s).",
                step_progress_current=len(clips),
                step_progress_total=len(clips),
            )
            self._record_event(job_id, PipelineStep.COMPLETED, "success", f"Job completed with {len(clips)} clip(s).")
            logger.info("Completed job %s with %s generated clips.", job_id, len(clips))
        except Exception as exc:
            logger.exception("Pipeline failed for job %s", job_id)
            latest_job = self.job_repository.get(job_id) or {}
            failed_step = latest_job.get("current_step", PipelineStep.FAILED.value)
            self.job_repository.update(
                job_id,
                status=JobStatus.FAILED.value,
                error_message=str(exc),
                completed_at=utcnow_iso(),
                active_task=str(exc),
            )
            self.job_event_repository.create(
                job_id=job_id,
                step=failed_step,
                level="error",
                message=str(exc),
            )
            raise

    def _set_step(self, job_id: str, step: PipelineStep, message: str) -> None:
        self._emit_runtime(
            job_id,
            step,
            message,
            current=0,
            total=0,
            record_event=True,
        )

    def _record_event(self, job_id: str, step: PipelineStep, level: str, message: str) -> None:
        self.job_event_repository.create(
            job_id=job_id,
            step=step.value,
            level=level,
            message=message,
        )

    def _progress_callback(self, job_id: str, step: PipelineStep):
        def emit(message: str, current: int | None = None, total: int | None = None) -> None:
            self._emit_runtime(
                job_id,
                step,
                message,
                current=current,
                total=total,
                record_event=True,
            )

        return emit

    def _emit_runtime(
        self,
        job_id: str,
        step: PipelineStep,
        message: str,
        *,
        current: int | None = None,
        total: int | None = None,
        record_event: bool,
    ) -> None:
        update_fields: dict[str, object] = {
            "current_step": step.value,
            "active_task": message,
        }
        if current is not None:
            update_fields["step_progress_current"] = current
        if total is not None:
            update_fields["step_progress_total"] = total
        self.job_repository.update(job_id, **update_fields)
        if record_event:
            self.job_event_repository.create(
                job_id=job_id,
                step=step.value,
                level="info",
                message=message,
            )
