from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2

from src.core.config import AppSettings
from src.core.models import EditPlan, OutputAspectRatio, SpeakerFocusPoint, SpeakerFocusTrack
from src.modules.clip_generator.service import ClipGenerator


class SpeakerTracker:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade_path))

    def track(
        self,
        video_path: Path,
        plans: list[EditPlan],
        output_aspect_ratio: OutputAspectRatio | str = OutputAspectRatio.VERTICAL_9_16.value,
        progress_callback: Callable[[str, int | None, int | None], None] | None = None,
    ) -> dict[int, SpeakerFocusTrack]:
        if not self.settings.speaker_tracking_enabled or not plans:
            return {}

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return {}

        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if source_width <= 0 or source_height <= 0:
            capture.release()
            return {}

        if not self._needs_tracking(source_width, source_height, output_aspect_ratio):
            capture.release()
            return {}

        tracks: dict[int, SpeakerFocusTrack] = {}
        try:
            total_plans = len(plans)
            for sequence_number, plan in enumerate(plans, start=1):
                if progress_callback is not None:
                    progress_callback(
                        f"Analyzing speaker framing for clip {sequence_number}/{total_plans}.",
                        sequence_number,
                        total_plans,
                    )
                points = self._track_plan(capture, plan, source_width, source_height)
                if points:
                    tracks[sequence_number] = SpeakerFocusTrack(
                        source_width=source_width,
                        source_height=source_height,
                        points=points,
                    )
                    if progress_callback is not None:
                        progress_callback(
                            f"Tracked {len(points)} focus sample(s) for clip {sequence_number}/{total_plans}.",
                            sequence_number,
                            total_plans,
                        )
        finally:
            capture.release()

        return tracks

    def _track_plan(
        self,
        capture: cv2.VideoCapture,
        plan: EditPlan,
        source_width: int,
        source_height: int,
    ) -> list[SpeakerFocusPoint]:
        duration = max(plan.duration_seconds, 0.1)
        interval = max(self.settings.speaker_tracking_sample_interval_seconds, 0.25)
        sample_times = [0.0]
        current = interval
        while current < duration:
            sample_times.append(round(current, 3))
            current += interval
        if sample_times[-1] < duration:
            sample_times.append(round(duration, 3))

        previous_center: tuple[float, float] | None = None
        points: list[SpeakerFocusPoint] = []
        for relative_time in sample_times:
            capture.set(cv2.CAP_PROP_POS_MSEC, (plan.start_seconds + relative_time) * 1000)
            success, frame = capture.read()
            if not success:
                continue

            center = self._detect_primary_face(frame, previous_center)
            if center is None:
                center = previous_center or (0.5, 0.42)
            previous_center = center
            points.append(
                SpeakerFocusPoint(
                    time_seconds=relative_time,
                    center_x=center[0],
                    center_y=center[1],
                )
            )

        if not points:
            return [
                SpeakerFocusPoint(time_seconds=0.0, center_x=0.5, center_y=0.5),
                SpeakerFocusPoint(time_seconds=duration, center_x=0.5, center_y=0.5),
            ]

        smoothed = self._smooth_points(points)
        if smoothed[-1].time_seconds < duration:
            smoothed.append(
                SpeakerFocusPoint(
                    time_seconds=duration,
                    center_x=smoothed[-1].center_x,
                    center_y=smoothed[-1].center_y,
                )
            )
        return smoothed

    def _detect_primary_face(
        self,
        frame: cv2.typing.MatLike,
        previous_center: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if self.detector.empty():
            return None

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_size = max(min(grayscale.shape[:2]) // 9, 56)
        faces = self.detector.detectMultiScale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_size, min_size),
        )
        if len(faces) == 0:
            return None

        frame_height, frame_width = grayscale.shape[:2]
        best_score = float("-inf")
        best_center: tuple[float, float] | None = None
        for x, y, width, height in faces:
            center_x = (x + (width / 2)) / frame_width
            center_y = (y + (height / 2)) / frame_height
            area_score = float(width * height)
            center_penalty = abs(center_x - 0.5) * frame_width * 2.5
            continuity_penalty = 0.0
            if previous_center is not None:
                continuity_penalty = (
                    abs(center_x - previous_center[0]) * frame_width * 2.0
                    + abs(center_y - previous_center[1]) * frame_height * 1.4
                )
            score = area_score - center_penalty - continuity_penalty
            if score > best_score:
                best_score = score
                best_center = (center_x, center_y)
        return best_center

    def _smooth_points(self, points: list[SpeakerFocusPoint]) -> list[SpeakerFocusPoint]:
        if not points:
            return []

        carry = max(0.0, min(self.settings.speaker_tracking_smoothing, 0.95))
        smoothed = [points[0]]
        previous_x = points[0].center_x
        previous_y = points[0].center_y
        for point in points[1:]:
            previous_x = (carry * previous_x) + ((1.0 - carry) * point.center_x)
            previous_y = (carry * previous_y) + ((1.0 - carry) * point.center_y)
            smoothed.append(
                SpeakerFocusPoint(
                    time_seconds=point.time_seconds,
                    center_x=previous_x,
                    center_y=previous_y,
                )
            )
        return smoothed

    @staticmethod
    def _needs_tracking(source_width: int, source_height: int, output_aspect_ratio: OutputAspectRatio | str) -> bool:
        target_width, target_height = ClipGenerator.render_dimensions(output_aspect_ratio)
        if source_width <= 0 or source_height <= 0:
            return False
        source_ratio = source_width / source_height
        target_ratio = target_width / target_height
        return abs(source_ratio - target_ratio) > 0.01
