from __future__ import annotations

import re

from src.core.config import AppSettings
from src.core.models import SentenceBlock, ViralMoment

HOOK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwhy\b", re.IGNORECASE),
    re.compile(r"\bhow\b", re.IGNORECASE),
    re.compile(r"\bmost people\b", re.IGNORECASE),
    re.compile(r"\bnobody\b", re.IGNORECASE),
    re.compile(r"\beveryone\b.*\bno one\b", re.IGNORECASE),
    re.compile(r"\bif you\b.*\byou will\b", re.IGNORECASE),
)


class ClipOptimizer:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def optimize(self, sentence_blocks: list[SentenceBlock], moments: list[ViralMoment]) -> list[ViralMoment]:
        if not sentence_blocks:
            return moments

        optimized: list[ViralMoment] = []
        for moment in moments:
            optimized.extend(self._optimize_moment(sentence_blocks, moment))

        deduplicated: list[ViralMoment] = []
        for moment in optimized:
            if any(self._is_duplicate(moment, existing) for existing in deduplicated):
                continue
            deduplicated.append(moment)
        return deduplicated

    def _optimize_moment(self, sentence_blocks: list[SentenceBlock], moment: ViralMoment) -> list[ViralMoment]:
        start_index = self._find_block_index(sentence_blocks, moment.start_seconds)
        end_index = self._find_block_index(sentence_blocks, moment.end_seconds)
        if start_index is None or end_index is None:
            return [moment]

        start_index, end_index = self._expand_to_minimum(sentence_blocks, start_index, end_index)
        duration = sentence_blocks[end_index].end_seconds - sentence_blocks[start_index].start_seconds
        if duration <= self.settings.max_clip_duration_seconds:
            return [self._build_moment_from_range(sentence_blocks, moment, start_index, end_index)]
        return self._split_long_range(sentence_blocks, moment, start_index, end_index)

    def _expand_to_minimum(
        self,
        sentence_blocks: list[SentenceBlock],
        start_index: int,
        end_index: int,
    ) -> tuple[int, int]:
        def current_duration() -> float:
            return sentence_blocks[end_index].end_seconds - sentence_blocks[start_index].start_seconds

        while current_duration() < self.settings.min_clip_duration_seconds:
            can_expand_forward = end_index + 1 < len(sentence_blocks)
            can_expand_backward = start_index > 0
            if not can_expand_forward and not can_expand_backward:
                break

            forward_gain = (
                sentence_blocks[end_index + 1].duration_seconds if can_expand_forward else -1.0
            )
            backward_gain = (
                sentence_blocks[start_index - 1].duration_seconds if can_expand_backward else -1.0
            )

            if can_expand_forward and (forward_gain >= backward_gain or not can_expand_backward):
                end_index += 1
                continue
            if can_expand_backward:
                start_index -= 1

        while (
            current_duration() < self.settings.target_clip_duration_seconds
            and end_index + 1 < len(sentence_blocks)
            and sentence_blocks[end_index + 1].end_seconds - sentence_blocks[start_index].start_seconds
            <= self.settings.max_clip_duration_seconds
        ):
            gap = sentence_blocks[end_index + 1].start_seconds - sentence_blocks[end_index].end_seconds
            if gap > self.settings.clip_optimizer_split_tolerance_seconds:
                break
            end_index += 1

        return start_index, end_index

    def _split_long_range(
        self,
        sentence_blocks: list[SentenceBlock],
        moment: ViralMoment,
        start_index: int,
        end_index: int,
    ) -> list[ViralMoment]:
        chunks: list[tuple[int, int]] = []
        chunk_start = start_index
        while chunk_start <= end_index:
            chunk_end = chunk_start
            while chunk_end + 1 <= end_index:
                next_duration = sentence_blocks[chunk_end + 1].end_seconds - sentence_blocks[chunk_start].start_seconds
                if next_duration > self.settings.target_clip_duration_seconds and (
                    sentence_blocks[chunk_end].end_seconds - sentence_blocks[chunk_start].start_seconds
                    >= self.settings.min_clip_duration_seconds
                ):
                    break
                if next_duration > self.settings.max_clip_duration_seconds:
                    break
                chunk_end += 1

            while (
                sentence_blocks[chunk_end].end_seconds - sentence_blocks[chunk_start].start_seconds
                < self.settings.min_clip_duration_seconds
                and chunk_end < end_index
            ):
                chunk_end += 1

            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end + 1

        if len(chunks) >= 2:
            last_start, last_end = chunks[-1]
            last_duration = sentence_blocks[last_end].end_seconds - sentence_blocks[last_start].start_seconds
            if last_duration < self.settings.min_clip_duration_seconds:
                previous_start, previous_end = chunks[-2]
                if sentence_blocks[last_end].end_seconds - sentence_blocks[previous_start].start_seconds <= (
                    self.settings.max_clip_duration_seconds + self.settings.clip_optimizer_split_tolerance_seconds
                ):
                    chunks[-2] = (previous_start, last_end)
                    chunks.pop()

        return [
            self._build_moment_from_range(sentence_blocks, moment, chunk_start, chunk_end, chunk_index)
            for chunk_index, (chunk_start, chunk_end) in enumerate(chunks, start=1)
        ]

    def _build_moment_from_range(
        self,
        sentence_blocks: list[SentenceBlock],
        template: ViralMoment,
        start_index: int,
        end_index: int,
        chunk_index: int = 1,
    ) -> ViralMoment:
        start_seconds = sentence_blocks[start_index].start_seconds
        end_seconds = min(
            sentence_blocks[end_index].end_seconds + self.settings.clip_trailing_pad_seconds,
            start_seconds + self.settings.max_clip_duration_seconds,
        )
        strongest_block = self._strongest_hook_block(sentence_blocks[start_index : end_index + 1])
        hook_text = strongest_block.text if chunk_index > 1 or not template.hook else template.hook
        duration_score = self._duration_score(end_seconds - start_seconds)
        base_score = max(template.score, self.settings.viral_min_score)
        adjusted_score = min(
            100.0,
            base_score
            - max(chunk_index - 1, 0) * 4.0
            + (duration_score - template.duration_score) * 10.0,
        )
        return ViralMoment(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            score=adjusted_score,
            hook=hook_text.strip() or template.hook,
            reason=template.reason if chunk_index == 1 else f"{template.reason} (segment {chunk_index})",
            hook_start_seconds=start_seconds,
            core_start_seconds=max(template.core_start_seconds or template.start_seconds, start_seconds),
            core_end_seconds=min(template.core_end_seconds or template.end_seconds, end_seconds),
            hook_strength=template.hook_strength,
            emotion_level=template.emotion_level,
            statement_strength=template.statement_strength,
            novelty=template.novelty,
            duration_score=duration_score,
            phrase_score=template.phrase_score,
        )

    @staticmethod
    def _find_block_index(sentence_blocks: list[SentenceBlock], timestamp: float) -> int | None:
        for index, block in enumerate(sentence_blocks):
            if block.start_seconds <= timestamp <= block.end_seconds:
                return index
        for index, block in enumerate(sentence_blocks):
            if block.start_seconds >= timestamp:
                return index
        return len(sentence_blocks) - 1 if sentence_blocks else None

    @staticmethod
    def _is_duplicate(left: ViralMoment, right: ViralMoment) -> bool:
        overlap = max(0.0, min(left.end_seconds, right.end_seconds) - max(left.start_seconds, right.start_seconds))
        shorter = max(min(left.duration_seconds, right.duration_seconds), 1.0)
        return overlap / shorter >= 0.8

    @staticmethod
    def _strongest_hook_block(blocks: list[SentenceBlock]) -> SentenceBlock:
        return max(
            blocks,
            key=lambda block: (
                sum(1 for pattern in HOOK_PATTERNS if pattern.search(block.text)),
                -len(block.text),
            ),
        )

    def _duration_score(self, duration_seconds: float) -> float:
        target = self.settings.target_clip_duration_seconds
        if duration_seconds < self.settings.min_clip_duration_seconds:
            return max(0.0, duration_seconds / max(self.settings.min_clip_duration_seconds, 1.0))
        if duration_seconds <= self.settings.max_clip_duration_seconds:
            delta = abs(duration_seconds - target)
            return max(0.0, 1.0 - (delta / max(target, 1.0)))
        overflow = duration_seconds - self.settings.max_clip_duration_seconds
        return max(0.0, 0.5 - (overflow / max(self.settings.max_clip_duration_seconds, 1.0)))
