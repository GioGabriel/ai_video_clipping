from __future__ import annotations

import re

from src.core.config import AppSettings
from src.core.models import SentenceBlock, ViralMoment

HOOK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bmost people\b", re.IGNORECASE),
    re.compile(r"\bnobody talks about\b", re.IGNORECASE),
    re.compile(r"\bhere is the truth\b", re.IGNORECASE),
    re.compile(r"\bthis mistake\b", re.IGNORECASE),
    re.compile(r"\bthe truth about\b", re.IGNORECASE),
    re.compile(r"\bwhy does\b", re.IGNORECASE),
    re.compile(r"\bhow do you\b", re.IGNORECASE),
    re.compile(r"\beveryone\b.*\bno one\b", re.IGNORECASE),
    re.compile(r"\beverybody\b.*\bno one\b", re.IGNORECASE),
    re.compile(r"\bpeople\b.*\bbut\b.*\bnot\b", re.IGNORECASE),
    re.compile(r"\bif you\b.*\byou will\b", re.IGNORECASE),
)
OPINIONATED_PHRASES = (
    "wrong",
    "nobody",
    "truth",
    "mistake",
    "millions",
    "discipline",
    "freedom",
    "destroy",
    "nobody talks",
    "nobody wants",
    "jealous",
)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "why",
    "with",
    "you",
}


class HookDetector:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def detect(self, sentence_blocks: list[SentenceBlock], moments: list[ViralMoment]) -> list[ViralMoment]:
        if not sentence_blocks:
            return moments
        return [self._detect_for_moment(sentence_blocks, moment) for moment in moments]

    def _detect_for_moment(self, sentence_blocks: list[SentenceBlock], moment: ViralMoment) -> ViralMoment:
        moment_start = moment.core_start_seconds or moment.start_seconds
        lookback_start = max(moment_start - self.settings.hook_detector_lookback_seconds, 0.0)
        candidate_blocks = [
            block
            for block in sentence_blocks
            if block.end_seconds <= moment_start + 0.05 and block.start_seconds >= lookback_start
        ]
        if not candidate_blocks:
            return moment

        strongest = max(
            candidate_blocks,
            key=lambda block: (
                self._hook_score(block, moment),
                -abs(moment_start - block.start_seconds),
                -block.start_seconds,
            ),
        )
        strongest_score = self._hook_score(strongest, moment)
        current_hook_score = self._raw_text_hook_score(moment.hook)
        should_rewind = strongest_score >= current_hook_score * 0.75
        selected_block = strongest if should_rewind else None
        hook_text = strongest.text if should_rewind else moment.hook
        start_seconds = selected_block.start_seconds if selected_block is not None else moment.start_seconds
        hook_start_seconds = selected_block.start_seconds if selected_block is not None else moment.hook_start_seconds

        return ViralMoment(
            start_seconds=start_seconds,
            end_seconds=moment.end_seconds,
            score=moment.score,
            hook=hook_text.strip() or moment.hook,
            reason=moment.reason,
            hook_start_seconds=hook_start_seconds,
            core_start_seconds=moment.core_start_seconds or moment.start_seconds,
            core_end_seconds=moment.core_end_seconds or moment.end_seconds,
            hook_strength=max(moment.hook_strength, min(strongest_score / 10.0, 1.0)) if should_rewind else moment.hook_strength,
            emotion_level=moment.emotion_level,
            statement_strength=moment.statement_strength,
            novelty=moment.novelty,
            duration_score=moment.duration_score,
            phrase_score=max(moment.phrase_score, strongest.phrase_score) if should_rewind else moment.phrase_score,
        )

    def _hook_score(self, block: SentenceBlock, moment: ViralMoment) -> float:
        lowered = block.text.lower().strip()
        score = self._raw_text_hook_score(block.text)
        if any(pattern.search(lowered) for pattern in HOOK_PATTERNS):
            score += 2.5
        if lowered.endswith("?"):
            score += 0.8
        if any(phrase in lowered for phrase in OPINIONATED_PHRASES):
            score += 1.1
        score += min(block.phrase_score / 20.0, 2.5)
        if "curiosity" in block.detected_triggers:
            score += 0.8
        if "contrarian" in block.detected_triggers:
            score += 0.8

        hook_tokens = self._content_tokens(moment.hook)
        block_tokens = self._content_tokens(block.text)
        if hook_tokens.intersection(block_tokens):
            score += 1.4

        seconds_before_moment = max((moment.core_start_seconds or moment.start_seconds) - block.start_seconds, 0.0)
        proximity_bonus = max(0.0, 1.2 - (seconds_before_moment / max(self.settings.hook_detector_lookback_seconds, 1.0)))
        score += proximity_bonus
        return score

    @staticmethod
    def _raw_text_hook_score(text: str) -> float:
        lowered = text.lower().strip()
        score = 0.0
        if lowered.startswith(("why", "how", "what", "nobody", "most people", "here is", "this mistake")):
            score += 2.2
        if "?" in lowered:
            score += 1.2
        if any(pattern.search(lowered) for pattern in HOOK_PATTERNS):
            score += 2.1
        if len(lowered.split()) <= 16:
            score += 0.7
        if len(lowered.split()) >= 5:
            score += 0.5
        return score

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        return {
            token.lower()
            for token in TOKEN_PATTERN.findall(text)
            if token.lower() not in STOPWORDS and len(token) > 2
        }
