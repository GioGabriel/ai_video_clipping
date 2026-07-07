from __future__ import annotations

import re

from src.core.config import AppSettings
from src.core.models import EditPlan, HookOverlay, TranscriptArtifact, TranscriptWord, ViralMoment, ZoomEffect

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
SENTENCE_BREAK_PATTERN = re.compile(r"[.!?]")
CLAUSE_BREAK_PATTERNS = (
    re.compile(r"\s*,\s*"),
    re.compile(r"\s+\bbut\b\s+", re.IGNORECASE),
    re.compile(r"\s+\bbecause\b\s+", re.IGNORECASE),
    re.compile(r"\s+\bwhen\b\s+", re.IGNORECASE),
    re.compile(r"\s+\bwhich\b\s+", re.IGNORECASE),
)
LEADING_FILLER_PATTERNS = (
    re.compile(r"^(and|so|but)\s+", re.IGNORECASE),
    re.compile(r"^number\s+\w+\s+is\s+(just\s+)?(to\s+)?", re.IGNORECASE),
    re.compile(r"^the\s+(first|second|third)\s+thing\s+is\s+(just\s+)?(to\s+)?", re.IGNORECASE),
    re.compile(r"^(and\s+so\s+that\s+goes\s+to\s+)?number\s+\w+\s*,?\s*which\s+is\s+(just\s+)?(to\s+)?", re.IGNORECASE),
)
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
    "get",
    "got",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
    "your",
}
IMPACT_WORDS = {
    "always",
    "best",
    "crazy",
    "dangerous",
    "destroy",
    "everything",
    "insane",
    "massive",
    "mistake",
    "never",
    "nothing",
    "secret",
    "stop",
    "truth",
    "viral",
    "worst",
}


class EditPlanner:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def plan(self, transcript: TranscriptArtifact, moments: list[ViralMoment]) -> list[EditPlan]:
        flattened_words = self._flatten_words(transcript)
        return [self._build_plan(moment, flattened_words) for moment in moments]

    def _build_plan(self, moment: ViralMoment, words: list[TranscriptWord]) -> EditPlan:
        clip_start = max(moment.start_seconds, 0.0)
        clip_end = max(moment.end_seconds, clip_start + 1.0)
        clip_duration = clip_end - clip_start
        clip_words = [
            word
            for word in words
            if word.end_seconds > clip_start and word.start_seconds < clip_end
        ]

        zoom_effects = self._build_zoom_effects(
            clip_words=clip_words,
            clip_start=clip_start,
            clip_duration=clip_duration,
            hook=moment.hook,
        )
        hook_overlay = self._build_hook_overlay(moment.hook, clip_duration)

        return EditPlan(
            start_seconds=clip_start,
            end_seconds=clip_end,
            score=moment.score,
            hook=moment.hook,
            reason=moment.reason,
            zoom_effects=zoom_effects,
            hook_overlay=hook_overlay,
        )

    @staticmethod
    def _flatten_words(transcript: TranscriptArtifact) -> list[TranscriptWord]:
        words: list[TranscriptWord] = []
        for segment in transcript.segments:
            if segment.words:
                words.extend(segment.words)
                continue

            segment_tokens = [token for token in segment.text.split() if token.strip()]
            if not segment_tokens:
                continue

            duration = max(segment.end_seconds - segment.start_seconds, 0.4)
            slice_duration = duration / len(segment_tokens)
            cursor = segment.start_seconds
            for index, token in enumerate(segment_tokens):
                end_seconds = segment.end_seconds if index == len(segment_tokens) - 1 else cursor + slice_duration
                words.append(
                    TranscriptWord(
                        start_seconds=cursor,
                        end_seconds=end_seconds,
                        text=token.strip(),
                    )
                )
                cursor = end_seconds

        return words

    def _build_zoom_effects(
        self,
        clip_words: list[TranscriptWord],
        clip_start: float,
        clip_duration: float,
        hook: str,
    ) -> list[ZoomEffect]:
        if not self.settings.edit_plan_zoom_enabled:
            return []

        if not clip_words:
            return []

        ordered_hook_terms = [
            token.lower()
            for token in TOKEN_PATTERN.findall(hook)
            if token.lower() not in STOPWORDS
        ]
        hook_terms = set(ordered_hook_terms)
        terminal_hook_term = ordered_hook_terms[-1] if ordered_hook_terms else None
        scored_candidates: list[tuple[float, TranscriptWord]] = []
        total_words = len(clip_words)
        for index, word in enumerate(clip_words):
            token = self._normalize_token(word.text)
            if not token:
                continue

            score = 0.0
            if token in hook_terms:
                score += 4.0
            if terminal_hook_term and token == terminal_hook_term:
                score += 2.0
            if token not in STOPWORDS:
                score += 1.0
            if token in IMPACT_WORDS:
                score += 1.4
            if len(token) >= 7:
                score += 1.1
            if word.text.strip().endswith(("!", "?")):
                score += 0.9
            if index <= 2:
                score += 0.8
            if index >= max(total_words - 3, 0):
                score += 0.5

            if score >= 1.6:
                scored_candidates.append((score, word))

        selected: list[ZoomEffect] = []
        for score, word in sorted(scored_candidates, key=lambda item: (-item[0], item[1].start_seconds)):
            relative_start = max(word.start_seconds - clip_start - self.settings.edit_plan_zoom_pre_roll_seconds, 0.0)
            relative_end = min(
                word.end_seconds - clip_start + self.settings.edit_plan_zoom_post_roll_seconds,
                clip_duration,
            )
            if relative_end - relative_start < 0.18:
                relative_end = min(relative_start + 0.18, clip_duration)

            if any(abs(relative_start - effect.start_seconds) < self.settings.edit_plan_zoom_min_gap_seconds for effect in selected):
                continue

            normalized_score = min(score / 5.5, 1.0)
            peak_scale = self.settings.edit_plan_zoom_min_scale + (
                (self.settings.edit_plan_zoom_max_scale - self.settings.edit_plan_zoom_min_scale) * normalized_score
            )
            selected.append(
                ZoomEffect(
                    start_seconds=relative_start,
                    end_seconds=max(relative_end, relative_start + 0.18),
                    peak_scale=round(peak_scale, 3),
                    anchor_text=word.text.strip(),
                )
            )
            if len(selected) >= self.settings.edit_plan_max_zoom_beats:
                break

        return sorted(selected, key=lambda item: item.start_seconds)

    def _build_hook_overlay(self, hook: str, clip_duration: float) -> HookOverlay | None:
        cleaned = " ".join(hook.strip().split())
        if not cleaned:
            return None

        summary = self._summarize_overlay_text(cleaned)
        if not summary:
            return None

        return HookOverlay(
            text=self._split_overlay_text(summary),
            start_seconds=0.0,
            end_seconds=min(self.settings.edit_plan_hook_overlay_seconds, clip_duration),
        )

    def _summarize_overlay_text(self, text: str) -> str:
        summary = text.strip()
        sentence_match = SENTENCE_BREAK_PATTERN.search(summary)
        if sentence_match is not None:
            summary = summary[: sentence_match.start()].strip() or summary

        for pattern in LEADING_FILLER_PATTERNS:
            summary = pattern.sub("", summary).strip()

        clause_candidates = [summary]
        for pattern in CLAUSE_BREAK_PATTERNS:
            split_parts = [part.strip() for part in pattern.split(summary) if part.strip()]
            if split_parts:
                clause_candidates.append(split_parts[0])

        summary = min(
            (candidate for candidate in clause_candidates if candidate),
            key=lambda candidate: (len(candidate) > 42, len(candidate.split()) > 8, len(candidate)),
            default=summary,
        )

        words = summary.split()
        if len(words) > 8:
            summary = " ".join(words[:8])
        if len(summary) > 42:
            trimmed = summary[:41].rstrip()
            if " " in trimmed:
                trimmed = trimmed.rsplit(" ", 1)[0]
            summary = trimmed + "…"

        if not summary:
            fallback_words = text.split()[:6]
            summary = " ".join(fallback_words)
        return summary.strip()

    @staticmethod
    def _split_overlay_text(text: str) -> str:
        words = text.split()
        if len(words) <= 3:
            return text

        midpoint = len(words) // 2
        best_split = midpoint
        best_delta = float("inf")
        for index in range(2, len(words)):
            left = " ".join(words[:index])
            right = " ".join(words[index:])
            delta = abs(len(left) - len(right))
            if delta < best_delta:
                best_delta = delta
                best_split = index
        return f"{' '.join(words[:best_split])}\n{' '.join(words[best_split:])}"

    @staticmethod
    def _normalize_token(value: str) -> str:
        match = TOKEN_PATTERN.search(value)
        return match.group(0).lower() if match else ""
