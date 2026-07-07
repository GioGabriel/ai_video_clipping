from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx

from src.core.config import AppSettings
from src.core.models import SentenceBlock, TranscriptSegment, ViralMoment
from src.core.timecode import seconds_to_timecode, timecode_to_seconds
from src.modules.viral_phrase_classifier.service import ViralPhraseClassifier

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
QUESTION_START_PATTERN = re.compile(r"^(why|how|what|when|are|do|does|did|can|should|could)\b", re.IGNORECASE)
QUESTION_ANSWER_PATTERN = re.compile(r"\?\s*because\b", re.IGNORECASE)
HOOK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bmost people\b", re.IGNORECASE),
    re.compile(r"\bnobody talks about\b", re.IGNORECASE),
    re.compile(r"\bhere is the truth\b", re.IGNORECASE),
    re.compile(r"\bthis mistake\b", re.IGNORECASE),
    re.compile(r"\beveryone\b.*\bno one\b", re.IGNORECASE),
    re.compile(r"\beverybody\b.*\bno one\b", re.IGNORECASE),
    re.compile(r"\bpeople\b.*\bbut\b.*\bnot\b", re.IGNORECASE),
    re.compile(r"\bif you\b.*\byou will\b", re.IGNORECASE),
    re.compile(r"\bdiscipline\b.*\bfreedom\b", re.IGNORECASE),
    re.compile(r"\bhard now\b.*\beasy later\b", re.IGNORECASE),
    re.compile(r"\beasy now\b.*\bhard later\b", re.IGNORECASE),
    re.compile(r"\bgiving up is the easiest thing\b", re.IGNORECASE),
    re.compile(r"\btrain yourself\b", re.IGNORECASE),
    re.compile(r"\bdo what needs to be done\b", re.IGNORECASE),
    re.compile(r"\bregardless of how you feel\b", re.IGNORECASE),
    re.compile(r"\bfollow through\b", re.IGNORECASE),
    re.compile(r"\bconsistent people win\b", re.IGNORECASE),
    re.compile(r"\bdo it scared\b", re.IGNORECASE),
    re.compile(r"\bwhen your why is strong enough\b", re.IGNORECASE),
)
EMOTION_TERMS = {
    "crazy",
    "destroy",
    "freedom",
    "fear",
    "god",
    "hate",
    "jealous",
    "love",
    "mistake",
    "powerful",
    "regret",
    "slave",
    "truth",
    "win",
    "wrong",
}
NOVELTY_TERMS = {
    "millions",
    "nobody",
    "nobody talks",
    "nobody wants",
    "overnight",
    "secret",
    "truth",
    "wrong",
}
STATEMENT_TERMS = {
    "always",
    "discipline",
    "consistent",
    "freedom",
    "hard",
    "must",
    "never",
    "nobody",
    "follow",
    "through",
    "train",
    "truth",
    "wrong",
}
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
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "with",
    "you",
    "your",
}


class ViralMomentDetector:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.prompt_template = Path(__file__).with_name("prompt_template.txt").read_text(encoding="utf-8")
        self._client = httpx.Client(
            base_url=self.settings.ollama_base_url,
            timeout=self.settings.ollama_request_timeout_seconds,
        )

    def close(self) -> None:
        self._client.close()

    def detect(
        self,
        segments: Sequence[TranscriptSegment | SentenceBlock],
        model_name: str | None = None,
        progress_callback: Callable[[str, int | None, int | None], None] | None = None,
    ) -> list[ViralMoment]:
        ordered_segments = sorted(segments, key=lambda segment: segment.start_seconds)
        if not ordered_segments:
            raise RuntimeError("Transcript did not contain any segments to analyze.")
        effective_model = str(model_name or self.settings.ollama_model).strip() or self.settings.ollama_model

        windows = self._build_windows(ordered_segments)
        candidates: list[ViralMoment] = []
        successful_windows = 0
        last_error: str | None = None

        for index, window in enumerate(windows, start=1):
            if progress_callback is not None:
                progress_callback(
                    f"Scoring transcript chunk {index}/{len(windows)} "
                    f"({seconds_to_timecode(window['start'])} to {seconds_to_timecode(window['end'])}) with {effective_model}.",
                    index,
                    len(windows),
                )

            prompt = self.prompt_template.format(
                chunk_start=seconds_to_timecode(window["start"]),
                chunk_end=seconds_to_timecode(window["end"]),
                transcript=window["transcript"],
                min_duration=int(self.settings.min_clip_duration_seconds),
                max_duration=int(self.settings.max_clip_duration_seconds),
                max_candidates=self.settings.max_candidates_per_chunk,
            )

            try:
                parsed = self._generate_candidates(prompt, effective_model)
                successful_windows += 1
            except RuntimeError as exc:
                last_error = str(exc)
                logger.warning(
                    "Skipping transcript chunk %s-%s after Ollama/parsing error: %s",
                    seconds_to_timecode(window["start"]),
                    seconds_to_timecode(window["end"]),
                    exc,
                )
                continue

            chunk_count = 0
            for item in parsed.get("moments", []):
                moment = self._normalize_moment(item, ordered_segments, window["start"], window["end"])
                if moment is None:
                    continue
                candidates.append(moment)
                chunk_count += 1

            if progress_callback is not None:
                progress_callback(
                    f"Chunk {index}/{len(windows)} returned {chunk_count} scored candidate clip(s).",
                    index,
                    len(windows),
                )

        candidates.extend(self._generate_heuristic_candidates(ordered_segments))
        filtered = [candidate for candidate in self._deduplicate(candidates) if candidate.score >= self.settings.viral_min_score]
        if not filtered:
            if successful_windows == 0 and last_error:
                raise RuntimeError(last_error)
            raise RuntimeError("Ollama returned responses, but none contained usable viral clip candidates.")
        return filtered

    def _generate_candidates(self, prompt: str, model_name: str | None = None) -> dict[str, Any]:
        effective_model = str(model_name or self.settings.ollama_model).strip() or self.settings.ollama_model
        try:
            response = self._client.post(
                "/api/generate",
                json={
                    "model": effective_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.15},
                },
            )
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.settings.ollama_base_url}. Start it with `ollama serve`."
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ollama timed out after {self.settings.ollama_request_timeout_seconds}s while scoring transcript chunks."
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Failed to reach Ollama at {self.settings.ollama_base_url}: {exc}") from exc

        if response.status_code >= 400:
            raise self._build_http_error(response, effective_model)

        try:
            raw_payload = response.json().get("response", "")
        except ValueError as exc:
            raise RuntimeError("Ollama returned a non-JSON response.") from exc

        if not str(raw_payload).strip():
            raise RuntimeError("Ollama returned an empty response.")

        try:
            return self._extract_json_payload(str(raw_payload))
        except ValueError as exc:
            raise RuntimeError("Ollama returned a response, but it did not contain valid JSON clip data.") from exc

    def _build_http_error(self, response: httpx.Response, model_name: str | None = None) -> RuntimeError:
        message = ""
        effective_model = str(model_name or self.settings.ollama_model).strip() or self.settings.ollama_model
        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if isinstance(payload, dict):
            message = str(payload.get("error", "")).strip()

        normalized = message.lower()
        if response.status_code == 404 and "model" in normalized and "not found" in normalized:
            return RuntimeError(
                f"Ollama model '{effective_model}' is not installed. Run `ollama pull {effective_model}`."
            )

        if message:
            return RuntimeError(f"Ollama request failed with HTTP {response.status_code}: {message}")
        return RuntimeError(f"Ollama request failed with HTTP {response.status_code}.")

    def _build_windows(self, segments: Sequence[TranscriptSegment | SentenceBlock]) -> list[dict[str, Any]]:
        if not segments:
            return []

        windows: list[dict[str, Any]] = []
        step = max(self.settings.viral_chunk_duration_seconds - self.settings.viral_chunk_overlap_seconds, 1)
        cursor = max(0.0, segments[0].start_seconds)
        transcript_end = segments[-1].end_seconds

        while cursor < transcript_end:
            window_end = cursor + self.settings.viral_chunk_duration_seconds
            included = [segment for segment in segments if segment.end_seconds > cursor and segment.start_seconds < window_end]
            if included:
                lines = []
                for segment in included:
                    speaker = getattr(segment, "speaker", "speaker_1")
                    trigger_labels = getattr(segment, "detected_triggers", [])
                    trigger_suffix = f" [phrases: {', '.join(trigger_labels)}]" if trigger_labels else ""
                    lines.append(
                        f"[{seconds_to_timecode(segment.start_seconds)} - {seconds_to_timecode(segment.end_seconds)}] "
                        f"{speaker}: {segment.text}{trigger_suffix}"
                    )
                windows.append(
                    {
                        "start": cursor,
                        "end": window_end,
                        "transcript": "\n".join(lines),
                    }
                )
            cursor += step

        return windows

    def _normalize_moment(
        self,
        item: dict[str, Any],
        segments: Sequence[TranscriptSegment | SentenceBlock],
        window_start: float,
        window_end: float,
    ) -> ViralMoment | None:
        start_seconds = timecode_to_seconds(item.get("start_time", item.get("start", 0)))
        end_seconds = timecode_to_seconds(item.get("end_time", item.get("end", start_seconds)))
        if end_seconds <= start_seconds:
            return None
        if start_seconds > window_end + 5 or end_seconds < window_start - 5:
            return None

        overlapping = [segment for segment in segments if segment.end_seconds > start_seconds and segment.start_seconds < end_seconds]
        if not overlapping:
            return None

        aligned_start = max(overlapping[0].start_seconds, 0.0)
        aligned_end = max(overlapping[-1].end_seconds, aligned_start + self.settings.min_clip_duration_seconds)
        hook = str(item.get("hook") or "").strip() or overlapping[0].text.strip()
        reason = str(item.get("reason") or "").strip()
        text = " ".join(segment.text.strip() for segment in overlapping if segment.text.strip())
        hook_strength, emotion_level, statement_strength, novelty, duration_score = self._score_components(
            hook=hook,
            text=text,
            duration_seconds=aligned_end - aligned_start,
        )
        viral_detection_score = round(
            (
                hook_strength * 0.35
                + emotion_level * 0.25
                + statement_strength * 0.20
                + novelty * 0.10
                + duration_score * 0.10
            )
            * 100,
            2,
        )
        phrase_score = self._phrase_score(overlapping)
        score = round(max(viral_detection_score, (viral_detection_score * 0.6) + (phrase_score * 0.4)), 2)
        if score < self.settings.viral_min_score:
            return None

        if not reason:
            reason = self._build_reason(
                hook_strength,
                emotion_level,
                statement_strength,
                novelty,
                self._phrase_trigger_summary(overlapping),
            )

        return ViralMoment(
            start_seconds=aligned_start,
            end_seconds=aligned_end,
            score=score,
            hook=self._short_hook(hook),
            reason=reason,
            hook_start_seconds=aligned_start,
            core_start_seconds=aligned_start,
            core_end_seconds=aligned_end,
            hook_strength=hook_strength,
            emotion_level=emotion_level,
            statement_strength=statement_strength,
            novelty=novelty,
            duration_score=duration_score,
            phrase_score=phrase_score,
        )

    def _generate_heuristic_candidates(self, segments: Sequence[TranscriptSegment | SentenceBlock]) -> list[ViralMoment]:
        candidates: list[ViralMoment] = []
        for index, segment in enumerate(segments):
            heuristic_score = self._heuristic_block_score(segment)
            if heuristic_score < 2.6:
                continue

            end_index = index
            while end_index + 1 < len(segments):
                next_segment = segments[end_index + 1]
                gap = next_segment.start_seconds - segments[end_index].end_seconds
                if gap > self.settings.clip_sentence_gap_threshold_seconds:
                    break
                combined_duration = next_segment.end_seconds - segment.start_seconds
                if combined_duration > self.settings.max_clip_duration_seconds:
                    break
                if not self._share_theme(segments[end_index].text, next_segment.text) and heuristic_score < 3.4:
                    break
                end_index += 1
                if combined_duration >= self.settings.target_clip_duration_seconds:
                    break

            text = " ".join(item.text.strip() for item in segments[index : end_index + 1])
            hook = segments[index].text.strip()
            duration = max(
                segments[end_index].end_seconds - segment.start_seconds,
                self.settings.min_clip_duration_seconds,
            )
            hook_strength, emotion_level, statement_strength, novelty, duration_score = self._score_components(
                hook=hook,
                text=text,
                duration_seconds=duration,
            )
            phrase_score = self._phrase_score(segments[index : end_index + 1])
            viral_detection_score = round(
                (
                    hook_strength * 0.35
                    + emotion_level * 0.25
                    + statement_strength * 0.20
                    + novelty * 0.10
                    + duration_score * 0.10
                )
                * 100,
                2,
            )
            score = round(max(viral_detection_score, (viral_detection_score * 0.6) + (phrase_score * 0.4)), 2)
            if score < self.settings.viral_min_score:
                continue

            candidates.append(
                ViralMoment(
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.start_seconds + duration,
                    score=score,
                    hook=self._short_hook(hook),
                    reason=self._build_reason(
                        hook_strength,
                        emotion_level,
                        statement_strength,
                        novelty,
                        self._phrase_trigger_summary(segments[index : end_index + 1]),
                    ),
                    hook_start_seconds=segment.start_seconds,
                    core_start_seconds=segment.start_seconds,
                    core_end_seconds=segments[end_index].end_seconds,
                    hook_strength=hook_strength,
                    emotion_level=emotion_level,
                    statement_strength=statement_strength,
                    novelty=novelty,
                    duration_score=duration_score,
                    phrase_score=phrase_score,
                )
            )
        return candidates

    def _heuristic_block_score(self, segment: TranscriptSegment | SentenceBlock) -> float:
        text = segment.text
        lowered = text.lower().strip()
        score = 0.0
        phrase_score = float(getattr(segment, "phrase_score", 0.0))
        detected_triggers = list(getattr(segment, "detected_triggers", []))
        if phrase_score <= 0:
            phrase_score, detected_triggers = ViralPhraseClassifier.classify_text(text)
        if QUESTION_ANSWER_PATTERN.search(lowered):
            score += 2.8
        if QUESTION_START_PATTERN.search(lowered):
            score += 1.6
        if any(pattern.search(lowered) for pattern in HOOK_PATTERNS):
            score += 2.4
        if any(term in lowered for term in EMOTION_TERMS):
            score += 1.2
        if any(term in lowered for term in NOVELTY_TERMS):
            score += 1.0
        if text.count("!") > 0 or text.count("?") > 0:
            score += 0.6
        if "advice" in detected_triggers:
            score += 1.0
        if "contrarian" in detected_triggers or "thesis" in detected_triggers:
            score += 1.2
        score += min(phrase_score / 20.0, 2.5)
        return score

    def _score_components(self, hook: str, text: str, duration_seconds: float) -> tuple[float, float, float, float, float]:
        hook_strength = self._normalize_component(self._hook_strength(hook))
        emotion_level = self._normalize_component(self._emotion_level(text))
        statement_strength = self._normalize_component(self._statement_strength(text))
        novelty = self._normalize_component(self._novelty_score(text))
        duration_score = self._duration_score(duration_seconds)
        return hook_strength, emotion_level, statement_strength, novelty, duration_score

    def _hook_strength(self, text: str) -> float:
        lowered = text.lower().strip()
        score = 0.0
        if QUESTION_ANSWER_PATTERN.search(lowered):
            score += 4.0
        if QUESTION_START_PATTERN.search(lowered):
            score += 2.0
        if any(pattern.search(lowered) for pattern in HOOK_PATTERNS):
            score += 3.2
        if lowered.startswith(("most people", "nobody", "here is", "this mistake", "why")):
            score += 1.8
        if 4 <= len(lowered.split()) <= 18:
            score += 1.0
        return score

    def _emotion_level(self, text: str) -> float:
        lowered = text.lower()
        token_matches = [token.lower() for token in TOKEN_PATTERN.findall(lowered)]
        score = sum(1.0 for token in token_matches if token in EMOTION_TERMS)
        if "!" in text or "?" in text:
            score += 0.8
        if "my god" in lowered or "i love" in lowered:
            score += 1.0
        return score

    def _statement_strength(self, text: str) -> float:
        lowered = text.lower()
        token_matches = [token.lower() for token in TOKEN_PATTERN.findall(lowered)]
        score = sum(0.7 for token in token_matches if token in STATEMENT_TERMS)
        if any(pattern.search(lowered) for pattern in HOOK_PATTERNS):
            score += 1.4
        if lowered.startswith(("do ", "don't ", "do not ", "go ", "stop ")):
            score += 1.2
        if " is " in lowered or " are " in lowered:
            score += 0.6
        return score

    def _novelty_score(self, text: str) -> float:
        lowered = text.lower()
        tokens = {
            token.lower()
            for token in TOKEN_PATTERN.findall(lowered)
            if token.lower() not in STOPWORDS and len(token) > 2
        }
        score = sum(0.8 for term in NOVELTY_TERMS if term in lowered)
        if len(tokens) >= 8:
            score += 0.6
        if len(tokens) >= 12:
            score += 0.4
        return score

    def _duration_score(self, duration_seconds: float) -> float:
        if duration_seconds < self.settings.min_clip_duration_seconds:
            return max(0.0, duration_seconds / max(self.settings.min_clip_duration_seconds, 1.0))
        if 25.0 <= duration_seconds <= 35.0:
            return 1.0
        if duration_seconds < 25.0:
            return max(0.0, 1.0 - ((25.0 - duration_seconds) / 10.0))
        if duration_seconds <= self.settings.max_clip_duration_seconds:
            return max(0.0, 1.0 - ((duration_seconds - 35.0) / 15.0))
        return 0.0

    @staticmethod
    def _normalize_component(value: float) -> float:
        return max(0.0, min(value / 5.0, 1.0))

    @staticmethod
    def _build_reason(
        hook_strength: float,
        emotion_level: float,
        statement_strength: float,
        novelty: float,
        phrase_summary: str | None = None,
    ) -> str:
        signals = [
            ("Strong hook", hook_strength),
            ("emotional delivery", emotion_level),
            ("bold statement", statement_strength),
            ("novel framing", novelty),
        ]
        top_signals = [label for label, _ in sorted(signals, key=lambda item: item[1], reverse=True)[:2] if _ > 0.2]
        if phrase_summary:
            top_signals.append(phrase_summary)
        if top_signals:
            return " + ".join(dict.fromkeys(top_signals))
        return "Complete idea with short-form potential"

    @staticmethod
    def _phrase_score(segments: Sequence[TranscriptSegment | SentenceBlock]) -> float:
        if not segments:
            return 0.0
        phrase_scores = []
        for segment in segments:
            raw_score = float(getattr(segment, "phrase_score", 0.0))
            if raw_score <= 0:
                raw_score, _ = ViralPhraseClassifier.classify_text(segment.text)
            phrase_scores.append(raw_score)
        strongest = max(phrase_scores, default=0.0)
        leading = phrase_scores[0] if phrase_scores else 0.0
        raw_phrase_score = strongest * 0.75 + leading * 0.25
        return ViralMomentDetector._normalize_phrase_score(raw_phrase_score)

    @staticmethod
    def _phrase_trigger_summary(segments: Sequence[TranscriptSegment | SentenceBlock]) -> str | None:
        ordered_triggers: list[str] = []
        for segment in segments:
            detected_triggers = list(getattr(segment, "detected_triggers", []))
            if not detected_triggers:
                _, detected_triggers = ViralPhraseClassifier.classify_text(segment.text)
            for trigger in detected_triggers:
                if trigger not in ordered_triggers:
                    ordered_triggers.append(trigger)
        if not ordered_triggers:
            return None
        labels = {
            "curiosity": "curiosity hook",
            "contrarian": "contrarian phrasing",
            "advice": "direct advice",
            "emotional": "emotional phrasing",
            "list": "list framing",
            "thesis": "strong thesis line",
        }
        return labels.get(ordered_triggers[0], ordered_triggers[0])

    @staticmethod
    def _normalize_phrase_score(raw_phrase_score: float) -> float:
        return min(100.0, raw_phrase_score * 2.5)

    def _deduplicate(self, moments: list[ViralMoment]) -> list[ViralMoment]:
        kept: list[ViralMoment] = []
        for moment in sorted(moments, key=lambda item: (item.score, -item.start_seconds), reverse=True):
            if any(self._is_duplicate(moment, existing) for existing in kept):
                continue
            kept.append(moment)
            if len(kept) >= self.settings.max_clip_candidates:
                break
        return sorted(kept, key=lambda item: item.start_seconds)

    @staticmethod
    def _is_duplicate(left: ViralMoment, right: ViralMoment) -> bool:
        overlap = max(0.0, min(left.end_seconds, right.end_seconds) - max(left.start_seconds, right.start_seconds))
        shorter_duration = max(1.0, min(left.duration_seconds, right.duration_seconds))
        return overlap / shorter_duration >= 0.65

    @staticmethod
    def _extract_json_payload(raw_payload: str) -> dict[str, Any]:
        cleaned = raw_payload.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        for opening, closing in (("{", "}"), ("[", "]")):
            start = cleaned.find(opening)
            end = cleaned.rfind(closing)
            if start == -1 or end == -1 or end <= start:
                continue
            snippet = cleaned[start : end + 1]
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                return {"moments": parsed}
            if isinstance(parsed, dict):
                return parsed

        raise ValueError("Unable to locate valid JSON in Ollama response.")

    @staticmethod
    def _share_theme(left_text: str, right_text: str) -> bool:
        left_tokens = {
            token.lower()
            for token in TOKEN_PATTERN.findall(left_text)
            if token.lower() not in STOPWORDS and len(token) > 2
        }
        right_tokens = {
            token.lower()
            for token in TOKEN_PATTERN.findall(right_text)
            if token.lower() not in STOPWORDS and len(token) > 2
        }
        return bool(left_tokens.intersection(right_tokens))

    @staticmethod
    def _short_hook(text: str) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) > 110:
            return cleaned[:109].rstrip() + "…"
        return cleaned or "Untitled clip"
