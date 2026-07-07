from __future__ import annotations


def timecode_to_seconds(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Timecode is empty.")

    parts = cleaned.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    raise ValueError(f"Unsupported timecode format: {value}")


def seconds_to_timecode(value: float) -> str:
    total_milliseconds = round(max(value, 0) * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)

    if milliseconds:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def seconds_to_srt_timestamp(value: float) -> str:
    total_milliseconds = round(max(value, 0) * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

