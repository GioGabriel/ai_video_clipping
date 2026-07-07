from __future__ import annotations

import logging
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


class CommandError(RuntimeError):
    pass


class CommandRunner:
    def run(
        self,
        command: Sequence[str],
        cwd: Path | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        logger.info("Running command: %s", shlex.join(command))
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        output_chunks: list[str] = []
        assert process.stdout is not None
        for raw_line in process.stdout:
            output_chunks.append(raw_line)
            line = raw_line.rstrip()
            if not line:
                continue
            logger.debug("Command output: %s", line)
            if on_output is not None:
                try:
                    on_output(line)
                except Exception:  # pragma: no cover - defensive logging path
                    logger.exception("Command output handler failed while processing: %s", line)

        return_code = process.wait()
        stdout = "".join(output_chunks)
        result = subprocess.CompletedProcess(
            args=list(command),
            returncode=return_code,
            stdout=stdout,
            stderr="",
        )

        if result.returncode != 0:
            raise CommandError(
                f"Command failed ({result.returncode}): {shlex.join(command)}\n"
                f"{result.stdout.strip()}"
            )

        return result
