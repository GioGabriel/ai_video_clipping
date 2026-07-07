from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class ProcessingJobQueue:
    def __init__(self, worker_count: int, job_handler: Callable[[str], None]) -> None:
        self.worker_count = max(worker_count, 1)
        self.job_handler = job_handler
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._started = False

    def start(self) -> None:
        if self._started:
            return

        for worker_index in range(self.worker_count):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(worker_index + 1,),
                daemon=True,
                name=f"clip-worker-{worker_index + 1}",
            )
            worker.start()
            self._workers.append(worker)

        self._started = True
        logger.info("Started %s processing worker(s).", self.worker_count)

    def stop(self) -> None:
        if not self._started:
            return

        for _ in self._workers:
            self._queue.put(None)

        for worker in self._workers:
            worker.join(timeout=5)

        self._workers.clear()
        self._started = False
        logger.info("Stopped processing workers.")

    def submit(self, job_id: str) -> None:
        logger.info("Queued job %s", job_id)
        self._queue.put(job_id)

    def _worker_loop(self, worker_id: int) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                logger.info("Worker %s started job %s", worker_id, job_id)
                self.job_handler(job_id)
            except Exception:
                logger.exception("Worker %s failed while handling job %s", worker_id, job_id)
            finally:
                self._queue.task_done()

