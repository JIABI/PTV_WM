from __future__ import annotations

import logging
import time
from dataclasses import dataclass

LOGGER = logging.getLogger('ralagwm.progress')


@dataclass
class ProgressTracker:
    total: int
    completed: int = 0
    start_time: float = time.time()

    def _elapsed(self) -> float:
        return time.time() - self.start_time

    def start(self, label: str, **meta) -> None:
        payload = ' | '.join([label] + [f'{k}={v}' for k, v in meta.items()])
        LOGGER.info('[%03d/%03d] START %s', self.completed + 1, self.total, payload)

    def done(self, label: str, **meta) -> None:
        self.completed += 1
        payload = ' | '.join([label] + [f'{k}={v}' for k, v in meta.items()])
        LOGGER.info('[%03d/%03d] DONE  %s | elapsed=%.1fs', self.completed, self.total, payload, self._elapsed())

    def fail(self, label: str, error: Exception | str) -> None:
        self.completed += 1
        LOGGER.error('[%03d/%03d] FAIL  %s | error=%s | elapsed=%.1fs', self.completed, self.total, label, error, self._elapsed())
