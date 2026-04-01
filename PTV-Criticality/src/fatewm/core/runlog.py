"""Runtime logging utilities.

We keep this intentionally lightweight (no external logging deps).
When using Hydra, we write a `train.log` file into the Hydra output dir
and tee stdout/stderr to it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class _Tee:
    stream: object
    file: object

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def setup_train_log(output_dir: str, filename: str = "train.log") -> Optional[str]:
    """Tee stdout/stderr to a log file under `output_dir`.

    Returns the log path if successfully set, else None.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, filename)
        f = open(log_path, "a", encoding="utf-8")
        sys.stdout = _Tee(sys.stdout, f)  # type: ignore
        sys.stderr = _Tee(sys.stderr, f)  # type: ignore
        return log_path
    except Exception:
        return None
