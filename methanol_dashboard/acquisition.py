import threading
import time
from collections import deque
from typing import Deque, Dict, Any, Optional, TYPE_CHECKING

from .io_link_master import IoLinkMaster

if TYPE_CHECKING:
    from .experiment_log import ExperimentLogger


class AcquisitionManager:
    """
    Background data acquisition manager.

    - Periodically calls IoLinkMaster.sample_all_sensors(...)
    - Keeps the most recent sample
    - Keeps a rolling history (limited by history_seconds)
    - Optionally forwards samples to an ExperimentLogger
    """

    def __init__(
        self,
        *,
        io_master: IoLinkMaster,
        sample_period_s: float,
        history_seconds: float,
        sample_kwargs: Optional[Dict[str, Any]] = None,
        experiment_logger: "Optional[ExperimentLogger]" = None,
    ) -> None:
        self.io_master = io_master
        self.sample_period_s = sample_period_s
        self.history_seconds = history_seconds
        self.sample_kwargs = sample_kwargs or {}
        self.experiment_logger = experiment_logger

        self._latest: Optional[Dict[str, Any]] = None
        self._history: Deque[Dict[str, Any]] = deque()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._latest) if self._latest is not None else None

    def get_history(self, max_age_s: Optional[float] = None) -> list[Dict[str, Any]]:
        with self._lock:
            if max_age_s is None:
                max_age_s = self.history_seconds
            now = time.time()
            return [
                dict(row)
                for row in self._history
                if (now - float(row.get("timestamp_unix_s", now))) <= max_age_s
            ]

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            start = time.time()
            try:
                row = self.io_master.sample_all_sensors(**self.sample_kwargs)
            except Exception as exc:
                print(f"[Acquisition] error while sampling: {exc}")
                row = None

            if row is not None:
                with self._lock:
                    self._latest = row
                    self._history.append(row)

                    now = time.time()
                    while self._history and (
                        now - float(self._history[0].get("timestamp_unix_s", now))
                        > self.history_seconds
                    ):
                        self._history.popleft()

                if self.experiment_logger is not None:
                    try:
                        self.experiment_logger.log_row(row)
                    except Exception as exc:
                        print(f"[Acquisition] error logging row: {exc}")

            elapsed = time.time() - start
            to_sleep = max(0.0, self.sample_period_s - elapsed)
            time.sleep(to_sleep)
