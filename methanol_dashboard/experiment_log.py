import csv
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Deque

from zoneinfo import ZoneInfo

AMS_TZ = ZoneInfo("Europe/Amsterdam")


@dataclass
class ExperimentLogger:
    base_dir: Path

    active: bool = field(default=False, init=False)

    # Main sensor CSV
    _csv_path: Optional[Path] = field(default=None, init=False)
    _csv_file: Optional[Any] = field(default=None, init=False)
    _writer: Optional[csv.DictWriter] = field(default=None, init=False)
    _fieldnames: Optional[List[str]] = field(default=None, init=False)

    # NEW: Events CSV
    _events_path: Optional[Path] = field(default=None, init=False)
    _events_file: Optional[Any] = field(default=None, init=False)
    _events_writer: Optional[csv.DictWriter] = field(default=None, init=False)

    _meta: Dict[str, Any] = field(default_factory=dict, init=False)

    # Thread-safety + dashboard buffer
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _recent_events: Deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=200),
        init=False,
        repr=False,
    )

    def start_experiment(self, meta: Dict[str, Any]) -> None:
        with self._lock:
            if self.active:
                self.stop_experiment()

            self.base_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(AMS_TZ).strftime("%Y%m%d_%H%M%S")
            name = meta.get("name") or meta.get("title") or "experiment"
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

            stem = f"{timestamp}_{safe_name}"

            # Main sensor CSV
            self._csv_path = self.base_dir / f"{stem}.csv"
            self._csv_file = self._csv_path.open("w", newline="")
            self._writer = None
            self._fieldnames = None

            # Events CSV (header is known up-front, so write immediately)
            self._events_path = self.base_dir / f"{stem}_events.csv"
            self._events_file = self._events_path.open("w", newline="")
            self._events_writer = csv.DictWriter(
                self._events_file,
                fieldnames=["timestamp_ams", "timestamp_unix_s", "event"],
            )
            self._events_writer.writeheader()
            self._events_file.flush()

            # Meta JSON
            meta_path = self.base_dir / f"{stem}_meta.json"
            self._meta = dict(meta)
            self._meta["start_time_ams"] = timestamp
            meta_path.write_text(json.dumps(self._meta, indent=2))

            self._recent_events.clear()
            self.active = True
            # print(f"[ExperimentLogger] Started experiment: {self._csv_path.name}")
            # print(f"[ExperimentLogger] Events file: {self._events_path.name}")

    def _make_event_row(self, action: str) -> Dict[str, Any]:
        now_ams = datetime.now(AMS_TZ).isoformat(timespec="seconds")
        return {
            "timestamp_ams": now_ams,
            "timestamp_unix_s": time.time(),
            "event": action,
        }

    def log_event(self, action: str) -> None:
        action = (action or "").strip()
        if not action:
            return

        with self._lock:
            if not self.active or self._events_file is None or self._events_writer is None:
                return

            evt = self._make_event_row(action)

            # dashboard buffer (so UI can show immediately)
            self._recent_events.append(evt)

            # write to events CSV
            self._events_writer.writerow(evt)
            self._events_file.flush()

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            evts = list(self._recent_events)
        return evts[-limit:]

    def log_row(self, row: Dict[str, Any]) -> None:
        with self._lock:
            if not self.active or self._csv_file is None:
                return

            row = dict(row)

            if self._writer is None:
                # header from first sensor row
                self._fieldnames = sorted(row.keys())
                self._writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
                self._writer.writeheader()

            row_to_write = {k: row.get(k, "") for k in self._fieldnames}
            self._writer.writerow(row_to_write)
            self._csv_file.flush()

    def stop_experiment(self) -> None:
        with self._lock:
            if not self.active:
                return

            # close sensor CSV
            if self._csv_file is not None:
                self._csv_file.close()
            self._csv_file = None
            self._writer = None
            self._fieldnames = None

            # close events CSV
            if self._events_file is not None:
                self._events_file.close()
            self._events_file = None
            self._events_writer = None

            self.active = False
            # print("[ExperimentLogger] Stopped experiment")
