import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


@dataclass
class ExperimentLogger:
    base_dir: Path
    active: bool = field(default=False, init=False)
    _csv_path: Optional[Path] = field(default=None, init=False)
    _csv_file: Optional[Any] = field(default=None, init=False)
    _writer: Optional[csv.DictWriter] = field(default=None, init=False)
    _fieldnames: Optional[List[str]] = field(default=None, init=False)
    _meta: Dict[str, Any] = field(default_factory=dict, init=False)

    def start_experiment(self, meta: Dict[str, Any]) -> None:
        if self.active:
            self.stop_experiment()

        self.base_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = meta.get("name") or meta.get("title") or "experiment"
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

        self._csv_path = self.base_dir / f"{timestamp}_{safe_name}.csv"
        meta_path = self.base_dir / f"{timestamp}_{safe_name}_meta.json"

        self._csv_file = self._csv_path.open("w", newline="")
        self._writer = None
        self._fieldnames = None

        self._meta = dict(meta)
        self._meta["start_time"] = timestamp
        meta_path.write_text(json.dumps(self._meta, indent=2))

        self.active = True
        print(f"[ExperimentLogger] Started experiment: {self._csv_path.name}")

    def log_row(self, row: Dict[str, Any]) -> None:
        if not self.active or self._csv_file is None:
            return

        if self._writer is None:
            self._fieldnames = sorted(row.keys())
            self._writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._writer.writeheader()

        row_to_write = {k: row.get(k, "") for k in self._fieldnames}
        self._writer.writerow(row_to_write)
        self._csv_file.flush()

    def stop_experiment(self) -> None:
        if not self.active:
            return
        if self._csv_file is not None:
            self._csv_file.close()
        self.active = False
        print("[ExperimentLogger] Stopped experiment")
