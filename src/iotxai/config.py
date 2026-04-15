from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def experiment_name(self) -> str:
        return self.raw.get("experiment_name", "iot_xai_repro")

    @property
    def random_state(self) -> int:
        return int(self.raw.get("random_state", 42))

    @property
    def device(self) -> str:
        return str(self.raw.get("device", "auto"))

    @property
    def output_root(self) -> Path:
        return Path(self.raw["outputs"]["root"]).resolve()

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)
