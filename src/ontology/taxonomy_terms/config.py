from __future__ import annotations

"""Configuration helpers for taxonomy term mining."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class TaxonomyConfig(BaseModel):
    """Pipeline configuration with sensible defaults."""

    random_seed: int = 17

    # Candidate extraction
    min_frequency: int = 2
    max_ngram: int = 3
    contexts_per_term: int = 3
    max_segments_per_doc: int = 40
    max_chars_per_label: int = 6000

    # Scoring thresholds
    margin_min: float = 0.05
    specificity_min: float = 0.08
    parent_similarity_min: float = 0.55
    collision_delta: float = 0.03

    # Selection / diversity
    mmr_k: int = 20
    mmr_lambda: float = 0.7
    dedupe_ratio: float = 92.0  # RapidFuzz token_set_ratio threshold

    # Output options
    contexts_export: int = 3
    taxonomy_version: str = "v1"

    class Config:
        validate_assignment = True


def load_config(path: Optional[str] = None) -> TaxonomyConfig:
    """Load configuration from optional JSON/TOML file; fallback to defaults."""

    if not path:
        return TaxonomyConfig()

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    if cfg_path.suffix in {".json", ""}:
        import json

        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    elif cfg_path.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pyyaml is required to load YAML configs") from exc
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    elif cfg_path.suffix in {".toml"}:
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - py<3.11 guard
            import tomli as tomllib  # type: ignore
        data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config format: {cfg_path.suffix}")

    return TaxonomyConfig(**(data or {}))
