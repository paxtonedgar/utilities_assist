from __future__ import annotations

"""Typer CLI entry point for taxonomy term mining pipeline."""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from src.infra.resource_manager import initialize_resources
from src.infra.settings import get_settings

from .build_gazetteer import build_gazetteer
from .config import TaxonomyConfig, load_config
from .extract_candidates import generate_candidates
from .schemas import BuildManifest, LabelPrototype, ScoredTerm, TermCandidate
from .score_contrastive import score_candidates

app = typer.Typer(help="Taxonomy term mining commands")
logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, items) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, default=str) + "\n")


def _terms_summary(
    scored_terms: list[ScoredTerm],
    config: TaxonomyConfig,
    model_name: str,
) -> dict:
    labels = {}
    by_label: dict[str, list[ScoredTerm]] = {}
    for term in scored_terms:
        by_label.setdefault(term.class_id, []).append(term)

    for label, terms in sorted(by_label.items()):
        entries = []
        for term in sorted(
            terms,
            key=lambda t: (
                not t.selected,
                -t.scores.get("contrastive_margin", 0.0),
                -t.scores.get("specificity_margin", 0.0),
                t.surface,
            ),
        ):
            entries.append(
                {
                    "term": term.surface,
                    "selected": term.selected,
                    "assignment": term.assignment,
                    "frequency": term.frequency,
                    "scores": term.scores,
                    "contexts": term.evidence[: config.contexts_export],
                    "doc_ids": term.doc_ids,
                }
            )
        labels[label] = {
            "terms": entries,
            "selected_count": sum(1 for t in terms if t.selected),
            "candidate_count": len(terms),
        }

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": model_name,
        "parameters": config.dict(),
        "labels": labels,
    }


def _build_manifest(
    config: TaxonomyConfig,
    prototypes_count: int,
    candidates_count: int,
    selected_terms: int,
) -> BuildManifest:
    try:
        from subprocess import check_output

        code_sha = check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:  # pragma: no cover - git may not be available
        code_sha = None

    return BuildManifest(
        taxonomy_version=config.taxonomy_version,
        seed=config.random_seed,
        code_sha=code_sha,
        parameters={
            "margin_min": config.margin_min,
            "specificity_min": config.specificity_min,
            "mmr_lambda": config.mmr_lambda,
            "mmr_k": float(config.mmr_k),
        },
        counts={
            "prototypes": prototypes_count,
            "candidates": candidates_count,
            "selected_terms": selected_terms,
        },
    )


def _prepare_resources():
    settings = get_settings()
    return initialize_resources(settings)


@app.command()
def pipeline(
    semantic_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    out: Path = typer.Option(Path("outputs/taxonomy_terms"), file_okay=False),
    config_path: Optional[Path] = typer.Option(None, exists=False),
) -> None:
    """Run extraction, contrastive scoring, and gazetteer build in one pass."""

    config = load_config(str(config_path) if config_path else None)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    _ensure_dir(out)
    resources = _prepare_resources()

    prototypes, candidates = generate_candidates(semantic_dir, config, resources)
    scored_terms, metrics = score_candidates(prototypes, candidates, config)
    gazetteer_entries = build_gazetteer(scored_terms, config)

    _write_json(out / "prototypes.json", [p.dict() for p in prototypes])
    _write_jsonl(out / "candidates.jsonl", [c.dict() for c in candidates])
    _write_jsonl(out / "scored_terms.jsonl", [s.dict() for s in scored_terms])
    _write_json(out / "contrastive_metrics.json", metrics.dict())
    _write_jsonl(out / "gazetteer.jsonl", [g.dict() for g in gazetteer_entries])

    terms_summary = _terms_summary(
        scored_terms,
        config,
        resources.settings.embed.model if resources.settings.embed else "unknown",
    )
    _write_json(out / "terms.json", terms_summary)

    manifest = _build_manifest(
        config,
        len(prototypes),
        len(candidates),
        sum(1 for term in scored_terms if term.selected),
    )
    _write_json(out / "build_manifest.json", manifest.dict())

    logger.info(
        "Pipeline complete: %d prototypes, %d candidates, %d selected terms, %d gazetteer entries",
        len(prototypes),
        len(candidates),
        manifest.counts.get("selected_terms", 0),
        len(gazetteer_entries),
    )


@app.command()
def extract(
    semantic_dir: Path = typer.Option(..., exists=True, file_okay=False),
    out: Path = typer.Option(Path("outputs/taxonomy_terms"), file_okay=False),
    config_path: Optional[Path] = typer.Option(None, exists=False),
) -> None:
    """Only run candidate extraction and prototype building."""

    config = load_config(str(config_path) if config_path else None)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    _ensure_dir(out)
    resources = _prepare_resources()
    prototypes, candidates = generate_candidates(semantic_dir, config, resources)
    _write_json(out / "prototypes.json", [p.dict() for p in prototypes])
    _write_jsonl(out / "candidates.jsonl", [c.dict() for c in candidates])
    logger.info("Extracted %d prototypes and %d candidates", len(prototypes), len(candidates))


@app.command()
def score(
    candidates_path: Path = typer.Option(..., exists=True, dir_okay=False),
    prototypes_path: Path = typer.Option(..., exists=True, dir_okay=False),
    out: Path = typer.Option(Path("outputs/taxonomy_terms"), file_okay=False),
    config_path: Optional[Path] = typer.Option(None, exists=False),
) -> None:
    """Score existing prototypes/candidates and emit selection metrics."""

    config = load_config(str(config_path) if config_path else None)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    _ensure_dir(out)

    prototypes_data = json.loads(prototypes_path.read_text(encoding="utf-8"))
    candidates_data = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    prototypes = [LabelPrototype(**item) for item in prototypes_data]
    candidates = [TermCandidate(**item) for item in candidates_data]

    scored_terms, metrics = score_candidates(prototypes, candidates, config)
    _write_jsonl(out / "scored_terms.jsonl", [s.dict() for s in scored_terms])
    _write_json(out / "contrastive_metrics.json", metrics.dict())
    logger.info("Scored %d candidates", len(scored_terms))


@app.command()
def build(
    scored_path: Path = typer.Option(..., exists=True, dir_okay=False),
    out: Path = typer.Option(Path("outputs/taxonomy_terms"), file_okay=False),
    config_path: Optional[Path] = typer.Option(None, exists=False),
) -> None:
    """Produce gazetteer entries from scored terms."""

    config = load_config(str(config_path) if config_path else None)
    _ensure_dir(out)
    scored_terms = [ScoredTerm(**json.loads(line)) for line in scored_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    entries = build_gazetteer(scored_terms, config)
    _write_jsonl(out / "gazetteer.jsonl", [g.dict() for g in entries])
    logger.info("Wrote %d gazetteer entries", len(entries))


def main():
    logging.basicConfig(level=logging.INFO)
    app()


if __name__ == "__main__":
    main()
