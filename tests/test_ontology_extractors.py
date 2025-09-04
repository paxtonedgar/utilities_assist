import pytest

from src.ontology.extractors import regex_step_extractor


def test_regex_step_extractor_bullets():
    text = """
    1) Install the CLI tools
    2) Configure the CI pipeline
    3) Verify the deployment
    """
    steps = regex_step_extractor(text)
    assert len(steps) == 3
    assert steps[0]["order"] == 1
    assert steps[0]["verb"] == "install"
    assert "cli tools" in steps[0]["label"].lower()
    assert steps[1]["order"] == 2
    assert steps[1]["verb"] == "configure"
    assert steps[2]["order"] == 3
    assert steps[2]["verb"] == "verify"


def test_regex_step_extractor_imperatives():
    text = "Configure the CI pipeline. Then verify the results."
    steps = regex_step_extractor(text)
    assert len(steps) >= 1
    assert steps[0]["verb"] == "configure"

