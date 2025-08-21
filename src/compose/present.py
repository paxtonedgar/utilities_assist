# src/compose/present.py
"""
Multi-suite presenters for structured output formatting.
Renders procedure, info, and fallback presentations with suite-aware CTAs.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def render(material: Dict[str, Any]) -> str:
    """
    Render structured material into final presentation format.

    Args:
        material: Materialized content from actionability module

    Returns:
        Formatted string for display
    """
    presentation_type = material.get("type", "fallback")

    if presentation_type == "procedure":
        return _render_procedure(material)
    elif presentation_type == "info":
        return _render_info(material)
    else:
        return _render_fallback(material)


def _render_procedure(material: Dict[str, Any]) -> str:
    """Render procedure presenter with multi-suite actions."""
    lines = []

    # Title
    lines.append("## Procedure\n")

    # Prerequisites section
    if material.get("prerequisites"):
        lines.append("### Prerequisites")
        for prereq in material["prerequisites"]:
            lines.append(f"• {prereq['text']} {prereq['citation']}")
        lines.append("")

    # Steps section
    if material.get("steps"):
        lines.append("### Steps")
        for i, step in enumerate(material["steps"], 1):
            lines.append(f"{i}. {step['text']} {step['citation']}")
        lines.append("")

    # Suite-specific actions
    suite_actions = material.get("suite_actions", {})
    if suite_actions:
        lines.append("### Actions")

        for suite, actions in suite_actions.items():
            # Suite header with proper capitalization
            suite_name = _format_suite_name(suite)
            lines.append(f"**{suite_name}:**")

            for action in actions:
                cta = action.get("cta", "Open link")
                attrs_text = _format_attributes(action.get("attrs", {}))
                attrs_suffix = f" — {attrs_text}" if attrs_text else ""

                lines.append(
                    f"• {cta}{attrs_suffix} [Open]({action['url']}) {action['citation']}"
                )

            lines.append("")

    # Metrics footer for debugging
    if material.get("metrics"):
        metrics = material["metrics"]
        score = metrics.get("actionable_score", 0)
        suite_counts = metrics.get("suite_counts", {})

        suite_summary = ", ".join(f"{k}={v}" for k, v in suite_counts.items()) or "none"
        lines.append(f"*Actionability: {score:.1f} | Suites: {suite_summary}*")

    return "\n".join(lines)


def _render_info(material: Dict[str, Any]) -> str:
    """Render info presenter with cited facts."""
    lines = []

    # Definition section
    definition = material.get("definition", {})
    if definition and definition.get("text"):
        lines.append("## Definition")
        lines.append(f"{definition['text']} {definition['citation']}")
        lines.append("")

    # Key facts section
    key_facts = material.get("key_facts", [])
    if key_facts:
        lines.append("## Key Information")
        for fact in key_facts:
            lines.append(f"• {fact['text']} {fact['citation']}")
        lines.append("")

    # Metrics footer for debugging
    if material.get("metrics"):
        metrics = material["metrics"]
        score = metrics.get("actionable_score", 0)
        suite_counts = metrics.get("suite_counts", {})

        suite_summary = ", ".join(f"{k}={v}" for k, v in suite_counts.items()) or "none"
        lines.append(
            f"*Info confidence: high | Actionability: {score:.1f} | Detected: {suite_summary}*"
        )

    return "\n".join(lines)


def _render_fallback(material: Dict[str, Any]) -> str:
    """Render fallback presenter with closest guidance."""
    lines = []

    # Banner message
    banner = material.get("banner", "Limited guidance available.")
    lines.append(f"⚠️ **{banner}**\n")

    # Available paragraphs
    paragraphs = material.get("paragraphs", [])
    if paragraphs:
        lines.append("### Closest Guidance")
        for para in paragraphs:
            lines.append(f"{para['text']} {para['citation']}")
            lines.append("")

    # Detected spans info
    detected_spans = material.get("detected_spans", 0)
    if detected_spans > 0:
        lines.append(
            f"*Detected {detected_spans} actionable elements but insufficient for structured guidance.*"
        )

    # Metrics footer
    if material.get("metrics"):
        metrics = material["metrics"]
        score = metrics.get("actionable_score", 0)
        suite_counts = metrics.get("suite_counts", {})

        suite_summary = ", ".join(f"{k}={v}" for k, v in suite_counts.items()) or "none"
        lines.append(f"*Actionability: {score:.1f} | Detected: {suite_summary}*")

    return "\n".join(lines)


def _format_suite_name(suite: str) -> str:
    """Format suite name for display."""
    suite_display_names = {
        "jira": "Jira",
        "servicenow": "ServiceNow",
        "api": "API",
        "teams": "Microsoft Teams",
        "outlook": "Outlook",
        "github": "GitHub",
        "global": "General",
    }

    return suite_display_names.get(suite.lower(), suite.title())


def _format_attributes(attrs: Dict[str, str]) -> str:
    """Format span attributes for display."""
    if not attrs:
        return ""

    # Format key attributes in readable way
    formatted_parts = []

    for key, value in attrs.items():
        if key == "project_key":
            formatted_parts.append(f"Project: {value}")
        elif key == "issue_type":
            formatted_parts.append(f"Type: {value}")
        elif key == "method":
            formatted_parts.append(f"{value}")
        elif key == "channel_name":
            formatted_parts.append(f"Channel: {value}")
        elif key == "team_name":
            formatted_parts.append(f"Team: {value}")
        elif key == "dl_name":
            formatted_parts.append(f"DL: {value}")
        elif key == "service_name":
            formatted_parts.append(f"Service: {value}")
        elif key in ["ticket_id", "incident_number"]:
            formatted_parts.append(f"#{value}")
        else:
            # Generic fallback
            formatted_parts.append(f"{key}: {value}")

    return ", ".join(formatted_parts)
