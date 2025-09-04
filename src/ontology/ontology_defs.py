"""
Ontology definitions: entity types and relation types used in the domain graph.
"""

from enum import Enum


class EntityType(str, Enum):
    Team = "Team"
    Division = "Division"
    Application = "Application"
    Tool = "Tool"
    Platform = "Platform"
    Diagram = "Diagram"
    Step = "Step"


class RelType(str, Enum):
    IN_DIVISION = "IN_DIVISION"  # (Team)->(Division)
    OWNS = "OWNS"  # (Team)->(Application|Tool)
    RUNS_ON = "RUNS_ON"  # (Application)->(Platform)
    DOCUMENTED_BY = "DOCUMENTED_BY"  # (Team|Application)->(Diagram)
    HAS_STEP = "HAS_STEP"  # (Process/Doc)->(Step) placeholder, or (Application)->(Step)
    NEXT = "NEXT"
    REQUIRES = "REQUIRES"
    SAME_AS = "SAME_AS"

