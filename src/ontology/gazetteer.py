from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Set, Tuple, List


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\u200b\u200c\u200d]+", "", s)  # zero-width
    return s


class Gazetteer:
    def __init__(self, base: str | Path = "data/gazetteer"):
        base = Path(base)
        self.base = base
        self.teams: Set[str] = set()
        self.apps: Set[str] = set()
        self.platforms: Set[str] = set()
        self.divisions: Set[str] = set()
        self.synonyms: Dict[str, str] = {}
        self._load()

    def _load_list(self, name: str) -> Set[str]:
        p = self.base / f"{name}.txt"
        if not p.exists():
            return set()
        items = set()
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            items.add(_norm(ln))
        return items

    def _load_synonyms(self) -> Dict[str, str]:
        p = self.base / "synonyms.json"
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        # flatten type-specific maps into global normalized alias->canonical
        flat: Dict[str, str] = {}
        for _type, mapping in data.items():
            for alias, canon in mapping.items():
                flat[_norm(alias)] = _norm(canon)
        return flat

    def _load(self) -> None:
        self.teams = self._load_list("teams")
        self.apps = self._load_list("applications")
        self.platforms = self._load_list("platforms")
        self.divisions = self._load_list("divisions")
        self.synonyms = self._load_synonyms()

    def canonical(self, name: str) -> str | None:
        key = _norm(name)
        if key in self.synonyms:
            key = self.synonyms[key]
        if (key in self.teams) or (key in self.apps) or (key in self.platforms) or (key in self.divisions):
            return key
        return None

    def find_mentions(self, text: str) -> List[Tuple[str, str]]:
        """Return list of (type, canonical_name) mentions found in text.

        Simple case-insensitive exact phrase search over gazetteer entries and synonyms.
        """
        mentions: List[Tuple[str, str]] = []
        blob = text or ""
        blob_lc = blob.lower()

        def scan(items: Set[str], _type: str):
            for term in items:
                if term and term in blob_lc:
                    mentions.append((_type, term))

        scan(self.teams, "Team")
        scan(self.apps, "Application")
        scan(self.platforms, "Platform")
        scan(self.divisions, "Division")

        # include synonyms mapping to canonical if aliases appear
        for alias, canon in self.synonyms.items():
            if alias in blob_lc:
                # decide type by membership of canonical
                if canon in self.teams:
                    mentions.append(("Team", canon))
                elif canon in self.apps:
                    mentions.append(("Application", canon))
                elif canon in self.platforms:
                    mentions.append(("Platform", canon))
                elif canon in self.divisions:
                    mentions.append(("Division", canon))
        # dedup
        seen = set()
        out: List[Tuple[str, str]] = []
        for t, n in mentions:
            k = (t, n)
            if k not in seen:
                seen.add(k)
                out.append((t, n))
        return out

