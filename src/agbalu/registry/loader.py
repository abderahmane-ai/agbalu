"""Load and validate the corpus registry from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agbalu.registry.models import Registry, SiblingRegistry


class RegistryError(Exception):
    """The registry file is missing, unparseable, or invalid."""


def _read_document(path: Path) -> dict[str, object]:
    try:
        # utf-8-sig: tolerate a BOM. Sources in this project routinely carry one.
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        msg = f"registry not found: {path}"
        raise RegistryError(msg) from exc
    except UnicodeDecodeError as exc:
        msg = f"registry is not valid UTF-8: {path}"
        raise RegistryError(msg) from exc

    try:
        document: Any = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        msg = f"registry is not valid YAML: {path}"
        raise RegistryError(msg) from exc

    if not isinstance(document, dict):
        kind = type(document).__name__
        msg = f"registry must be a YAML mapping, got {kind}: {path}"
        raise RegistryError(msg)
    mapping: dict[str, object] = document
    return mapping


def load_registry(path: Path) -> Registry:
    """Read `path` and return a validated Kabyle `Registry`.

    Raises:
        RegistryError: the file does not exist, is not UTF-8, is not a YAML
            mapping, or fails schema validation.
    """
    try:
        return Registry.model_validate(_read_document(path))
    except ValidationError as exc:
        msg = f"registry failed validation ({exc.error_count()} error(s)): {path}\n{exc}"
        raise RegistryError(msg) from exc


def load_sibling_registry(path: Path) -> SiblingRegistry:
    """Read `path` and return a validated `SiblingRegistry`.

    Raises:
        RegistryError: as `load_registry`, plus any source that declares Kabyle
            or declares no Berber sibling at all.
    """
    try:
        return SiblingRegistry.model_validate(_read_document(path))
    except ValidationError as exc:
        msg = f"sibling registry failed validation ({exc.error_count()} error(s)): {path}\n{exc}"
        raise RegistryError(msg) from exc
