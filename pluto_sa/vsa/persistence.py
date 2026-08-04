"""Versioned, human-readable persistence for VSA patterns and configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


PATTERN_SCHEMA = "pluto-vsa-pattern"
CONFIG_SCHEMA = "pluto-vsa-meas-config"
FORMAT_VERSION = 1
PATTERN_FORMATS = ("Binary", "Decimal", "Hexadecimal")


def _read_document(path: str | Path, schema: str) -> dict[str, Any]:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {source.name}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("VSA file root must be a JSON object")
    if document.get("schema") != schema:
        raise ValueError(f"not a {schema} file")
    if document.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported {schema} version: {document.get('version')!r}")
    return document


def _write_document(path: str | Path, document: Mapping[str, Any]) -> None:
    target = Path(path)
    try:
        target.write_text(
            json.dumps(dict(document), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"could not write {target.name}: {error}") from error


def save_pattern(
    path: str | Path,
    *,
    name: str,
    symbols: list[int] | tuple[int, ...],
    symbol_format: str,
) -> None:
    normalized = [int(symbol) for symbol in symbols]
    if len(normalized) < 4:
        raise ValueError("known pattern must contain at least four symbols")
    if any(symbol < 0 for symbol in normalized):
        raise ValueError("pattern symbols must be non-negative")
    if symbol_format not in PATTERN_FORMATS:
        raise ValueError(f"unsupported symbol format: {symbol_format}")
    _write_document(
        path,
        {
            "schema": PATTERN_SCHEMA,
            "version": FORMAT_VERSION,
            "name": str(name).strip() or "Known Pattern",
            "symbol_format": symbol_format,
            "symbols": normalized,
        },
    )


def load_pattern(path: str | Path) -> dict[str, Any]:
    document = _read_document(path, PATTERN_SCHEMA)
    name = str(document.get("name", "")).strip()
    symbol_format = document.get("symbol_format")
    symbols = document.get("symbols")
    if not name:
        raise ValueError("pattern name must not be empty")
    if symbol_format not in PATTERN_FORMATS:
        raise ValueError(f"unsupported symbol format: {symbol_format!r}")
    if not isinstance(symbols, list) or len(symbols) < 4:
        raise ValueError("pattern symbols must be an array of at least four values")
    if any(not isinstance(symbol, int) or isinstance(symbol, bool) or symbol < 0 for symbol in symbols):
        raise ValueError("pattern symbols must be non-negative integers")
    return {"name": name, "symbol_format": symbol_format, "symbols": symbols}


def save_meas_config(path: str | Path, settings: Mapping[str, Any]) -> None:
    _write_document(
        path,
        {
            "schema": CONFIG_SCHEMA,
            "version": FORMAT_VERSION,
            "settings": dict(settings),
        },
    )


def load_meas_config(path: str | Path) -> dict[str, Any]:
    document = _read_document(path, CONFIG_SCHEMA)
    settings = document.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("measurement configuration settings must be a JSON object")
    return settings
