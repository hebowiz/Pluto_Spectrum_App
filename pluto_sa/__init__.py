"""Compatibility imports for the former combined RTSA/VSA package.

Application code lives in pluto_rtsa, pluto_vsa and pluto_common. Aliases load
the canonical module once, preserving Enum, dataclass and singleton identity.
New application code must not import this compatibility namespace.
"""

from __future__ import annotations

import importlib
from importlib.abc import Loader, MetaPathFinder
from importlib.util import find_spec, spec_from_loader
import sys


_MODULE_MOVES = (
    ("pluto_sa.vsa", "pluto_vsa"),
    ("pluto_sa.standards", "pluto_vsa.standards"),
    ("pluto_sa.sdr", "pluto_common.sdr"),
    ("pluto_sa.config.input_frontend", "pluto_common.config.input_frontend"),
    ("pluto_sa.config.spectrum_config", "pluto_common.config.spectrum_config"),
    ("pluto_sa.modes.analyzer_mode", "pluto_common.config.analyzer_mode"),
    ("pluto_sa", "pluto_rtsa"),
)


class _LegacyModuleLoader(Loader):
    def __init__(self, target: str) -> None:
        self.target = target

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self._canonical_spec = module.__spec__
        return module

    def exec_module(self, module) -> None:
        # Import machinery installs the alias spec even for an existing module.
        # Keep introspection, reload and pickling tied to the canonical name.
        module.__spec__ = self._canonical_spec

    def get_code(self, fullname):
        # Preserve the old `python -m ...` commands without re-executing the
        # implementation under a second module name.
        source = (
            "from importlib import import_module\n"
            f"raise SystemExit(import_module({self.target!r}).main())\n"
        )
        return compile(source, f"<legacy entry point {fullname}>", "exec")


class _LegacyModuleFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("pluto_sa."):
            return None
        for old, new in _MODULE_MOVES:
            if fullname == old or fullname.startswith(old + "."):
                canonical_name = new + fullname[len(old):]
                canonical_spec = find_spec(canonical_name)
                if canonical_spec is None:
                    return None
                return spec_from_loader(
                    fullname,
                    _LegacyModuleLoader(canonical_name),
                    origin=canonical_spec.origin,
                    is_package=canonical_spec.submodule_search_locations is not None,
                )
        return None


sys.meta_path.insert(0, _LegacyModuleFinder())
