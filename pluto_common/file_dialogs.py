"""Persistent, per-file-kind folders for native file dialogs."""

from pathlib import Path


def file_dialog_path(
    settings,
    directory_key: str,
    *,
    filename: str = "",
    default_directory: str | Path | None = None,
) -> str:
    """Return an explicit existing folder, optionally with a suggested filename.

    An empty directory lets native dialogs reuse another menu's history, so
    always fall back to an existing application default or the working folder.
    """
    stored = settings.value(directory_key, "")
    if not isinstance(stored, (str, Path)):
        stored = ""
    directory = Path.cwd()
    for candidate in (stored, default_directory):
        if candidate and Path(candidate).is_dir():
            directory = Path(candidate).resolve()
            break
    return str(directory / filename) if filename else str(directory)


def remember_file_directory(settings, directory_key: str, path: str | Path) -> None:
    """Persist the selected file's parent; cancellation leaves history intact."""
    if not path:
        return
    settings.setValue(directory_key, str(Path(path).resolve().parent))
    settings.sync()
