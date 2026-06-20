"""Shared loading of stellarator benchmark configurations.

Each configuration is a subdirectory of ``benchmarks/configurations/<name>/``
containing two files:

- ``eq.h5``    — DESC equilibrium (required)
- ``coils.h5`` — DESC coil / external field (optional; error if absent unless
  ``--allow-missing-coils`` is passed)

Select a configuration with ``--config <name>`` in the benchmark and submit
scripts. Explicit ``--eq-file`` / ``--coil-file`` flags override the config.
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "configurations"


def available_configs() -> list[str]:
    """Return the names of configurations that have at least an eq.h5."""
    if not CONFIG_DIR.exists():
        return []
    return sorted(
        d.name for d in CONFIG_DIR.iterdir()
        if d.is_dir() and (d / "eq.h5").exists()
    )


def load_config(name: str) -> dict:
    """Return ``{"name", "eq_file", "coil_file"}`` for the named configuration.

    Raises ``SystemExit`` if the directory or ``eq.h5`` is missing so the error
    message guides the user to copy the files.
    """
    config_dir = CONFIG_DIR / name
    if not config_dir.is_dir():
        options = ", ".join(available_configs()) or "(none — copy eq.h5 into a subdirectory of benchmarks/configurations/)"
        raise SystemExit(
            f"Unknown benchmark configuration {name!r}.\n"
            f"Available: {options}\n"
            f"To add {name!r}, create benchmarks/configurations/{name}/ and copy "
            f"the equilibrium as eq.h5 and the coil field as coils.h5."
        )
    eq = config_dir / "eq.h5"
    if not eq.exists():
        raise SystemExit(
            f"Configuration {name!r} exists but is missing eq.h5.\n"
            f"Copy the DESC equilibrium .h5 to {eq}"
        )
    coils = config_dir / "coils.h5"
    return {
        "name": name,
        "eq_file": eq,
        "coil_file": coils if coils.exists() else None,
    }


def config_overrides(config: dict) -> dict:
    """Map ``eq_file`` / ``coil_file`` to argparse defaults (None skipped)."""
    overrides = {}
    if config.get("eq_file") is not None:
        overrides["eq_file"] = Path(config["eq_file"])
    if config.get("coil_file") is not None:
        overrides["coil_file"] = Path(config["coil_file"])
    return overrides
