"""
Resolution of the Google Earth Engine project id.

The project id is personal to whoever runs the code, so it is not committed.
It is resolved in this order:

    1. the GEE_PROJECT environment variable
    2. configs/gee_config.json, which is listed in .gitignore and stays local
    3. a placeholder, which is not a usable project id

Importing this module never fails and never raises, so a reader who only wants
to inspect the configuration can import it anywhere. Call require() instead of
resolve() immediately before an Earth Engine call, and it raises with
instructions if no real project has been configured.

This module deliberately imports nothing beyond the standard library, so the
standalone Earth Engine scripts can use it without pulling in torch or numpy.
"""

import json
import os

PLACEHOLDER = "YOUR-GEE-PROJECT"

CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "gee_config.json")

EXAMPLE_FILE = "configs/gee_config.example.json"

_MESSAGE = (
    "No Earth Engine project is configured.\n"
    "Set one in either of these ways:\n"
    "  export GEE_PROJECT=your-project-id\n"
    "  cp %s configs/gee_config.json   and put your project id in it\n"
    "configs/gee_config.json is listed in .gitignore and is never committed."
) % EXAMPLE_FILE


def resolve():
    """Return the configured project id, or PLACEHOLDER if there is none."""
    env = os.environ.get("GEE_PROJECT", "").strip()
    if env:
        return env
    try:
        with open(CONFIG_FILE) as f:
            local = json.load(f)
    except (OSError, ValueError):
        return PLACEHOLDER
    if isinstance(local, dict):
        return str(local.get("project", "")).strip() or PLACEHOLDER
    return PLACEHOLDER


def is_configured():
    """True when a real project id is available."""
    return resolve() != PLACEHOLDER


def require():
    """Return the project id, or explain how to set it."""
    project = resolve()
    if project == PLACEHOLDER:
        raise RuntimeError(_MESSAGE)
    return project
