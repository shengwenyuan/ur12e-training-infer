"""Explicit training backends; numerical dependencies load only on selection."""

from importlib import import_module

BACKENDS = {"pi05_rtc": "training.pi05_rtc"}


def component(name, module):
    """Resolve one backend component without an implicit global registry."""
    return import_module(f"{BACKENDS[name]}.{module}")
