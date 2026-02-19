"""Diagnostic registry — auto-discovery via ``@register`` decorator.

Usage
-----
In a diagnostic module::

    from feather.diag.registry import register

    @register
    class GlobalBiases(DiagnosticBase):
        name = "global_biases"
        ...

Then anywhere::

    from feather.diag.registry import get_diagnostic, list_diagnostics

    diag_cls = get_diagnostic("global_biases")
    all_diags = list_diagnostics()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from feather.diag.base import DiagnosticBase

_REGISTRY: dict[str, type[DiagnosticBase]] = {}


def register(cls: type[DiagnosticBase]) -> type[DiagnosticBase]:
    """Class decorator that registers a diagnostic.

    The class must have a non-empty ``name`` attribute.

    Parameters
    ----------
    cls : type[DiagnosticBase]
        The diagnostic class to register.

    Returns
    -------
    cls
        The same class, unmodified.
    """
    if not getattr(cls, "name", ""):
        raise ValueError(
            f"Diagnostic class {cls.__name__} must have a non-empty 'name' attribute"
        )
    _REGISTRY[cls.name] = cls
    return cls


def get_diagnostic(name: str) -> type[DiagnosticBase]:
    """Get a registered diagnostic class by name.

    Parameters
    ----------
    name : str
        Diagnostic identifier (e.g. ``"global_biases"``).

    Returns
    -------
    type[DiagnosticBase]

    Raises
    ------
    KeyError
        If no diagnostic is registered with that name.
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"Unknown diagnostic: {name!r}. Available: {available}"
        )
    return _REGISTRY[name]


def list_diagnostics(group: str | None = None) -> list[dict]:
    """List registered diagnostics with metadata.

    Parameters
    ----------
    group : str, optional
        Filter by thematic group (e.g. ``"temperature"``).

    Returns
    -------
    list of dict
        Each dict has keys: ``name``, ``title``, ``domain``,
        ``group``, ``variables``.
    """
    result = []
    for cls in _REGISTRY.values():
        info = {
            "name": cls.name,
            "title": cls.title,
            "domain": cls.domain,
            "group": cls.group,
            "variables": cls.variables,
        }
        if group is None or cls.group == group:
            result.append(info)
    return result


def registered_names() -> list[str]:
    """Return sorted list of all registered diagnostic names."""
    return sorted(_REGISTRY.keys())
