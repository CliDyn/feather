"""Cross-version helper for requesting cftime time decoding from xarray.

Newer xarray (>= 2025) deprecates the ``use_cftime=True`` kwarg in favour of
passing a ``CFDatetimeCoder`` to ``decode_times``.  This helper returns the
right kwargs for the installed version so loaders work either way.
"""


def cftime_decode_kwargs() -> dict:
    """Return open_*-kwargs that decode time coordinates as cftime objects."""
    try:
        from xarray.coders import CFDatetimeCoder
        return {"decode_times": CFDatetimeCoder(use_cftime=True)}
    except Exception:  # older xarray without xarray.coders
        return {"use_cftime": True}
