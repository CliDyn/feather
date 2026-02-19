"""Unit conversions for climate variables."""


def kelvin_to_celsius(da):
    """Convert Kelvin to Celsius."""
    return da - 273.15


def celsius_to_kelvin(da):
    """Convert Celsius to Kelvin."""
    return da + 273.15


def precip_flux_to_mm_day(da):
    """Convert precipitation flux (kg/m2/s) to mm/day."""
    return da * 86400


def mm_day_to_precip_flux(da):
    """Convert mm/day to precipitation flux (kg/m2/s)."""
    return da / 86400


def pa_to_hpa(da):
    """Convert Pascal to hectoPascal."""
    return da / 100


def hpa_to_pa(da):
    """Convert hectoPascal to Pascal."""
    return da * 100


def flux_sign_convention(da, convention="positive_down"):
    """Flip sign of flux if needed.

    Parameters
    ----------
    da : DataArray
        Flux data.
    convention : str
        Target convention: "positive_down" or "positive_up".
    """
    if convention == "positive_up":
        return -da
    return da


def fraction_to_percent(da):
    """Convert fraction (0-1) to percent (0-100)."""
    return da * 100


def percent_to_fraction(da):
    """Convert percent (0-100) to fraction (0-1)."""
    return da / 100
