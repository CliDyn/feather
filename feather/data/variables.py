"""Variable metadata and model-obs-CMIP6 mapping registry.

Unit conversion notes
---------------------
ERA5 monthly data stores radiation/flux variables as **daily accumulations**
(J/m² per day for radiation, m per day for precipitation).  To convert to
instantaneous rates (W/m² or kg/m²/s) we divide by 86400 (seconds per day).

Sign conventions
~~~~~~~~~~~~~~~~
ERA5 and DestinE (both IFS-based) share the same sign convention:
- Surface heat fluxes: **positive downward** (into surface)
- Net radiation: **positive downward**

CMIP6 uses the opposite convention for several variables:
- ``hfss`` / ``hfls``: **positive upward** (surface → atmosphere)
- ``rsut`` / ``rlut``: **positive upward** (outgoing at TOA)

Variables where CMIP6 has a sign mismatch have ``cmip6_variable=""`` to
prevent incorrect comparisons until sign-flip support is added.

When comparing CMOR-convention model data (EERIE, CMIP6) against ERA5 obs,
variables with ``cmor_obs_sign=-1.0`` need their obs values negated to match
the CMOR sign convention (positive upward for surface fluxes).

Naming convention
~~~~~~~~~~~~~~~~~
Registry keys are CMOR-style canonical names (e.g., ``"tas"``, ``"pr"``).
The ``destine_variable`` field stores the DestinE model variable name
(e.g., ``"avg_2t"``).  ``get_var()`` accepts both CMOR and DestinE names
for backward compatibility.
"""

from dataclasses import dataclass

# ERA5 accumulated → instantaneous rate conversion factor
_ACCUM_FACTOR = 1.0 / 86400.0   # J/m²/day → W/m²  (or m/day → m/s)


@dataclass(frozen=True)
class VarInfo:
    """Metadata for a single model variable."""

    name: str            # Canonical variable name (CMOR-style, e.g., "tas")
    long_name: str       # Human-readable name
    units: str           # Physical units (model convention)
    domain: str          # "sfc", "o2d", "pl", "o3d"
    cmap: str            # Default colormap for bias plots
    obs_dataset: str     # Observation dataset name (e.g., "ERA5")
    obs_variable: str    # Variable name in obs dataset config
    destine_variable: str = ""   # DestinE model variable name (e.g., "avg_2t")
    cmip6_variable: str = ""
    cmip6_table: str = ""
    obs_unit_factor: float = 1.0   # Multiply obs by this to match model units
    obs_unit_offset: float = 0.0   # Add after multiplying
    cmor_obs_sign: float = 1.0    # Sign flip for obs when comparing vs CMOR data
    group: str = ""
    display_offset: float = 0.0   # Subtract from values at plot time (e.g. -273.15 for K→°C)
    display_units: str = ""        # Display label override (e.g. "°C"); empty → use units


VARIABLE_REGISTRY: dict[str, VarInfo] = {

    # ═══════════════════════════════════════════════════════════════════
    #  Surface atmospheric variables  (domain = "sfc")
    # ═══════════════════════════════════════════════════════════════════

    # --- Temperature & pressure (units match directly) ---------------

    "tas": VarInfo(
        name="tas", long_name="2m Temperature", units="K",
        domain="sfc", cmap="cmo.thermal",
        obs_dataset="ERA5", obs_variable="t2m",
        destine_variable="avg_2t",
        cmip6_variable="tas", cmip6_table="Amon",
        group="temperature",
        display_offset=-273.15, display_units="°C",
    ),
    "ts": VarInfo(
        name="ts", long_name="Skin Temperature", units="K",
        domain="sfc", cmap="cmo.thermal",
        obs_dataset="ERA5", obs_variable="sst",
        destine_variable="avg_skt",
        # NOTE: ERA5 sst is SST (ocean-only); skt is global skin temp.
        # Comparison is only meaningful over ocean.
        cmip6_variable="ts", cmip6_table="Amon",
        group="temperature",
        display_offset=-273.15, display_units="°C",
    ),
    "psl": VarInfo(
        name="psl", long_name="Mean Sea Level Pressure", units="Pa",
        domain="sfc", cmap="viridis",
        obs_dataset="ERA5", obs_variable="msl",
        destine_variable="avg_msl",
        cmip6_variable="psl", cmip6_table="Amon",
        group="circulation",
    ),

    # --- Wind (units match directly) ---------------------------------

    "uas": VarInfo(
        name="uas", long_name="10m U Wind", units="m/s",
        domain="sfc", cmap="coolwarm",
        obs_dataset="ERA5", obs_variable="u10",
        destine_variable="avg_10u",
        cmip6_variable="uas", cmip6_table="Amon",
        group="wind",
    ),
    "vas": VarInfo(
        name="vas", long_name="10m V Wind", units="m/s",
        domain="sfc", cmap="coolwarm",
        obs_dataset="ERA5", obs_variable="v10",
        destine_variable="avg_10v",
        cmip6_variable="vas", cmip6_table="Amon",
        group="wind",
    ),
    "sfcWind": VarInfo(
        name="sfcWind", long_name="10m Wind Speed", units="m/s",
        domain="sfc", cmap="YlOrRd",
        obs_dataset="ERA5", obs_variable="u10",
        destine_variable="avg_10ws",
        # NOTE: Derived variable — needs sqrt(u10² + v10²) from ERA5.
        # Not directly comparable via simple load; excluded from
        # GlobalBiases until derived-variable support is added.
        group="wind",
    ),

    # --- Cloud cover -------------------------------------------------

    "clt": VarInfo(
        name="clt", long_name="Total Cloud Cover", units="%",
        domain="sfc", cmap="Greys_r",
        obs_dataset="ERA5", obs_variable="tcc",
        destine_variable="avg_tcc",
        obs_unit_factor=100.0,  # ERA5 is 0-1 fraction → model is 0-100%
        cmip6_variable="clt", cmip6_table="Amon",
        group="clouds",
    ),

    # --- Moisture / column quantities --------------------------------

    "prw": VarInfo(
        name="prw", long_name="Total Column Water Vapour", units="kg/m2",
        domain="sfc", cmap="YlGnBu",
        obs_dataset="ERA5", obs_variable="tcwv",
        destine_variable="avg_tcwv",
        # WARNING: No ERA5 TCWV file in the current collection.
        # Config key 'tcwv' was wrongly mapped to cloud liquid water.
        cmip6_variable="prw", cmip6_table="Amon",
        group="moisture",
    ),
    "clwvi": VarInfo(
        name="clwvi", long_name="Total Column Cloud Liquid Water",
        units="kg/m2", domain="sfc", cmap="YlGnBu",
        obs_dataset="ERA5", obs_variable="tclw",
        destine_variable="avg_tclw",
        group="clouds",
    ),
    "clivi": VarInfo(
        name="clivi", long_name="Total Column Cloud Ice Water",
        units="kg/m2", domain="sfc", cmap="YlGnBu",
        obs_dataset="ERA5", obs_variable="tciw",
        destine_variable="avg_tciw",
        group="clouds",
    ),

    # --- Precipitation -----------------------------------------------

    "pr": VarInfo(
        name="pr", long_name="Total Precipitation Rate",
        units="kg/m2/s", domain="sfc", cmap="BrBG",
        obs_dataset="ERA5", obs_variable="tp",
        destine_variable="avg_tprate",
        obs_unit_factor=1000.0 * _ACCUM_FACTOR,  # m/day → kg/m²/s
        cmip6_variable="pr", cmip6_table="Amon",
        group="precipitation",
    ),

    # --- Surface heat fluxes (positive downward, same as ERA5) -------
    # CMIP6 hfss/hfls are positive UPWARD → sign mismatch → no mapping.

    "hfss": VarInfo(
        name="hfss", long_name="Surface Sensible Heat Flux",
        units="W/m2", domain="sfc", cmap="coolwarm",
        obs_dataset="ERA5", obs_variable="sshf",
        destine_variable="avg_ishf",
        obs_unit_factor=_ACCUM_FACTOR,  # J/m²/day → W/m²
        cmor_obs_sign=-1.0,  # ERA5 positive downward → CMOR positive upward
        # CMIP6 hfss is positive upward — sign mismatch with DestinE, omitted
        group="surface_fluxes",
    ),
    "hfls": VarInfo(
        name="hfls", long_name="Surface Latent Heat Flux",
        units="W/m2", domain="sfc", cmap="coolwarm",
        obs_dataset="ERA5", obs_variable="slhf",
        destine_variable="avg_slhtf",
        obs_unit_factor=_ACCUM_FACTOR,  # J/m²/day → W/m²
        cmor_obs_sign=-1.0,  # ERA5 positive downward → CMOR positive upward
        # CMIP6 hfls is positive upward — sign mismatch with DestinE, omitted
        group="surface_fluxes",
    ),

    # --- Surface downwelling radiation (positive downward) -----------

    "rsds": VarInfo(
        name="rsds", long_name="Surface Downwelling Shortwave",
        units="W/m2", domain="sfc", cmap="YlOrRd",
        obs_dataset="ERA5", obs_variable="ssrd",
        destine_variable="avg_sdswrf",
        obs_unit_factor=_ACCUM_FACTOR,  # J/m²/day → W/m²
        cmip6_variable="rsds", cmip6_table="Amon",
        group="radiation",
    ),
    "rlds": VarInfo(
        name="rlds", long_name="Surface Downwelling Longwave",
        units="W/m2", domain="sfc", cmap="YlOrRd",
        obs_dataset="ERA5", obs_variable="strd",
        destine_variable="avg_sdlwrf",
        obs_unit_factor=_ACCUM_FACTOR,  # J/m²/day → W/m²
        cmip6_variable="rlds", cmip6_table="Amon",
        group="radiation",
    ),

    # --- Surface net radiation (positive downward) -------------------
    # No single CMIP6 variable for net surface radiation.

    "rss": VarInfo(
        name="rss", long_name="Surface Net Shortwave Radiation",
        units="W/m2", domain="sfc", cmap="coolwarm",
        obs_dataset="ERA5", obs_variable="ssr",
        destine_variable="avg_snswrf",
        obs_unit_factor=_ACCUM_FACTOR,  # J/m²/day → W/m²
        group="radiation",
    ),
    "rls": VarInfo(
        name="rls", long_name="Surface Net Longwave Radiation",
        units="W/m2", domain="sfc", cmap="coolwarm",
        obs_dataset="ERA5", obs_variable="str",
        destine_variable="avg_snlwrf",
        obs_unit_factor=_ACCUM_FACTOR,  # J/m²/day → W/m²
        group="radiation",
    ),
    "rsscs": VarInfo(
        name="rsscs",
        long_name="Surface Net Shortwave Radiation (Clear-Sky)",
        units="W/m2", domain="sfc", cmap="coolwarm",
        obs_dataset="ERA5", obs_variable="ssrc",
        destine_variable="avg_snswrfcs",
        obs_unit_factor=_ACCUM_FACTOR,
        group="radiation",
    ),
    "rlscs": VarInfo(
        name="rlscs",
        long_name="Surface Net Longwave Radiation (Clear-Sky)",
        units="W/m2", domain="sfc", cmap="coolwarm",
        obs_dataset="ERA5", obs_variable="strc",
        destine_variable="avg_snlwrfcs",
        obs_unit_factor=_ACCUM_FACTOR,
        group="radiation",
    ),

    # --- TOA net radiation (positive downward) -----------------------
    # CMIP6 rsut/rlut are outgoing components only (positive upward),
    # not net fluxes → sign mismatch → no mapping.

    "rst": VarInfo(
        name="rst", long_name="TOA Net Shortwave Radiation",
        units="W/m2", domain="sfc", cmap="coolwarm",
        obs_dataset="ERA5", obs_variable="tsr",
        destine_variable="avg_tnswrf",
        obs_unit_factor=_ACCUM_FACTOR,  # J/m²/day → W/m²
        # CMIP6 rsut is outgoing SW only — not net; omitted
        group="radiation",
    ),
    "rlt": VarInfo(
        name="rlt", long_name="TOA Net Longwave Radiation",
        units="W/m2", domain="sfc", cmap="coolwarm",
        obs_dataset="ERA5", obs_variable="ttr",
        destine_variable="avg_tnlwrf",
        obs_unit_factor=_ACCUM_FACTOR,  # J/m²/day → W/m²
        # CMIP6 rlut is outgoing LW only — not net; omitted
        group="radiation",
    ),
    "rstcs": VarInfo(
        name="rstcs",
        long_name="TOA Net Shortwave Radiation (Clear-Sky)",
        units="W/m2", domain="sfc", cmap="coolwarm",
        obs_dataset="ERA5", obs_variable="tsrc",
        destine_variable="avg_tnswrfcs",
        obs_unit_factor=_ACCUM_FACTOR,
        group="radiation",
    ),
    "rltcs": VarInfo(
        name="rltcs",
        long_name="TOA Net Longwave Radiation (Clear-Sky)",
        units="W/m2", domain="sfc", cmap="coolwarm",
        obs_dataset="ERA5", obs_variable="ttrc",
        destine_variable="avg_tnlwrfcs",
        obs_unit_factor=_ACCUM_FACTOR,
        group="radiation",
    ),

    # ═══════════════════════════════════════════════════════════════════
    #  Ocean 2D variables  (domain = "o2d")
    # ═══════════════════════════════════════════════════════════════════

    "tos": VarInfo(
        name="tos", long_name="Sea Surface Temperature", units="K",
        domain="o2d", cmap="cmo.thermal",
        obs_dataset="ESA_CCI", obs_variable="analysed_sst",
        destine_variable="avg_tos",
        cmip6_variable="tos", cmip6_table="Omon",
        group="ocean_surface",
    ),
    "siconc": VarInfo(
        name="siconc", long_name="Sea Ice Concentration", units="0-1",
        domain="o2d", cmap="Blues_r",
        obs_dataset="OSI_SAF", obs_variable="ice_conc",
        destine_variable="avg_siconc",
        cmip6_variable="siconc", cmip6_table="SImon",
        obs_unit_factor=0.01,  # OSI-SAF is 0-100%, model is 0-1
        group="sea_ice",
    ),
    "sithick": VarInfo(
        name="sithick", long_name="Sea Ice Thickness", units="m",
        domain="o2d", cmap="Blues",
        obs_dataset="PSC", obs_variable="hi",
        destine_variable="avg_sithick",
        cmip6_variable="sithick", cmip6_table="SImon",
        group="sea_ice",
    ),
    "zos": VarInfo(
        name="zos", long_name="Sea Surface Height", units="m",
        domain="o2d", cmap="coolwarm",
        obs_dataset="AVISO", obs_variable="adt",
        destine_variable="avg_zos",
        cmip6_variable="zos", cmip6_table="Omon",
        group="ocean_surface",
    ),
    "sos": VarInfo(
        name="sos", long_name="Sea Surface Salinity", units="PSU",
        domain="o2d", cmap="PRGn_r",
        obs_dataset="EN4", obs_variable="so",
        destine_variable="avg_sos",
        cmip6_variable="sos", cmip6_table="Omon",
        group="ocean_surface",
    ),

    # ═══════════════════════════════════════════════════════════════════
    #  Ocean 3D variables  (domain = "o3d")
    # ═══════════════════════════════════════════════════════════════════

    "thetao": VarInfo(
        name="thetao", long_name="Ocean Temperature", units="K",
        domain="o3d", cmap="cmo.thermal",
        obs_dataset="EN4", obs_variable="thetao",
        destine_variable="avg_thetao",
        group="ocean_3d",
    ),
    "so": VarInfo(
        name="so", long_name="Ocean Salinity", units="PSU",
        domain="o3d", cmap="PRGn_r",
        obs_dataset="EN4", obs_variable="so",
        destine_variable="avg_so",
        group="ocean_3d",
    ),

    # ═══════════════════════════════════════════════════════════════════
    #  Pressure level variables  (domain = "pl")
    # ═══════════════════════════════════════════════════════════════════

    "ta": VarInfo(
        name="ta", long_name="Temperature (pressure levels)", units="K",
        domain="pl", cmap="cmo.thermal",
        obs_dataset="ERA5", obs_variable="t2m",  # placeholder
        destine_variable="avg_t",
        group="atmosphere_3d",
    ),
    "ua": VarInfo(
        name="ua", long_name="Zonal Wind (pressure levels)", units="m/s",
        domain="pl", cmap="coolwarm",
        obs_dataset="ERA5", obs_variable="u10",  # placeholder
        destine_variable="avg_u",
        group="atmosphere_3d",
    ),
    "va": VarInfo(
        name="va", long_name="Meridional Wind (pressure levels)",
        units="m/s", domain="pl", cmap="coolwarm",
        obs_dataset="ERA5", obs_variable="v10",  # placeholder
        destine_variable="avg_v",
        group="atmosphere_3d",
    ),
    "hus": VarInfo(
        name="hus", long_name="Specific Humidity (pressure levels)",
        units="kg/kg", domain="pl", cmap="YlGnBu",
        obs_dataset="ERA5", obs_variable="tcwv",  # placeholder
        destine_variable="avg_q",
        group="atmosphere_3d",
    ),

    # ═══════════════════════════════════════════════════════════════════
    #  Climate extremes indices  (domain = "sfc", daily inputs)
    # ═══════════════════════════════════════════════════════════════════

    "tasmin": VarInfo(
        name="tasmin", long_name="Daily Minimum 2m Temperature", units="K",
        domain="sfc", cmap="cmo.thermal",
        obs_dataset="ERA5",   # ERA5 daily tasmin not available at standard path
        obs_variable="mn2t",
        destine_variable="",
        cmip6_variable="tasmin", cmip6_table="day",
        group="extremes",
        display_offset=-273.15, display_units="°C",
    ),
}

# Reverse mapping: DestinE variable name → canonical (CMOR) name
_DESTINE_TO_CANONICAL: dict[str, str] = {
    v.destine_variable: k
    for k, v in VARIABLE_REGISTRY.items()
    if v.destine_variable
}


def get_var(name: str) -> VarInfo:
    """Look up variable metadata by canonical (CMOR) or DestinE name."""
    if name in VARIABLE_REGISTRY:
        return VARIABLE_REGISTRY[name]
    # Fallback: try DestinE name → canonical lookup
    canonical = _DESTINE_TO_CANONICAL.get(name)
    if canonical is not None:
        return VARIABLE_REGISTRY[canonical]
    raise KeyError(f"Unknown variable: {name!r}. "
                   f"Available: {list(VARIABLE_REGISTRY.keys())}")


def canonical_name(name: str) -> str:
    """Convert any variable name (CMOR or DestinE) to canonical CMOR name.

    Returns the name unchanged if it's already canonical.
    """
    if name in VARIABLE_REGISTRY:
        return name
    canonical = _DESTINE_TO_CANONICAL.get(name)
    if canonical is not None:
        return canonical
    raise KeyError(f"Unknown variable: {name!r}")


def destine_name(name: str) -> str:
    """Get the DestinE variable name for a canonical (CMOR) name.

    Returns the DestinE name (e.g., ``"avg_2t"`` for ``"tas"``).
    Raises KeyError if no DestinE mapping exists.
    """
    vinfo = get_var(name)
    if vinfo.destine_variable:
        return vinfo.destine_variable
    raise KeyError(f"No DestinE variable name for {name!r}")


def list_vars(domain: str = None, group: str = None) -> list[VarInfo]:
    """List variables, optionally filtered by domain or group."""
    result = list(VARIABLE_REGISTRY.values())
    if domain is not None:
        result = [v for v in result if v.domain == domain]
    if group is not None:
        result = [v for v in result if v.group == group]
    return result
