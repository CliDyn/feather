"""Variable metadata and model-obs-CMIP6 mapping registry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VarInfo:
    """Metadata for a single model variable."""

    name: str            # Model variable name (e.g., "avg_2t")
    long_name: str       # Human-readable name
    units: str           # Physical units
    domain: str          # "sfc", "o2d", "pl", "o3d"
    cmap: str            # Default colormap for bias plots
    obs_dataset: str     # Observation dataset name (e.g., "ERA5")
    obs_variable: str    # Variable name in obs dataset
    cmip6_variable: str = ""
    cmip6_table: str = ""
    obs_unit_factor: float = 1.0   # Multiply obs by this to match model units
    obs_unit_offset: float = 0.0   # Add after multiplying
    group: str = ""


VARIABLE_REGISTRY: dict[str, VarInfo] = {
    # --- Surface atmospheric variables (SFC) ---
    "avg_2t": VarInfo(
        name="avg_2t", long_name="2m Temperature", units="K",
        domain="sfc", cmap="RdBu_r",
        obs_dataset="ERA5", obs_variable="t2m",
        cmip6_variable="tas", cmip6_table="Amon",
        group="temperature",
    ),
    "avg_skt": VarInfo(
        name="avg_skt", long_name="Skin Temperature", units="K",
        domain="sfc", cmap="RdBu_r",
        obs_dataset="ERA5", obs_variable="sst",
        cmip6_variable="ts", cmip6_table="Amon",
        group="temperature",
    ),
    "avg_tprate": VarInfo(
        name="avg_tprate", long_name="Total Precipitation Rate", units="kg/m2/s",
        domain="sfc", cmap="BrBG",
        obs_dataset="ERA5", obs_variable="tp",
        cmip6_variable="pr", cmip6_table="Amon",
        group="precipitation",
    ),
    "avg_msl": VarInfo(
        name="avg_msl", long_name="Mean Sea Level Pressure", units="Pa",
        domain="sfc", cmap="RdBu_r",
        obs_dataset="ERA5", obs_variable="msl",
        cmip6_variable="psl", cmip6_table="Amon",
        group="circulation",
    ),
    "avg_tcc": VarInfo(
        name="avg_tcc", long_name="Total Cloud Cover", units="0-1",
        domain="sfc", cmap="Greys_r",
        obs_dataset="ERA5", obs_variable="tcc",
        cmip6_variable="clt", cmip6_table="Amon",
        group="clouds",
    ),
    "avg_10u": VarInfo(
        name="avg_10u", long_name="10m U Wind", units="m/s",
        domain="sfc", cmap="RdBu_r",
        obs_dataset="ERA5", obs_variable="u10",
        cmip6_variable="uas", cmip6_table="Amon",
        group="wind",
    ),
    "avg_10v": VarInfo(
        name="avg_10v", long_name="10m V Wind", units="m/s",
        domain="sfc", cmap="RdBu_r",
        obs_dataset="ERA5", obs_variable="v10",
        cmip6_variable="vas", cmip6_table="Amon",
        group="wind",
    ),
    "avg_10ws": VarInfo(
        name="avg_10ws", long_name="10m Wind Speed", units="m/s",
        domain="sfc", cmap="YlOrRd",
        obs_dataset="ERA5", obs_variable="u10",  # derive from u10/v10
        group="wind",
    ),
    "avg_tcwv": VarInfo(
        name="avg_tcwv", long_name="Total Column Water Vapour", units="kg/m2",
        domain="sfc", cmap="YlGnBu",
        obs_dataset="ERA5", obs_variable="tcwv",
        cmip6_variable="prw", cmip6_table="Amon",
        group="moisture",
    ),

    # --- Surface radiation fluxes ---
    "avg_tnswrf": VarInfo(
        name="avg_tnswrf", long_name="TOA Net Shortwave Radiation", units="W/m2",
        domain="sfc", cmap="RdBu_r",
        obs_dataset="CERES_EBAF", obs_variable="toa_sw_all_mon",
        cmip6_variable="rsut", cmip6_table="Amon",
        group="radiation",
    ),
    "avg_tnlwrf": VarInfo(
        name="avg_tnlwrf", long_name="TOA Net Longwave Radiation", units="W/m2",
        domain="sfc", cmap="RdBu_r",
        obs_dataset="CERES_EBAF", obs_variable="toa_lw_all_mon",
        cmip6_variable="rlut", cmip6_table="Amon",
        group="radiation",
    ),
    "avg_ishf": VarInfo(
        name="avg_ishf", long_name="Surface Sensible Heat Flux", units="W/m2",
        domain="sfc", cmap="RdBu_r",
        obs_dataset="ERA5", obs_variable="sshf",
        cmip6_variable="hfss", cmip6_table="Amon",
        group="surface_fluxes",
    ),
    "avg_slhtf": VarInfo(
        name="avg_slhtf", long_name="Surface Latent Heat Flux", units="W/m2",
        domain="sfc", cmap="RdBu_r",
        obs_dataset="ERA5", obs_variable="slhf",
        cmip6_variable="hfls", cmip6_table="Amon",
        group="surface_fluxes",
    ),
    "avg_sdswrf": VarInfo(
        name="avg_sdswrf", long_name="Surface Downwelling Shortwave", units="W/m2",
        domain="sfc", cmap="YlOrRd",
        obs_dataset="ERA5", obs_variable="ssrd",
        cmip6_variable="rsds", cmip6_table="Amon",
        group="radiation",
    ),
    "avg_sdlwrf": VarInfo(
        name="avg_sdlwrf", long_name="Surface Downwelling Longwave", units="W/m2",
        domain="sfc", cmap="YlOrRd",
        obs_dataset="ERA5", obs_variable="strd",
        cmip6_variable="rlds", cmip6_table="Amon",
        group="radiation",
    ),

    # --- Ocean 2D variables (O2D) ---
    "avg_tos": VarInfo(
        name="avg_tos", long_name="Sea Surface Temperature", units="K",
        domain="o2d", cmap="RdBu_r",
        obs_dataset="ESA_CCI", obs_variable="analysed_sst",
        cmip6_variable="tos", cmip6_table="Omon",
        group="ocean_surface",
    ),
    "avg_siconc": VarInfo(
        name="avg_siconc", long_name="Sea Ice Concentration", units="0-1",
        domain="o2d", cmap="Blues_r",
        obs_dataset="OSI_SAF", obs_variable="ice_conc",
        cmip6_variable="siconc", cmip6_table="SImon",
        obs_unit_factor=0.01,  # OSI-SAF is 0-100%, model is 0-1
        group="sea_ice",
    ),
    "avg_sithick": VarInfo(
        name="avg_sithick", long_name="Sea Ice Thickness", units="m",
        domain="o2d", cmap="Blues",
        obs_dataset="PSC", obs_variable="hi",
        cmip6_variable="sithick", cmip6_table="SImon",
        group="sea_ice",
    ),
    "avg_zos": VarInfo(
        name="avg_zos", long_name="Sea Surface Height", units="m",
        domain="o2d", cmap="RdBu_r",
        obs_dataset="AVISO", obs_variable="adt",
        cmip6_variable="zos", cmip6_table="Omon",
        group="ocean_surface",
    ),
    "avg_sos": VarInfo(
        name="avg_sos", long_name="Sea Surface Salinity", units="PSU",
        domain="o2d", cmap="PRGn_r",
        obs_dataset="EN4", obs_variable="so",
        cmip6_variable="sos", cmip6_table="Omon",
        group="ocean_surface",
    ),

    # --- Ocean 3D variables (O3D) ---
    "avg_thetao": VarInfo(
        name="avg_thetao", long_name="Ocean Temperature", units="K",
        domain="o3d", cmap="RdBu_r",
        obs_dataset="EN4", obs_variable="thetao",
        group="ocean_3d",
    ),
    "avg_so": VarInfo(
        name="avg_so", long_name="Ocean Salinity", units="PSU",
        domain="o3d", cmap="PRGn_r",
        obs_dataset="EN4", obs_variable="so",
        group="ocean_3d",
    ),

    # --- Pressure level variables (PL) ---
    "avg_t": VarInfo(
        name="avg_t", long_name="Temperature (pressure levels)", units="K",
        domain="pl", cmap="RdBu_r",
        obs_dataset="ERA5", obs_variable="t2m",  # placeholder
        group="atmosphere_3d",
    ),
    "avg_u": VarInfo(
        name="avg_u", long_name="Zonal Wind (pressure levels)", units="m/s",
        domain="pl", cmap="RdBu_r",
        obs_dataset="ERA5", obs_variable="u10",  # placeholder
        group="atmosphere_3d",
    ),
    "avg_v": VarInfo(
        name="avg_v", long_name="Meridional Wind (pressure levels)", units="m/s",
        domain="pl", cmap="RdBu_r",
        obs_dataset="ERA5", obs_variable="v10",  # placeholder
        group="atmosphere_3d",
    ),
    "avg_q": VarInfo(
        name="avg_q", long_name="Specific Humidity (pressure levels)", units="kg/kg",
        domain="pl", cmap="YlGnBu",
        obs_dataset="ERA5", obs_variable="tcwv",  # placeholder
        group="atmosphere_3d",
    ),
}


def get_var(name: str) -> VarInfo:
    """Look up variable metadata by model variable name."""
    if name not in VARIABLE_REGISTRY:
        raise KeyError(f"Unknown variable: {name!r}. "
                       f"Available: {list(VARIABLE_REGISTRY.keys())}")
    return VARIABLE_REGISTRY[name]


def list_vars(domain: str = None, group: str = None) -> list[VarInfo]:
    """List variables, optionally filtered by domain or group."""
    result = list(VARIABLE_REGISTRY.values())
    if domain is not None:
        result = [v for v in result if v.domain == domain]
    if group is not None:
        result = [v for v in result if v.group == group]
    return result
