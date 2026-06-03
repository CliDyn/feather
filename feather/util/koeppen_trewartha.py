"""Köppen–Trewartha (KT14) climate classification.

Vectorised xarray implementation of the Köppen–Trewartha climate
classification with 14 types, following the rules used in the reference
notebook ``01bc_plot_compute_KT14Types_CORE_AFR22_HIST_only_9`` (Trewartha
dryness threshold ``R = 2.3·Tann − 0.64·Pw + 41`` in cm).

The classifier works on a **monthly climatology**:

- ``tmon`` : monthly-mean temperature in **°C**, dims ``(month, lat, lon)``
- ``pmon`` : monthly-mean precipitation in **cm/month**, dims ``(month, lat, lon)``
- ``lat``  : latitude DataArray (dim ``lat``) used for hemisphere-aware seasons

Assignment priority is **F → E → B → D → C → A**; a cell keeps the first
group whose criteria it satisfies (matching the notebook, which only fills
still-unclassified cells in that order).  Dry cells with zero annual
precipitation that remain unclassified are assigned ``BW``.

Final integer codes (1–14) and labels::

    1 Ar   2 Aw   3 As   4 BW   5 BS   6 Cs   7 Cw
    8 Cf   9 Do  10 Dc  11 Eo  12 Ec  13 Ft  14 Fi

Code ``0`` means *unclassified* (a handful of A cells can fall through all
A sub-type tests); ``NaN`` marks cells with missing input.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# ── KT14 code / label / colour tables ──────────────────────────────────
# Order matches the reference notebook's ``koeppen_trewartha14`` palette.

KT_CODES: dict[int, str] = {
    1: "Ar", 2: "Aw", 3: "As",
    4: "BW", 5: "BS",
    6: "Cs", 7: "Cw", 8: "Cf",
    9: "Do", 10: "Dc",
    11: "Eo", 12: "Ec",
    13: "Ft", 14: "Fi",
}

# Ordered list of the 14 labels (index 0 → code 1, …)
KT_LABELS: list[str] = [KT_CODES[i] for i in range(1, 15)]

# RGB colours (0–1) from the notebook palette, keyed by label.
KT_COLORS: dict[str, tuple[float, float, float]] = {
    "Ar": (41 / 255, 76 / 255, 41 / 255),
    "Aw": (0 / 255, 231 / 255, 0 / 255),
    "As": (204 / 255, 255 / 255, 204 / 255),
    "BW": (255 / 255, 170 / 255, 0 / 255),
    "BS": (255 / 255, 239 / 255, 125 / 255),
    "Cs": (255 / 255, 95 / 255, 0 / 255),
    "Cw": (12 / 255, 170 / 255, 12 / 255),
    "Cf": (25 / 255, 112 / 255, 241 / 255),
    "Do": (0 / 255, 102 / 255, 102 / 255),
    "Dc": (102 / 255, 178 / 255, 255 / 255),
    "Eo": (115 / 255, 95 / 255, 231 / 255),
    "Ec": (40 / 255, 25 / 255, 174 / 255),
    "Ft": (255 / 255, 213 / 255, 213 / 255),
    "Fi": (213 / 255, 54 / 255, 54 / 255),
}

# Human-readable descriptions for metadata / tables.
KT_DESCRIPTIONS: dict[str, str] = {
    "Ar": "Tropical rainforest",
    "Aw": "Tropical wet-and-dry (winter dry)",
    "As": "Tropical (summer dry)",
    "BW": "Arid desert",
    "BS": "Semi-arid steppe",
    "Cs": "Subtropical dry-summer (Mediterranean)",
    "Cw": "Subtropical dry-winter",
    "Cf": "Subtropical humid",
    "Do": "Temperate oceanic",
    "Dc": "Temperate continental",
    "Eo": "Boreal oceanic",
    "Ec": "Boreal continental",
    "Ft": "Polar tundra",
    "Fi": "Polar ice (frost)",
}

# Months (1-based) making up the Northern-Hemisphere winter / summer
# half-years.  In the Southern Hemisphere the two sets swap.
_NH_WINTER_MONTHS = [1, 2, 3, 10, 11, 12]
_NH_SUMMER_MONTHS = [4, 5, 6, 7, 8, 9]


def classify_kt(
    tmon: xr.DataArray,
    pmon: xr.DataArray,
    lat: xr.DataArray,
) -> xr.DataArray:
    """Classify a monthly climatology into the 14 Köppen–Trewartha types.

    Parameters
    ----------
    tmon : xr.DataArray
        Monthly-mean temperature in °C, dims ``(month, lat, lon)`` with a
        1-based ``month`` coordinate.
    pmon : xr.DataArray
        Monthly-mean precipitation in cm/month, same dims as *tmon*.
    lat : xr.DataArray
        Latitude (dim ``lat``), used to pick the local winter/summer
        half-years.

    Returns
    -------
    xr.DataArray
        Integer ``kt_code`` field, dims ``(lat, lon)``.  Values 1–14 map
        to :data:`KT_CODES`; ``0`` is unclassified; ``NaN`` is missing.
    """
    month = tmon["month"]

    # ── Temperature statistics ─────────────────────────────────────────
    Tann = tmon.mean("month")
    Tcold = tmon.min("month")
    Twarm = tmon.max("month")
    months_ge10 = (tmon >= 10.0).sum("month")

    # ── Precipitation statistics (cm) ──────────────────────────────────
    Pann = pmon.sum("month")

    nh_winter = month.isin(_NH_WINTER_MONTHS)
    nh_summer = month.isin(_NH_SUMMER_MONTHS)
    is_nh = lat >= 0

    # Half-year precipitation sums (hemisphere-aware)
    P_nhwin_sum = pmon.where(nh_winter).sum("month")   # NH winter / SH summer
    P_nhsum_sum = pmon.where(nh_summer).sum("month")   # NH summer / SH winter
    Pwinter = xr.where(is_nh, P_nhwin_sum, P_nhsum_sum)
    Psummer = xr.where(is_nh, P_nhsum_sum, P_nhwin_sum)

    # Driest month within each half-year (hemisphere-aware)
    P_nhwin_min = pmon.where(nh_winter).min("month")
    P_nhsum_min = pmon.where(nh_summer).min("month")
    Pdry_summer = xr.where(is_nh, P_nhsum_min, P_nhwin_min)
    Pdry_winter = xr.where(is_nh, P_nhwin_min, P_nhsum_min)

    # Wet/dry month counts (threshold 6 cm = 60 mm)
    Pwet = (pmon > 6.0).sum("month")
    Pdry = (pmon <= 6.0).sum("month")

    # Percentage of annual precip falling in the winter half-year
    Pw = xr.where(Pann > 0, 100.0 * Pwinter / Pann, 0.0)

    # Trewartha dryness threshold (cm)
    R = 2.3 * Tann - 0.64 * Pw + 41.0
    halfR = 0.5 * R

    # ── Assemble codes in priority order F → E → B → D → C → A ─────────
    code = xr.zeros_like(Tann)

    def _assign(mask, value):
        nonlocal code
        code = xr.where((code == 0) & mask, value, code)

    # F — Polar: warmest month ≤ 10 °C
    is_F = Twarm <= 10.0
    _assign(is_F & (Twarm > 0.0), 13)    # Ft
    _assign(is_F & (Twarm <= 0.0), 14)   # Fi

    # E — Boreal: 1–3 months ≥ 10 °C
    is_E = (months_ge10 >= 1) & (months_ge10 <= 3)
    _assign(is_E & (Tcold > -10.0), 11)  # Eo
    _assign(is_E & (Tcold <= -10.0), 12)  # Ec

    # B — Dry: annual precip below the dryness threshold
    is_B = Pann < R
    _assign(is_B & (Pann <= halfR), 4)   # BW
    _assign(is_B & (Pann > halfR), 5)    # BS

    # D — Temperate/continental: 4–7 months ≥ 10 °C
    is_D = (months_ge10 >= 4) & (months_ge10 <= 7)
    _assign(is_D & (Tcold > 0.0), 9)     # Do
    _assign(is_D & (Tcold <= 0.0), 10)   # Dc

    # C — Subtropical: coldest ≤ 18 °C and ≥ 8 months ≥ 10 °C
    is_C = (Tcold <= 18.0) & (months_ge10 >= 8)
    Cs = is_C & (Pdry_summer <= 3.0) & (Pwinter >= 3.0 * Psummer) & (Pann <= 89.0)
    Cw = is_C & (Pdry_winter <= 3.0) & (Psummer >= 10.0 * Pwinter)
    _assign(Cs, 6)
    _assign(Cw, 7)
    _assign(is_C, 8)                     # Cf — remaining subtropical

    # A — Tropical: coldest ≥ 18 °C and not dry
    is_A = (Tcold >= 18.0) & (Pann >= R)
    Ar = is_A & (Pwet >= 10) & (Pdry <= 2)
    Aw = is_A & (Pdry_winter < 6.0) & (Pdry > 2)
    As = is_A & (Pdry_summer <= 6.0)
    _assign(Ar, 1)
    _assign(Aw, 2)
    _assign(As, 3)

    # Dry fix: unclassified cells with zero annual precip → desert (BW)
    _assign(Pann == 0.0, 4)

    # Mask cells with missing input
    code = code.where(Tann.notnull())
    code.name = "kt_code"
    code.attrs.update(
        long_name="Köppen–Trewartha climate type code",
        flag_values=list(range(1, 15)),
        flag_meanings=" ".join(KT_LABELS),
        reference="Trewartha & Horn (1980); Belda et al. (2014)",
    )
    return code


def area_percent_by_type(
    code: xr.DataArray,
    area: xr.DataArray,
) -> dict[str, float]:
    """Return percentage of (classified) area in each KT type.

    Parameters
    ----------
    code : xr.DataArray
        Integer ``kt_code`` field (``(lat, lon)``); NaN/0 cells excluded.
    area : xr.DataArray
        Per-cell area weights aligned to *code* (``(lat, lon)``).

    Returns
    -------
    dict
        ``{label: percent}`` for all 14 labels (0.0 when absent).  The
        denominator is the total area of cells with code 1–14, so each
        source sums to 100 %.
    """
    valid = code.notnull() & (code >= 1) & (code <= 14)
    total = float(area.where(valid).sum())
    out = {label: 0.0 for label in KT_LABELS}
    if total <= 0:
        return out
    for c, label in KT_CODES.items():
        a = float(area.where(valid & (code == c)).sum())
        out[label] = 100.0 * a / total
    return out


def transition_matrix(
    ref_code: xr.DataArray,
    future_code: xr.DataArray,
    area: xr.DataArray,
) -> np.ndarray:
    """Area-weighted KT transition matrix (% of classified land area).

    ``M[i, j]`` is the percentage of the (jointly classified) land area that is
    KT type ``i+1`` in *ref_code* and type ``j+1`` in *future_code*.  Rows are
    the reference type, columns the future type; the whole matrix sums to 100 %.
    Cells unclassified in either field are excluded.
    """
    valid = (
        ref_code.notnull() & future_code.notnull()
        & (ref_code >= 1) & (ref_code <= 14)
        & (future_code >= 1) & (future_code <= 14)
    )
    total = float(area.where(valid).sum())
    M = np.zeros((14, 14), dtype=float)
    if total <= 0:
        return M
    rc = np.asarray(ref_code.values)
    fc = np.asarray(future_code.values)
    av = np.asarray(area.values)
    vmask = np.asarray(valid.values)
    for i in range(1, 15):
        ri = vmask & (rc == i)
        if not ri.any():
            continue
        for j in range(1, 15):
            cell = ri & (fc == j)
            if cell.any():
                M[i - 1, j - 1] = 100.0 * float(np.nansum(np.where(cell, av, 0.0))) / total
    return M
