"""Tests for area-conservative remapping of flux/precipitation fields.

Covers the variable classification, the method chooser on ``DiagnosticBase``
(availability probe + size guard + config switch), and the configurable
``precipitation_mswep`` common grid.
"""

import numpy as np
import pytest

from feather.config import FeatherConfig
from feather.data.variables import (
    VARIABLE_REGISTRY,
    _EXTRA_FLUX_VARS,
    _FLUX_GROUPS,
    is_flux_variable,
)
from feather.diag.base import DiagnosticBase


# ── Variable classification ──────────────────────────────────────────────


class TestFluxClassification:
    @pytest.mark.parametrize("var", ["pr", "hfss", "hfls"])
    def test_precip_and_turbulent_fluxes_are_fluxes(self, var):
        assert is_flux_variable(var)

    @pytest.mark.parametrize("var", ["rsds", "rlds", "rss", "rls", "rsscs",
                                     "rlscs", "rst", "rlt", "rstcs", "rltcs"])
    def test_radiation_fields_are_fluxes(self, var):
        assert is_flux_variable(var)

    @pytest.mark.parametrize("var", ["tas", "ts", "psl", "uas", "vas",
                                     "sfcWind", "clt", "tos", "sos", "siconc",
                                     "sithick", "thetao", "so", "zos",
                                     "tasmin", "tasmax", "prw"])
    def test_state_variables_are_not_fluxes(self, var):
        """Intensive state variables keep point interpolation."""
        assert not is_flux_variable(var)

    def test_unknown_variable_is_not_a_flux(self):
        """Unknown names fall back to the configured method, never crash."""
        assert not is_flux_variable("no_such_variable")

    def test_extra_flux_vars_covered_even_if_unregistered(self):
        for var in _EXTRA_FLUX_VARS:
            assert is_flux_variable(var)

    def test_classification_matches_groups(self):
        for name, info in VARIABLE_REGISTRY.items():
            if info.group in _FLUX_GROUPS:
                assert is_flux_variable(name), name

    def test_no_wind_or_stress_misclassified(self):
        """Wind is a state variable, not a flux — must not be conservative."""
        assert not is_flux_variable("uas")
        assert not is_flux_variable("sfcWind")


# ── Method chooser ───────────────────────────────────────────────────────


class _Probe(DiagnosticBase):
    name = "probe"
    title = "Probe"
    domain = "sfc"
    variables = ["tas"]
    group = "g"

    def compute(self):
        return {}

    def plot(self, results):
        return []


def _cfg(tmp_path, **nereus):
    base = {"method": "linear"}
    base.update(nereus)
    return FeatherConfig(
        model_catalogs={}, models=["M"], obs_root="", obs_datasets={},
        cmip6={"enabled": False}, dask={}, nereus=base,
        output_dir=str(tmp_path), data_source={"type": "cmor"},
        model_configs={},
    )


def _probe(tmp_path, **nereus):
    return _Probe(None, None, _cfg(tmp_path, **nereus))


class TestMethodChooser:
    def test_flux_gets_conservative(self, tmp_path):
        assert _probe(tmp_path)._regrid_method_for("pr") == "conservative"

    def test_non_flux_keeps_configured_method(self, tmp_path):
        assert _probe(tmp_path)._regrid_method_for("tas") == "linear"

    def test_non_flux_respects_nearest_default(self, tmp_path):
        p = _Probe(None, None, _cfg(tmp_path, method="nearest"))
        assert p._regrid_method_for("tas") == "nearest"

    def test_config_switch_disables_conservative(self, tmp_path):
        p = _probe(tmp_path, conservative_fluxes=False)
        assert p._regrid_method_for("pr") == "linear"

    def test_size_guard_falls_back(self, tmp_path):
        p = _probe(tmp_path, conservative_max_points=1000)
        assert p._regrid_method_for("pr", 5000) == "linear"

    def test_under_size_guard_keeps_conservative(self, tmp_path):
        p = _probe(tmp_path, conservative_max_points=1000)
        assert p._regrid_method_for("pr", 999) == "conservative"

    def test_no_size_given_skips_guard(self, tmp_path):
        p = _probe(tmp_path, conservative_max_points=1)
        assert p._regrid_method_for("pr") == "conservative"

    def test_target_guard_falls_back(self, tmp_path):
        """Cost grows with target cell count, not just source."""
        p = _probe(tmp_path, conservative_max_points=2_000_000)
        # 0.1 deg target = 6.48M cells
        assert p._regrid_method_for("pr", 1000, resolution=0.1) == "linear"

    def test_target_guard_passes_at_quarter_degree(self, tmp_path):
        """0.25 deg target = 1.04M cells, under the default guard."""
        p = _probe(tmp_path, conservative_max_points=2_000_000)
        assert (p._regrid_method_for("pr", 1000, resolution=0.25)
                == "conservative")

    def test_target_guard_respects_raised_limit(self, tmp_path):
        p = _probe(tmp_path, conservative_max_points=10_000_000)
        assert (p._regrid_method_for("pr", 1000, resolution=0.1)
                == "conservative")

    def test_target_guard_ignored_for_non_flux(self, tmp_path):
        p = _probe(tmp_path)
        assert p._regrid_method_for("tas", 1000, resolution=0.1) == "linear"

    def test_budget_is_source_plus_target(self, tmp_path):
        """Neither dimension alone exceeds the limit, but together they do."""
        p = _probe(tmp_path, conservative_max_points=8_000_000)
        # 6.48M source onto a 1.04M target (MSWEP -> 0.25 deg): affordable
        assert (p._regrid_method_for("pr", 6_480_000, resolution=0.25)
                == "conservative")
        # 6.48M source onto a 6.48M target (MSWEP -> 0.1 deg): not
        assert (p._regrid_method_for("pr", 6_480_000, resolution=0.1)
                == "linear")

    def test_mswep_coarsening_allowed_at_eerie_budget(self):
        """The EERIE config must conserve the MSWEP->0.25 deg aggregation.

        A 2M budget would pass the near-identity model regrid but skip this
        one, i.e. conserve only where it does not matter.
        """
        cfg = FeatherConfig.from_yaml("configs/eerie_10_mems_cmip6.yaml")
        probe = _Probe(None, None, cfg)
        res = cfg.get_precip_resolution(0.1)
        assert probe._regrid_method_for(
            "pr", 6_480_000, resolution=res) == "conservative"

    def test_destine_source_still_guarded(self):
        """12.6M-point HEALPix sources must not attempt conservative."""
        cfg = FeatherConfig.from_yaml("configs/eerie_10_mems_cmip6.yaml")
        probe = _Probe(None, None, cfg)
        assert probe._regrid_method_for(
            "pr", 12_582_912, resolution=0.25) == "linear"

    def test_is_flux_override_forces_conservative(self, tmp_path):
        """Derived-quantity keys aren't registry names."""
        p = _probe(tmp_path)
        assert p._regrid_method_for("toa_net", is_flux=True) == "conservative"

    def test_is_flux_override_can_force_off(self, tmp_path):
        p = _probe(tmp_path)
        assert p._regrid_method_for("pr", is_flux=False) == "linear"

    def test_fallback_when_nereus_lacks_support(self, tmp_path, monkeypatch):
        monkeypatch.setattr(DiagnosticBase, "_CONSERVATIVE_SUPPORTED", False)
        assert _probe(tmp_path)._regrid_method_for("pr") == "linear"

    def test_availability_probe_detects_installed_nereus(self):
        """This env has nereus from main, which supports conservative."""
        assert DiagnosticBase._conservative_available() is True


# ── Config plumbing ──────────────────────────────────────────────────────


class TestPrecipResolution:
    def test_defaults_to_obs_native(self, tmp_path):
        assert _cfg(tmp_path).get_precip_resolution(0.1) == 0.1

    def test_override_wins(self, tmp_path):
        cfg = _cfg(tmp_path, precip_resolution=0.25)
        assert cfg.get_precip_resolution(0.1) == 0.25

    def test_eerie_config_uses_quarter_degree(self):
        cfg = FeatherConfig.from_yaml("configs/eerie_10_mems_cmip6.yaml")
        assert cfg.get_precip_resolution(0.1) == 0.25

    def test_destine_config_uses_tenth_degree(self):
        cfg = FeatherConfig.from_yaml("configs/default.yaml")
        assert cfg.get_precip_resolution(0.25) == 0.1

    def test_precip_resolution_independent_of_nereus_resolution(self):
        """Both configs declare resolution: 0.25 — precip must not read it."""
        eerie = FeatherConfig.from_yaml("configs/eerie_10_mems_cmip6.yaml")
        destine = FeatherConfig.from_yaml("configs/default.yaml")
        assert eerie.nereus["resolution"] == destine.nereus["resolution"]
        assert (eerie.get_precip_resolution(0.1)
                != destine.get_precip_resolution(0.1))

    def test_conservative_defaults_on(self, tmp_path):
        assert _cfg(tmp_path).use_conservative_fluxes() is True

    def test_conservative_can_be_disabled(self, tmp_path):
        assert _cfg(tmp_path, conservative_fluxes=False
                    ).use_conservative_fluxes() is False


# ── End-to-end conservation property ─────────────────────────────────────


class TestConservationProperty:
    """The point of the change: the area integral must survive coarsening."""

    @staticmethod
    def _area_integral(field, lat2d, dlon, dlat):
        r = 6.371e6
        w = (np.deg2rad(dlon) * r) * (np.deg2rad(dlat) * r) * np.cos(
            np.deg2rad(lat2d))
        m = np.isfinite(field)
        return float(np.nansum(field[m] * w[m]))

    @pytest.fixture(scope="class")
    def spiky_source(self):
        rng = np.random.default_rng(0)
        lon = np.arange(1.0, 360, 2.0)
        lat = np.arange(-89.0, 90, 2.0)
        lo, la = np.meshgrid(lon, lat)
        data = rng.gamma(shape=0.4, scale=8.0, size=la.shape)
        return lo, la, data

    def test_conservative_beats_nearest_on_coarsening(self, spiky_source):
        import nereus as nr

        lo, la, data = spiky_source
        src = self._area_integral(data, la, 2.0, 2.0)

        errors = {}
        for method in ("nearest", "conservative"):
            out, it = nr.regrid(
                data.ravel(), lo.ravel(), la.ravel(), resolution=6.0,
                method=method, lon_bounds=(0.0, 360.0),
                influence_radius=800_000,
            )
            got = self._area_integral(out, it.target_lat, 6.0, 6.0)
            errors[method] = abs(got - src) / src

        assert errors["conservative"] < 0.01
        assert errors["conservative"] < errors["nearest"]

    def test_conservative_damps_spurious_extremes(self, spiky_source):
        """Nearest reports a point value as a box mean; conservative averages."""
        import nereus as nr

        lo, la, data = spiky_source
        out = {}
        for method in ("nearest", "conservative"):
            arr, _ = nr.regrid(
                data.ravel(), lo.ravel(), la.ravel(), resolution=6.0,
                method=method, lon_bounds=(0.0, 360.0),
                influence_radius=800_000,
            )
            out[method] = np.nanmax(arr)

        assert out["conservative"] < out["nearest"]
