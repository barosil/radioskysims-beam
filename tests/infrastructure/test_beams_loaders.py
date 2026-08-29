"""
Testes para os loaders de beam em infrastructure/beams/loaders/.

Cobre CSVBeamLoader, CustomBeamLoader, LITBeamLoader e _utils.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from radioskysims_beam.domain.beam.models import Beam
from radioskysims_beam.infrastructure.beams.loaders._utils import finalize_beam_dataset
from radioskysims_beam.infrastructure.beams.loaders.analytical import AnalyticalBeamLoader
from radioskysims_beam.infrastructure.beams.loaders.csv import CSVBeamLoader
from radioskysims_beam.infrastructure.beams.loaders.custom import CustomBeamLoader
from radioskysims_beam.infrastructure.beams.loaders.lit import LITBeamLoader

# =============================================================================
# _utils: finalize_beam_dataset
# =============================================================================


class TestFinalizeBeamDataset:
    def test_valid_dataset(self):
        ds = xr.Dataset(
            {"amplitude": (("polarization", "frequency", "angle"), np.ones((1, 1, 3)))},
            coords={
                "polarization": ["H"],
                "frequency": [100.0],
                "angle": np.array([-45.0, 0.0, 45.0]),
            },
        )
        result = finalize_beam_dataset(ds)
        assert result["amplitude"].dtype == np.float64
        assert result["angle"].dtype == np.float64

    def test_missing_polarization_dim(self):
        ds = xr.Dataset(
            {"amplitude": (("frequency", "angle"), np.ones((1, 3)))},
            coords={"frequency": [100.0], "angle": [0.0, 1.0, 2.0]},
        )
        with pytest.raises(ValueError, match="Dataset inválido"):
            finalize_beam_dataset(ds)

    def test_missing_amplitude(self):
        ds = xr.Dataset(
            {"foo": (("polarization", "frequency", "angle"), np.ones((1, 1, 3)))},
            coords={
                "polarization": ["H"],
                "frequency": [100.0],
                "angle": [0.0, 1.0, 2.0],
            },
        )
        with pytest.raises(ValueError, match="Dataset não contém"):
            finalize_beam_dataset(ds)

    def test_with_phase(self):
        ds = xr.Dataset(
            {
                "amplitude": (
                    ("polarization", "frequency", "angle"),
                    np.ones((1, 1, 3)),
                ),
                "phase": (("polarization", "frequency", "angle"), np.zeros((1, 1, 3))),
            },
            coords={
                "polarization": ["H"],
                "frequency": [100.0],
                "angle": np.array([-45.0, 0.0, 45.0]),
            },
        )
        result = finalize_beam_dataset(ds)
        assert "phase" in result
        assert result["phase"].dtype == np.float64


# =============================================================================
# CSVBeamLoader
# =============================================================================


class TestCSVBeamLoader:
    @pytest.fixture
    def sample_csv(self, tmp_path):
        data = {
            "polarization": ["X", "X", "Y", "Y"],
            "frequency": [100, 100, 200, 200],
            "angle": [-45, 45, -45, 45],
            "amplitude": [0.5, 1.0, 0.8, 0.9],
        }
        file_path = tmp_path / "beam.csv"
        pd.DataFrame(data).to_csv(file_path, index=False)
        return file_path

    def test_load_success(self, sample_csv):
        loader = CSVBeamLoader(str(sample_csv))
        beam = loader.load()
        assert isinstance(beam, Beam)
        assert "amplitude" in beam.dataset.data_vars
        assert "polarization" in beam.dataset.coords

    def test_missing_optional_columns(self, tmp_path):
        data = {
            "frequency": [100, 100],
            "angle": [-45, 45],
            "amplitude": [0.5, 1.0],
        }
        file_path = tmp_path / "beam_no_pol.csv"
        pd.DataFrame(data).to_csv(file_path, index=False)
        loader = CSVBeamLoader(str(file_path))
        beam = loader.load()
        assert all(beam.dataset["polarization"].to_numpy() == "UNDEFINED")

    def test_file_not_found(self):
        loader = CSVBeamLoader("/nonexistent/beam.csv")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_custom_varnames(self, tmp_path):
        data = {
            "pol": ["X", "Y"],
            "freq": [100, 200],
            "ang": [-45, 45],
            "amp": [0.5, 1.0],
        }
        file_path = tmp_path / "beam_custom.csv"
        pd.DataFrame(data).to_csv(file_path, index=False)
        varnames = {
            "polarization": "pol",
            "frequency": "freq",
            "angle": "ang",
            "amplitude": "amp",
            "phase": "phase",
        }
        loader = CSVBeamLoader(str(file_path), varnames=varnames)
        beam = loader.load()
        assert isinstance(beam, Beam)
        assert "amplitude" in beam.dataset.data_vars


# =============================================================================
# CustomBeamLoader
# =============================================================================


class TestCustomBeamLoader:
    def _create_temp_module(self, content: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
            f.write(content.encode("utf-8"))
            return f.name

    def test_load_success(self):
        code = """
def amplitude(pol, freq, angle):
    return abs(freq * 0.1)
def phase(pol, freq, angle):
    return angle * 0.01
"""
        path = self._create_temp_module(code)
        loader = CustomBeamLoader(path, ["X", "Y"], [100.0, 200.0], [0, 45, 90])
        beam = loader.load()
        assert isinstance(beam, Beam)
        assert beam.dataset["amplitude"].shape == (2, 2, 3)
        Path(path).unlink()

    def test_without_phase(self):
        code = """
def amplitude(pol, freq, angle):
    return 1.0
"""
        path = self._create_temp_module(code)
        loader = CustomBeamLoader(path, ["X"], [100.0], [0, 45])
        beam = loader.load()
        assert isinstance(beam, Beam)
        assert "phase" in beam.dataset.data_vars
        Path(path).unlink()

    def test_invalid_module_path(self):
        loader = CustomBeamLoader("/nonexistent/module.py", ["X"], [100.0], [0])
        with pytest.raises((ImportError, FileNotFoundError)):
            loader.load()


# =============================================================================
# LITBeamLoader
# =============================================================================


class TestLITBeamLoader:
    @pytest.fixture
    def sample_lit_file(self, tmp_path):
        """Gera um arquivo Excel simulando formato LIT com 2 sheets (H, V).

        Formato esperado pelo LITBeamLoader:
          - Linha de cabeçalho contendo: Angle, Amplitude, Phase
          - Linha anterior contém labels de frequência (ex: '1.0 GHz')
          - Dados começam na linha seguinte ao cabeçalho
          - Amplitude em dB (negativa), ângulos em graus, frequência com
            separador decimal pt-br ('1,0 GHz').
        """
        file_path = tmp_path / "beam_lit.xlsx"

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            for sheet_name in ("H", "V"):
                rows = [
                    ["", "", "1,0 GHz", "", "2,0 GHz", ""],
                    ["Angle", "", "Amplitude", "Phase", "Amplitude", "Phase"],
                    [-10.0, "", -20.0, 0.0, -25.0, 0.0],
                    [0.0, "", -2.0, 0.2, -3.0, 0.1],
                    [10.0, "", -20.0, 0.0, -25.0, 0.0],
                ]
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

        return file_path

    def test_load_success(self, sample_lit_file):
        loader = LITBeamLoader(sample_lit_file)
        beam = loader.load()
        assert isinstance(beam, Beam)
        assert "amplitude" in beam.dataset.data_vars
        assert "phase" in beam.dataset.data_vars
        assert len(beam.dataset.polarization) == 2  # H, V

    def test_load_normalizes_peak_to_zero_db(self, sample_lit_file):
        """A normalização deve ajustar o pico de cada banda para 0 dB (NÃO min-max).

        Dados em dB por banda (1,0 GHz / 2,0 GHz):
          [-20, -2, -20] -> pico deslocado para 0 dB -> [-18, 0, -18]
          [-25, -3, -25] -> pico deslocado para 0 dB -> [-22, 0, -22]
        """
        loader = LITBeamLoader(sample_lit_file)
        beam = loader.load()
        amp = beam.dataset["amplitude"]

        # pico de cada banda deve ser exatamente 0 dB
        peak_by_freq = amp.max(dim="angle").max(dim="polarization")
        assert np.allclose(peak_by_freq.values, 0.0)

        # amplitudes relativas ao pico (≤ 0 dB), e NÃO reescaladas para [0, 1]
        assert float(amp.min()) < 0.0
        assert np.all(amp.values <= 0.0 + 1e-9)

        # verifica valores concretos na primeira banda (1,0 GHz)
        h1 = amp.sel(polarization="H", frequency=1000.0)
        assert np.allclose(h1.values, [-18.0, 0.0, -18.0], atol=1e-9)

    def test_load_frequency_ptbr_decimal(self, tmp_path):
        """Frequência com separador decimal pt-br ('1,0 GHz') deve ser lida."""
        file_path = tmp_path / "beam_ptbr.xlsx"
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            rows = [
                ["", "1,5 GHz", ""],
                ["Angle", "Amplitude", "Phase"],
                [-10.0, -20.0, 0.0],
                [0.0, -2.0, 0.0],
                [10.0, -20.0, 0.0],
            ]
            pd.DataFrame(rows).to_excel(writer, sheet_name="H", index=False, header=False)
        loader = LITBeamLoader(file_path)
        beam = loader.load()
        assert np.allclose(beam.dataset.frequency.values, [1500.0])

    def test_file_not_found(self):
        loader = LITBeamLoader("/nonexistent/beam.xlsx")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_sheet_polarization_parsing(self, tmp_path):
        """Testa que nomes de sheet como 'Horizontal' e 'VERTICAL' são interpretados."""
        file_path = tmp_path / "beam_pol.xlsx"
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            for sheet_name in ("Horizontal", "VERTICAL"):
                rows = [
                    ["", "1.0 GHz", ""],
                    ["Angle", "Amplitude", "Phase"],
                    [0.0, 1.0, 0.2],
                    [10.0, 0.8, 0.1],
                ]
                pd.DataFrame(rows).to_excel(
                    writer, sheet_name=sheet_name, index=False, header=False
                )

        loader = LITBeamLoader(file_path)
        beam = loader.load()
        assert set(beam.dataset.polarization.to_numpy()) == {"H", "V"}


# =============================================================================
# AnalyticalBeamLoader
# =============================================================================


class TestAnalyticalBeamLoader:
    def test_load_sets_coordinate_units(self):
        beam = AnalyticalBeamLoader(
            beam_profile="gaussian",
            frequency=1.0e9,
            angular_range=(-2.0, 2.0),
            angular_resolution=0.005,
            polarization="H",
            sigma=0.36,
        ).load()

        assert beam.dataset["angle"].attrs["units"] == "deg"
        assert beam.dataset["frequency"].attrs["units"] == "Hz"
