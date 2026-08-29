from __future__ import annotations

import pandas as pd
import pytest

from radioskysims_beam.domain.beam.models import Beam
from radioskysims_beam.infrastructure.beams.loaders.csv import CSVBeamLoader


@pytest.fixture
def sample_csv(tmp_path):
    """Cria um CSV temporário válido."""
    data = {
        "polarization": ["X", "X", "Y", "Y"],
        "frequency": [100, 100, 200, 200],
        "angle": [-45, 45, -45, 45],
        "amplitude": [0.5, 1.0, 0.8, 0.9],
    }
    file_path = tmp_path / "beam.csv"
    pd.DataFrame(data).to_csv(file_path, index=False)
    return file_path


def test_csv_loader_success(sample_csv):
    loader = CSVBeamLoader(str(sample_csv))
    beam = loader.load()

    assert isinstance(beam, Beam)
    assert "amplitude" in beam.dataset.data_vars
    assert "polarization" in beam.dataset.coords
    assert beam.dataset["amplitude"].max() == pytest.approx(1.0)


def test_csv_loader_missing_optional_columns(tmp_path):
    """Testa caso sem polarization e phase (preenche default)."""
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
    assert "phase" in beam.dataset.data_vars
