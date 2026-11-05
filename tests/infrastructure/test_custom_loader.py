from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from radioskysims_beam.domain.beam.models import Beam
from radioskysims_beam.infrastructure.beams.loaders.custom import CustomBeamLoader


def create_temp_beam_module(content: str):
    """Cria um módulo Python temporário com funções amplitude/phase."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp_file:
        tmp_file.write(content.encode("utf-8"))
        return tmp_file.name


def test_custom_loader_success():
    # ✅ Módulo Python válido com amplitude e phase
    module_code = """
def amplitude(pol, freq, angle):
    return abs(freq * 0.1)  # valor constante baseado na frequência

def phase(pol, freq, angle):
    return angle * 0.01  # fase cresce linearmente
"""
    module_path = create_temp_beam_module(module_code)

    # Dados de entrada simulados
    polarizations = ["X", "Y"]
    frequencies = [100.0, 200.0]
    angles = [0, 45, 90]

    loader = CustomBeamLoader(module_path, polarizations, frequencies, angles)
    beam = loader.load()

    assert isinstance(beam, Beam)

    ds = beam.dataset
    assert "amplitude" in ds
    assert "phase" in ds

    # Verifica forma
    assert ds["amplitude"].shape == (2, 2, 3)

    # Verifica conteúdo conhecido
    assert np.all(
        ds["amplitude"].values
        == np.array([[[10, 10, 10], [20, 20, 20]], [[10, 10, 10], [20, 20, 20]]], dtype=float)
    )
    assert np.isclose(ds["phase"].values[0, 0, 1], 0.45)  # angle 45 * 0.01

    # Cleanup
    Path(module_path).unlink()


def test_custom_loader_without_phase():
    # ✅ Módulo sem função phase
    module_code = """
def amplitude(pol, freq, angle):
    return 1.0
"""
    module_path = create_temp_beam_module(module_code)

    polarizations = ["H"]
    frequencies = [50.0]
    angles = [0, 30, 60]

    loader = CustomBeamLoader(module_path, polarizations, frequencies, angles)
    beam = loader.load()

    ds = beam.dataset
    assert np.all(ds["amplitude"].values == 1.0)
    assert np.all(ds["phase"].values == 0.0)  # default zero phase

    Path(module_path).unlink()
