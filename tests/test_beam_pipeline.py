from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from sqlmodel import Session, SQLModel, create_engine

from radioskysims_beam.domain.beam.models import Beam
from radioskysims_beam.infrastructure.db.beam_db_model import BeamParamsDB as BeamParams
from radioskysims_beam.infrastructure.db.beam_repository import BeamRepository


# -----------------------------------------------------------------------------
# FIXTURES DE BANCO E REPOSITÓRIO
# -----------------------------------------------------------------------------
@pytest.fixture(name="session")
def session_fixture():
    """Cria uma sessão SQLite em memória (isolada por teste)."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def repo(session):
    """Instância do repositório para os testes."""
    return BeamRepository(session)


# -----------------------------------------------------------------------------
# FIXTURE DE BEAM DE TESTE (simulado)
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_beam_dataset():
    """Cria um Dataset simulado de feixe (xarray)."""
    angles = np.linspace(-90, 90, 181)  # graus
    polarizations = ["X", "Y"]
    frequencies = [100.0, 200.0]
    rng = np.random.default_rng(42)

    # Cria amplitude como senoides com ruído leve
    amp_data = np.zeros((len(polarizations), len(frequencies), len(angles)))
    for i, _pol in enumerate(polarizations):
        for j, _f in enumerate(frequencies):
            amp_data[i, j, :] = np.abs(
                np.cos(np.deg2rad(angles)) * (1 + 0.05 * rng.normal(size=len(angles)))
            )

    return xr.Dataset(
        {
            "amplitude": (["polarization", "frequency", "angle"], amp_data),
        },
        coords={
            "polarization": polarizations,
            "frequency": frequencies,
            "angle": angles,
        },
    )


# -----------------------------------------------------------------------------
# TESTE DE PIPELINE COMPLETO (Beam → BeamParams → DB → Recuperação)
# -----------------------------------------------------------------------------
def test_beam_repository_pipeline(repo, sample_beam_dataset):
    """Testa o fluxo completo: cálculo → persistência → recuperação."""

    # 1️⃣ Cria o objeto Beam com dataset
    beam = Beam(sample_beam_dataset)

    # 2️⃣ Calcula parâmetros físicos
    params = beam.params()
    assert isinstance(params, BeamParams) or hasattr(params, "directivity")
    assert params.directivity > 0

    # 3️⃣ Salva no banco via repositório
    saved = repo.add_beam_params(params)
    assert saved.id is not None

    # 4️⃣ Recupera do banco
    fetched = repo.get_beam_params_by_id(saved.id)
    assert fetched is not None
    assert fetched.directivity == pytest.approx(params.directivity)

    # 5️⃣ Converte de volta para domínio
    domain_obj = fetched.to_domain()
    assert np.isfinite(domain_obj.directivity)
    assert isinstance(domain_obj.fwhm, np.ndarray)

    # 6️⃣ Atualiza e verifica persistência
    params.directivity = 99.9
    updated = repo.update_beam_params(saved.id, params)
    assert updated.directivity == pytest.approx(99.9)

    # 7️⃣ Deleta e confirma exclusão
    deleted = repo.delete_by_id(saved.id)
    assert deleted is True
    assert repo.get_beam_params_by_id(saved.id) is None
