from __future__ import annotations

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, SQLModel, create_engine

from radioskysims_beam.domain.beam.models import BeamParamsDomain
from radioskysims_core.domain.common.exceptions import BeamRepositoryError
from radioskysims_beam.infrastructure.db.beam_db_model import BeamParamsDB
from radioskysims_beam.infrastructure.db.beam_repository import BeamRepository


# ------------------------------------------------------
# FIXTURES
# ------------------------------------------------------
@pytest.fixture
def in_memory_session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    SQLModel.metadata.create_all(engine)
    try:
        with Session(engine) as session:
            yield session
    finally:
        # Fecha o engine para evitar ResourceWarning (conexão sqlite órfã)
        engine.dispose()


@pytest.fixture
def repository(in_memory_session):
    return BeamRepository(in_memory_session)


@pytest.fixture
def sample_domain_params():
    return BeamParamsDomain(
        polarizations=["X", "Y"],
        frequencies=[100.0, 200.0],
        has_phase=True,
        fwhm=[1.0, 2.0],
        directivity=10.5,
        main_lobe_directivity=5.2,
        directivity_db=12.3,
        main_lobe_efficiency=0.85,
    )


# ------------------------------------------------------
# TESTE: ADD
# ------------------------------------------------------
def test_add_beam_params_success(repository, sample_domain_params):
    result = repository.add_beam_params(sample_domain_params)
    assert isinstance(result, BeamParamsDB)
    assert result.id is not None


def test_add_beam_params_error(repository, sample_domain_params, monkeypatch):
    def mock_add(*_args, **_kwargs):
        msg = "DB failure"
        raise SQLAlchemyError(msg)

    monkeypatch.setattr(repository.session, "add", mock_add)
    with pytest.raises(BeamRepositoryError):
        repository.add_beam_params(sample_domain_params)


# ------------------------------------------------------
# TESTE: GET BY ID
# ------------------------------------------------------
def test_get_beam_params_by_id_found(repository, sample_domain_params):
    saved = repository.add_beam_params(sample_domain_params)
    fetched = repository.get_beam_params_by_id(saved.id)
    assert fetched.id == saved.id


def test_get_beam_params_by_id_not_found(repository):
    assert repository.get_beam_params_by_id(999) is None


# ------------------------------------------------------
# TESTE: LIST_ALL
# ------------------------------------------------------
def test_list_all_success(repository, sample_domain_params):
    repository.add_beam_params(sample_domain_params)
    repository.add_beam_params(sample_domain_params)
    results = repository.list_all()
    assert len(results) == 2


# ------------------------------------------------------
# TESTE: UPDATE
# ------------------------------------------------------
def test_update_beam_params_success(repository, sample_domain_params):
    saved = repository.add_beam_params(sample_domain_params)
    updated_domain = BeamParamsDomain(
        polarizations=["A", "B"],
        frequencies=[300.0, 400.0],
        has_phase=False,
        fwhm=[5.0, 6.0],
        directivity=20.0,
        main_lobe_directivity=8.0,
        directivity_db=25.0,
        main_lobe_efficiency=0.90,
    )
    updated = repository.update_beam_params(saved.id, updated_domain)
    assert updated.directivity == 20.0


def test_update_beam_params_error(repository, sample_domain_params, monkeypatch):
    saved = repository.add_beam_params(sample_domain_params)

    def mock_commit(*_args, **_kwargs):
        msg = "Commit failed"
        raise SQLAlchemyError(msg)

    monkeypatch.setattr(repository.session, "commit", mock_commit)
    with pytest.raises(BeamRepositoryError):
        repository.update_beam_params(saved.id, sample_domain_params)


# ------------------------------------------------------
# TESTE: DELETE
# ------------------------------------------------------
def test_delete_by_id_success(repository, sample_domain_params):
    saved = repository.add_beam_params(sample_domain_params)
    assert repository.delete_by_id(saved.id) is True


def test_delete_by_id_error(repository, sample_domain_params, monkeypatch):
    saved = repository.add_beam_params(sample_domain_params)

    def mock_delete(*_args, **_kwargs):
        msg = "Delete failed"
        raise SQLAlchemyError(msg)

    monkeypatch.setattr(repository.session, "delete", mock_delete)
    with pytest.raises(BeamRepositoryError):
        repository.delete_by_id(saved.id)


# ------------------------------------------------------
# TESTE: DOMAIN CONVERSION
# ------------------------------------------------------
def test_get_as_domain_success(repository, sample_domain_params):
    saved = repository.add_beam_params(sample_domain_params)
    domain_obj = repository.get_as_domain(saved.id)
    assert isinstance(domain_obj, BeamParamsDomain)


def test_list_all_as_domain_success(repository, sample_domain_params):
    repository.add_beam_params(sample_domain_params)
    repository.add_beam_params(sample_domain_params)
    results = repository.list_all_as_domain()
    assert len(results) == 2


def test_list_all_as_domain_error(repository, monkeypatch):
    # Forçar erro em list_all
    def mock_list_all(*_args, **_kwargs):
        msg = "Conversion error"
        raise Exception(msg)

    monkeypatch.setattr(repository, "list_all", mock_list_all)
    with pytest.raises(BeamRepositoryError):
        repository.list_all_as_domain()
