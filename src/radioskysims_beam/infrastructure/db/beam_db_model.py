"""
Módulo: src/radioskysims/infrastructure/db/beam_db_model.py

Descrição:
    Modelo SQLModel para persistência das propriedades de feixe (Beam) no banco
    de dados. Pertence à camada de infraestrutura e provê conversão
    bidirecional com a entidade de domínio BeamParamsDomain.

Copyright BINGO Collaboration
Last modified: 2026-02-26
"""

from __future__ import annotations

import json

import numpy as np
from sqlmodel import Field, SQLModel

from radioskysims_beam.domain.beam.models import BeamParamsDomain


class BeamParamsDB(SQLModel, table=True):  # type: ignore[call-arg, misc]
    """
    Modelo persistente que representa as propriedades calculadas de um feixe.

    Armazena as informações em formato serializado (JSON) no banco de dados.
    Esta classe pertence à camada de infraestrutura e não deve ser usada
    diretamente por código de domínio.
    """

    __tablename__ = "beamparamsdb"

    id: int | None = Field(default=None, primary_key=True)
    polarizations: str
    frequencies: str
    has_phase: bool
    fwhm: str
    directivity: float
    main_lobe_directivity: float
    directivity_db: float
    main_lobe_efficiency: float

    @classmethod
    def from_domain(cls, params: BeamParamsDomain) -> BeamParamsDB:
        """Converte do domínio para formato SQL (JSON serializado)."""

        def _to_json_serializable(value: object) -> list[object]:
            return value.tolist() if hasattr(value, "tolist") else value  # type: ignore[return-value]

        return cls(
            polarizations=json.dumps(_to_json_serializable(params.polarizations)),
            frequencies=json.dumps(_to_json_serializable(params.frequencies)),
            has_phase=params.has_phase,
            fwhm=json.dumps(_to_json_serializable(params.fwhm)),
            directivity=params.directivity,
            main_lobe_directivity=params.main_lobe_directivity,
            directivity_db=params.directivity_db,
            main_lobe_efficiency=params.main_lobe_efficiency,
        )

    def to_domain(self) -> BeamParamsDomain:
        """Converte do formato SQL para o domínio (Python puro)."""
        return BeamParamsDomain(
            polarizations=json.loads(self.polarizations),
            frequencies=json.loads(self.frequencies),
            has_phase=self.has_phase,
            fwhm=np.array(json.loads(self.fwhm)),
            directivity=self.directivity,
            main_lobe_directivity=self.main_lobe_directivity,
            directivity_db=self.directivity_db,
            main_lobe_efficiency=self.main_lobe_efficiency,
        )
