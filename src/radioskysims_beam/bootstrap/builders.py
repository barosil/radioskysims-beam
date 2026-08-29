"""Builders de componentes do pacote (composition root — strategy).
Extraído de ``radioskysims.bootstrap.builders`` na Fase 5.
Copyright BINGO Collaboration
"""

from __future__ import annotations

from radioskysims_beam.infrastructure.beams.loaders.csv import CSVBeamLoader


def _load_beam_from_csv(
    file_path: str,
):
    return CSVBeamLoader(file=file_path).load()
