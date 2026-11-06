"""
Módulo: src/radioskysims/application/ports/beam.py

Ports (Interfaces) da camada de Application para operacoes de Beam.

Define o contrato entre o domínio e as implementações concretas de
manipulação de feixes de radiotelescópios.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from radioskysims_beam.domain.beam.models import Beam


class BeamLoaderPort(ABC):
    """
    Port para carregamento de dados de feixes.

    Define o contrato para leitura de feixes de diferentes formatos
    e fontes (arquivos HDF5, NetCDF, banco de dados, etc).
    """

    @abstractmethod
    def load(self) -> Beam:
        """
        Carrega um Beam da fonte configurada no adapter.

        Returns:
            Beam: Entidade de domínio Beam.

        Raises:
            FileNotFoundError: Se fonte não existir.
            ValueError: Se formato inválido.
        """
        raise NotImplementedError
