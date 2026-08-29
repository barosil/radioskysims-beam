"""
Módulo: src/radioskysims/application/ports/beam.py

Ports (Interfaces) da camada de Application para operacoes de Beam.

Define o contrato entre o domínio e as implementações concretas de
manipulação de feixes de radiotelescópios.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import xarray as xr

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


class BeamEvaluatorPort(ABC):
    """
    Port para avaliação e interpolação de feixes.

    Define o contrato para calcular amplitudes de beam em
    frequências e ângulos específicos.
    """

    @abstractmethod
    def evaluate(
        self,
        beam: Beam,
        *,
        frequency: xr.DataArray,
        angle: xr.DataArray,
        polarization: str | None = None,
    ) -> xr.DataArray:
        """
        Avalia o feixe B(freq, angle) em pontos específicos.

        Args:
            beam: Entidade Beam de domínio.
            frequency: DataArray com frequências (GHz ou Hz).
            angle: DataArray com ângulos (radianos).
            polarization: Polarização específica (H, V, etc; opcional).

        Returns:
            xr.DataArray: Amplitude do beam nos pontos requisitados.

        Raises:
            ValueError: Se dimensões incompatíveis.
            NotImplementedError: Se polarização não suportada.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_properties(self, beam: Beam) -> dict[str, Any]:
        """
        Computa propriedades físicas do beam.

        Calcula FWHM, directividade, eficiência de lóbulo principal, etc.

        Args:
            beam: Entidade Beam de domínio.

        Returns:
            dict: Dicionário com propriedades calculadas:
                - fwhm: Largura a meia altura
                - directivity: Directividade em dBi
                - main_lobe_efficiency: Eficiência do lóbulo principal
                - etc.

        Raises:
            ValueError: Se dados insuficientes.
        """
        raise NotImplementedError


class BeamCreatorPort(ABC):
    """
    Port para criação de estruturas de beam.

    Define o contrato para sintesis/criação de dados de beam
    a partir de parâmetros ou modelos analíticos.
    """

    @abstractmethod
    def create(
        self,
        *,
        frequency: list[float],
        angle: list[float],
        amplitudes: Any,
        polarizations: list[str] | None = None,
    ) -> Beam:
        """
        Cria um novo Beam com dados especificados.

        Args:
            frequency: Array de frequências.
            angle: Array de ângulos.
            amplitudes: Dados de amplitude do beam.
            polarizations: Lista de polarizações (opcional).

        Returns:
            Beam: Entidade Beam criada.

        Raises:
            ValueError: Se formas incompatíveis.
        """
        raise NotImplementedError
