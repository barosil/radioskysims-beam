"""
Módulo: src/radioskysims/beam/factory.py

Descrição:
    Fábrica de feixes LEGADO. Mantida para compatibilidade com código
    existente. Delega a geração de feixes analíticos para o novo
    AnalyticalBeamLoader em infrastructure/beams/loaders/.

    Novos desenvolvimentos devem usar AnalyticalBeamLoader diretamente.

Copyright BINGO Collaboration
Last modified: 2026-07-26
"""

from __future__ import annotations

import os

import numpy as np
import xarray as xr

from radioskysims_beam.domain.beam.models import Beam
from radioskysims_beam.infrastructure.beams.loaders.analytical import (
    AnalyticalBeamLoader,
)


class BeamFactory:
    """
    Fábrica de feixes LEGADO.

    Cria instâncias de Beam a partir de arquivos ou funções matemáticas.
    A geração de feixes analíticos (predefined) delega para
    AnalyticalBeamLoader.
    """

    @staticmethod
    def create_beam_from_file(
        filepath: str,
        name: str = "",
        is_db: bool = False,
        angular_resolution: float | None = None,
    ) -> Beam:
        """
        Cria um feixe a partir de arquivo contendo colunas [freq, ângulo, intensidade].

        Parameters
        ----------
        filepath : str
            Caminho do arquivo.
        name : str
            Nome do feixe (ignorado, mantido para compatibilidade).
        is_db : bool
            Se True, converte intensidade de dB para escala linear (potência).
        angular_resolution : float, optional
            Se fornecida, aplica interpolação na resolução angular desejada.

        Returns
        -------
        Beam
            Objeto feixe instanciado.
        """
        _ = name  # mantido para compatibilidade (ignorado)
        if not os.path.exists(filepath):
            msg = f"Arquivo não encontrado: {filepath}"
            raise FileNotFoundError(msg)

        try:
            data = np.loadtxt(filepath)
        except Exception as e:
            msg = "Erro ao ler arquivo. Verifique o formato (3 colunas: freq, ângulo, intensidade)."
            raise ValueError(msg) from e

        if data.shape[1] != 3:
            msg = "Arquivo deve conter exatamente 3 colunas: Frequência, Ângulo, Intensidade"
            raise ValueError(msg)

        if is_db:
            data[:, 2] = 10 ** (data[:, 2] / 10)

        unique_freqs = np.unique(data[:, 0])
        if len(unique_freqs) > 1:
            msg = "Atualmente, apenas uma frequência é suportada por feixe."
            raise ValueError(msg)

        freq = unique_freqs[0]

        if angular_resolution is not None:
            theta_min, theta_max = np.min(data[:, 1]), np.max(data[:, 1])
            new_theta = np.arange(theta_min, theta_max, angular_resolution)
            interpolated_intensity = np.interp(new_theta, data[:, 1], data[:, 2])
            angles = new_theta
            intensity = interpolated_intensity
        else:
            angles = data[:, 1]
            intensity = data[:, 2]

        ds = xr.Dataset(
            {
                "amplitude": (
                    ("angle", "frequency", "polarization"),
                    intensity[:, None, None],
                )
            },
            coords={
                "angle": angles,
                "frequency": [freq],
                "polarization": ["H"],
            },
        )
        return Beam(ds)

    @staticmethod
    def create_predefined_beam(
        name: str = "",
        beam_type: str = "gaussian",
        frequency: float = 1.0,
        angular_resolution: float = 1.0,
    ) -> Beam:
        """
        Gera feixes teóricos simulados (gaussian, airy, tapered_cosine).

        Delega para AnalyticalBeamLoader.

        Parameters
        ----------
        name : str
            Nome do feixe (ignorado, mantido para compatibilidade).
        beam_type : str
            Tipo do feixe ('gaussian', 'airy', 'tapered_cosine').
        frequency : float
            Frequência do feixe em Hz.
        angular_resolution : float
            Resolução angular em graus.

        Returns
        -------
        Beam
            Objeto Beam instanciado.
        """
        _ = name  # mantido para compatibilidade (ignorado)
        loader = AnalyticalBeamLoader(
            beam_profile=beam_type,
            frequency=frequency,
            angular_resolution=angular_resolution,
        )
        return loader.load()
