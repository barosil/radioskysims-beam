"""
Módulo: src/radioskysims/infrastructure/beams/loaders/analytical.py

Descrição:
    Implementa carregador de feixes analíticos (gaussiano, Airy, cosseno
    tangencial) a partir de parâmetros matemáticos. Gera padrões de feixe
    teóricos para testes e simulações.

Copyright BINGO Collaboration
Last modified: 2026-07-26
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import xarray as xr

from radioskysims_beam.domain.beam import Beam

from ._utils import finalize_beam_dataset
from .base import BeamLoader


class BeamProfile(str, Enum):
    """Perfis de feixe analítico suportados."""

    GAUSSIAN = "gaussian"
    AIRY = "airy"
    TAPERED_COSINE = "tapered_cosine"


class AnalyticalBeamLoader(BeamLoader):
    """
    Carregador que gera feixes analíticos a partir de parâmetros
    matemáticos (gaussiano, Airy, cosseno tangencial).

    Attributes:
        beam_profile (BeamProfile | str): Tipo de perfil de feixe.
        frequency (float): Frequência em Hz.
        angular_range (tuple[float, float]): Intervalo angular (min, max) em graus.
        angular_resolution (float): Resolução angular em graus.
        polarization (str): Polarização (H, V, etc).
        sigma (float | None): Largura do gaussiano (graus). Usado apenas
            para perfil gaussiano. Padrão: 10.0.
    """

    def __init__(
        self,
        beam_profile: BeamProfile | str = BeamProfile.GAUSSIAN,
        *,
        frequency: float | list[float] = 1.0,
        angular_range: tuple[float, float] = (-90.0, 90.0),
        angular_resolution: float = 1.0,
        polarization: str = "H",
        sigma: float | None = None,
    ) -> None:
        self.beam_profile = (
            BeamProfile(beam_profile) if isinstance(beam_profile, str) else beam_profile
        )
        self.frequency = [frequency] if isinstance(frequency, (int, float)) else list(frequency)
        self.angular_range = angular_range
        self.angular_resolution = angular_resolution
        self.polarization = polarization
        self.sigma = sigma

    def load(self) -> Beam:
        """
        Gera um Beam analítico baseado no perfil configurado.

        Returns:
            Beam: Objeto Beam com o padrão de feixe analítico.

        Raises:
            ValueError: Se perfil de feixe for desconhecido.
        """
        theta = np.arange(
            self.angular_range[0],
            self.angular_range[1] + self.angular_resolution,
            self.angular_resolution,
        )

        intensity = self._compute_profile(theta)
        intensity = intensity / np.max(intensity)

        n_freqs = len(self.frequency)
        amplitude = intensity[:, None, None]  # shape (angle, 1, 1)
        if n_freqs > 1:
            amplitude = np.repeat(amplitude, n_freqs, axis=1)

        ds = xr.Dataset(
            {
                "amplitude": (
                    ("angle", "frequency", "polarization"),
                    amplitude,
                )
            },
            coords={
                "angle": theta,
                "frequency": self.frequency,
                "polarization": [self.polarization],
            },
        )

        ds["angle"].attrs["units"] = "deg"
        ds["frequency"].attrs["units"] = "Hz"

        ds = finalize_beam_dataset(ds)
        return Beam(ds)

    def _compute_profile(self, theta: np.ndarray) -> np.ndarray:
        """Calcula a intensidade do feixe para o perfil configurado."""
        profile_map = {
            BeamProfile.GAUSSIAN: self._gaussian,
            BeamProfile.AIRY: self._airy,
            BeamProfile.TAPERED_COSINE: self._tapered_cosine,
        }

        func = profile_map.get(self.beam_profile)
        if func is None:
            msg = f"Perfil de feixe desconhecido: {self.beam_profile}"
            raise ValueError(msg)

        return func(theta)

    def _gaussian(self, theta: np.ndarray) -> np.ndarray:
        """Padrão gaussiano: exp(-0.5 * (theta / sigma)^2)."""
        sigma = self.sigma if self.sigma is not None else 10.0
        return np.exp(-0.5 * (theta / sigma) ** 2)

    def _airy(self, theta: np.ndarray) -> np.ndarray:
        """Padrão Airy: (2 * J1(x) / x)^2 onde x = theta_rad * pi."""
        from scipy.special import j1

        x = np.deg2rad(theta) * np.pi
        with np.errstate(invalid="ignore", divide="ignore"):
            intensity = (2 * j1(x) / x) ** 2
        intensity[np.isnan(intensity)] = 1.0
        return intensity

    def _tapered_cosine(self, theta: np.ndarray) -> np.ndarray:
        """Padrão cosseno tangencial: cos(theta_rad)^2."""
        return np.cos(np.deg2rad(theta)) ** 2
