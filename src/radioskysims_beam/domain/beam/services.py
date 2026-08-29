# domain/beam/services.py
from __future__ import annotations

import numpy as np

from radioskysims_beam.domain.beam.models import Beam
from radioskysims_core.logging_config import get_logger

logger = get_logger(__name__)


# =====================================================
# 🔧 Operações diretas sobre o Beam (normalização, corte etc.)
# =====================================================


class BeamOperation:
    """
    Serviços para manipulação do Beam (normalização, recorte, simetrização, etc.)
    """

    @staticmethod
    def normalize(beam: Beam) -> Beam:
        try:
            logger.info("beam_normalize_start", beam_dims=dict(beam.dataset.sizes))
            ds = beam.dataset.copy()
            ds["amplitude"] /= ds["amplitude"].max()
            logger.info("beam_normalize_success")
            return Beam(ds)
        except Exception as e:
            logger.exception("beam_normalize_failed", error=str(e))
            raise

    @staticmethod
    def slice_angular_range(beam: Beam, min_angle: float, max_angle: float) -> Beam:
        try:
            logger.info(
                "beam_slice_start",
                min_angle_deg=min_angle,
                max_angle_deg=max_angle,
            )
            ds = beam.dataset.sel(angle=slice(min_angle, max_angle))
            logger.info("beam_slice_success", output_points=len(ds.angle))
            return Beam(ds)
        except Exception as e:
            logger.exception(
                "beam_slice_failed",
                min_angle=min_angle,
                max_angle=max_angle,
                error=str(e),
            )
            raise

    @staticmethod
    def interpolate_angular_range(beam: Beam, new_angles: np.ndarray) -> Beam:
        try:
            logger.info(
                "beam_interpolate_start",
                original_points=len(beam.dataset.angle),
                new_points=len(new_angles),
            )
            ds = beam.dataset.interp(angle=new_angles)
            logger.info("beam_interpolate_success", new_points=len(ds.angle))
            return Beam(ds)
        except Exception as e:
            logger.exception(
                "beam_interpolate_failed",
                num_new_angles=len(new_angles),
                error=str(e),
            )
            raise

    @staticmethod
    def symmetrize(beam: Beam) -> Beam:
        ds = beam.dataset.copy()
        a = ds["amplitude"]
        ds["amplitude"] = (a + a.isel(angle=slice(None, None, -1))) / 2
        return Beam(ds)
