"""
Pacote: radioskysims_beam

RadioBeam do ecossistema RadioSkySim — domínio de feixes de antena,
loaders (analytical/csv/custom/lit), repositório em DB e painel de
visualização (PanelApplication).

Copyright BINGO Collaboration
"""

from __future__ import annotations

__version__ = "0.1.0"

from .application.beam_panel import BeamPanel
from .domain.beam import Beam, BeamOps, BeamProperties

__all__ = ["Beam", "BeamOps", "BeamPanel", "BeamProperties"]
