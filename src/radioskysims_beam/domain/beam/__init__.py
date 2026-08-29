"""  
Módulo: src/radioskysims/domain/beam/__init__.py

Descrição:
    Pacote de domínio para modelagem de feixes (beams) de radiotelescópios.
    Contém modelos, operações e propriedades físicas de padrões de radiação.
"""
from __future__ import annotations

from .models import Beam
from .operations import BeamOps
from .properties import BeamProperties

__all__ = ["Beam", "BeamOps", "BeamProperties"]
