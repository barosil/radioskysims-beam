"""
Módulo: src/radioskysims/domain/beam/operations.py

Descrição:
    Operações de transformação e manipulação de feixes (Beam).
    Re-exporta BeamOperation de services.py como BeamOps para
    compatibilidade retroativa.
"""

from __future__ import annotations

from .services import BeamOperation as BeamOps

__all__ = ["BeamOps"]
