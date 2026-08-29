"""
Conftest do pacote radioskysims-beam.

Reproduz os filtros de aviso do conftest raiz do monólito
(tests/conftest.py) para o ambiente isolado do pacote.
"""

from __future__ import annotations

import warnings

import pytest


def pytest_configure(config):
    # Recursos não fechados de libs de terceiros (Skyfield .bsp, openpyxl,
    # sqlite3) geram ResourceWarning no __del__ — promovidos a erro por
    # filterwarnings=error.
    warnings.filterwarnings(
        "ignore",
        category=pytest.PytestUnraisableExceptionWarning,
    )
    # DeprecationWarning do numpy.core (renomeado para numpy._core no NumPy 2.x)
    warnings.filterwarnings(
        "ignore",
        message=r"numpy\.core\.",
        category=DeprecationWarning,
    )
