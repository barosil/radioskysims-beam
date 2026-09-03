# domain/beam/value_objects.py
from __future__ import annotations

from enum import Enum


class Polarization(Enum):
    H = "H"
    V = "V"
    HX = "HX"
    VX = "VX"
    UNDEFINED = "UNDEFINED"
