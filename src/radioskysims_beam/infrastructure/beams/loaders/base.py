
"""
Módulo: src/radioskysims/infrastructure/beams/loaders/base.py

Descrição:
    Define a classe base abstrata para carregadores de feixes.
    Todos os loaders de beam devem implementar este contrato.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from radioskysims_beam.application.ports.beam import BeamLoaderPort
from radioskysims_beam.domain.beam import Beam


class BeamLoader(BeamLoaderPort, ABC):
    """
    Classe base abstrata para carregadores de padrões de feixe.

    Define o contrato que todos os loaders de beam devem implementar,
    permitindo carregar feixes de diferentes fontes, como CSV, LIT
    e formatos customizados.
    """

    @abstractmethod
    def load(self) -> Beam:
        """
        Carrega e retorna um objeto Beam de domínio.

        Returns:
            Beam: Objeto de domínio que representa o padrão de feixe.

        Raises:
            NotImplementedError: Quando o método não é implementado
                por uma subclasse concreta.
        """
        raise NotImplementedError

