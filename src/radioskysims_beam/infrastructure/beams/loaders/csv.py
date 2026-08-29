"""
Módulo: src/radioskysims/infrastructure/beams/loaders/csv.py

Descrição:
    Implementa o carregamento de padrões de feixe a partir de arquivos CSV.
    Os dados tabulares são validados, normalizados e convertidos em um
    objeto Beam de domínio.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from radioskysims_beam.domain.beam import Beam
from radioskysims_core.paths import get_project_root

from ._utils import finalize_beam_dataset
from .base import BeamLoader

_VARNAMES: dict[str, str] = {
    "polarization": "polarization",
    "frequency": "frequency",
    "angle": "angle",
    "amplitude": "amplitude",
    "phase": "phase",
}

_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "frequency",
        "angle",
        "amplitude",
    }
)


class CSVBeamLoader(BeamLoader):
    """
    Carrega padrões de feixe armazenados em arquivos CSV.

    O arquivo pode conter as colunas de polarização, frequência, ângulo,
    amplitude e fase. As colunas de polarização e fase são opcionais e
    recebem valores padrão quando não estiverem presentes.

    Attributes:
        file: Caminho para o arquivo CSV.
        varnames: Mapeamento entre os nomes canônicos do domínio e os
            nomes das colunas no arquivo.
    """

    def __init__(
        self,
        file: Path | str | None = None,
        varnames: dict[str, str] | None = None,
    ) -> None:
        """
        Inicializa o carregador CSV.

        Args:
            file: Caminho para o arquivo CSV.
            varnames: Mapeamento customizado dos nomes das colunas.
                As chaves obrigatórias são: ``polarization``,
                ``frequency``, ``angle``, ``amplitude`` e ``phase``.

        Raises:
            ValueError: Se o mapeamento não contiver exatamente as
                chaves esperadas.
        """
        self.file = Path(file) if file is not None else None
        self.varnames = dict(varnames) if varnames is not None else dict(_VARNAMES)

        expected_keys = set(_VARNAMES)
        received_keys = set(self.varnames)

        if received_keys != expected_keys:
            missing = sorted(expected_keys - received_keys)
            unexpected = sorted(received_keys - expected_keys)

            details: list[str] = []

            if missing:
                details.append(f"chaves ausentes: {missing}")

            if unexpected:
                details.append(f"chaves inesperadas: {unexpected}")

            msg = f"varnames deve conter exatamente as chaves {sorted(expected_keys)}"

            if details:
                msg = f"{msg}; {', '.join(details)}"

            raise ValueError(msg)

    def load(self) -> Beam:
        """
        Carrega o padrão de feixe do arquivo CSV.

        Returns:
            Beam: Objeto de domínio contendo o padrão de feixe carregado.

        Raises:
            FileNotFoundError: Se nenhum caminho for informado ou se o
                arquivo não for encontrado.
            ValueError: Se alguma coluna obrigatória estiver ausente.
            pandas.errors.ParserError: Se o arquivo CSV estiver malformado.
        """
        path = self._resolve_path()

        dataframe = pd.read_csv(path)
        dataframe = self._normalize_columns(dataframe)

        dataset = dataframe.set_index(["polarization", "frequency", "angle"]).to_xarray()

        dataset = finalize_beam_dataset(dataset)

        return Beam(dataset)

    def _resolve_path(self) -> Path:
        """
        Resolve o caminho do arquivo CSV.

        Caminhos relativos são procurados primeiro em relação ao diretório
        de execução e, depois, em relação ao pacote ``radioskysims``.

        Returns:
            Path: Caminho absoluto para um arquivo existente.

        Raises:
            FileNotFoundError: Se nenhum caminho for informado ou se o
                arquivo não existir.
        """
        if self.file is None:
            msg = "Caminho do arquivo CSV não informado."
            raise FileNotFoundError(msg)

        path = self.file.expanduser()

        if not path.is_absolute() and not path.exists():
            # package_root = src/radioskysims (compatibilidade legada)
            package_root = get_project_root() / "src" / "radioskysims"
            project_root = package_root.parent.parent  # → raiz do projeto
            if path.parts and path.parts[0] == "radioskysims":
                candidate = package_root.parent / path
            else:
                # Tenta: src/radioskysims/path → src/../path → raiz/path
                candidate = project_root / path
            if candidate.exists():
                path = candidate

        if path.exists():
            return path.resolve()

        package_root = get_project_root() / "src" / "radioskysims"

        if path.parts and path.parts[0] == "radioskysims":
            package_candidate = package_root.parent / path
        else:
            package_candidate = package_root / path

        if package_candidate.exists():
            return package_candidate.resolve()

        msg = f"Arquivo CSV de beam não encontrado: {self.file}"
        raise FileNotFoundError(msg)

    def _normalize_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Valida e normaliza as colunas do arquivo CSV.

        Args:
            dataframe: Dados tabulares lidos do arquivo.

        Returns:
            pd.DataFrame: DataFrame com nomes canônicos e valores padrão.

        Raises:
            ValueError: Se alguma coluna obrigatória estiver ausente.
        """
        dataframe = dataframe.copy()

        required_source_columns = {self.varnames[column] for column in _REQUIRED_COLUMNS}

        missing_columns = sorted(required_source_columns - set(dataframe.columns))

        if missing_columns:
            msg = (
                "O arquivo CSV não contém todas as colunas obrigatórias. "
                f"Colunas ausentes: {missing_columns}"
            )
            raise ValueError(msg)

        polarization_column = self.varnames["polarization"]
        phase_column = self.varnames["phase"]

        if polarization_column not in dataframe.columns:
            dataframe[polarization_column] = "UNDEFINED"

        if phase_column not in dataframe.columns:
            dataframe[phase_column] = 0.0

        rename_mapping = {
            source_name: canonical_name for canonical_name, source_name in self.varnames.items()
        }

        return dataframe.rename(columns=rename_mapping)
