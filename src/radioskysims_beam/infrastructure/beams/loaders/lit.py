from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from radioskysims_beam.domain.beam import Beam

from ._utils import finalize_beam_dataset
from .base import BeamLoader


class LITBeamLoader(BeamLoader):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> Beam:
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        sheets = pd.read_excel(self.path, sheet_name=None, header=None)
        records: list[pd.DataFrame] = []

        for pol_name, df_raw in sheets.items():
            header_row = self._find_header_row(df_raw)
            freq_row = max(header_row - 1, 0)

            header_vals = list(df_raw.iloc[header_row])
            # trata células mescladas na linha de frequência
            freq_vals = pd.Series(df_raw.iloc[freq_row]).ffill()

            angle_idx = next(
                (i for i, v in enumerate(header_vals) if self._is_angle_label(v)),
                None,
            )
            if angle_idx is None:
                continue

            # Procura pares adjacentes (Amplitude, Phase/Fase)
            pairs: list[tuple[int, int, str]] = []
            for i in range(angle_idx + 1, len(header_vals) - 1):
                if self._is_amplitude_label(header_vals[i]) and self._is_phase_label(
                    header_vals[i + 1]
                ):
                    freq_text = (
                        freq_vals.iloc[i] if pd.notna(freq_vals.iloc[i]) else freq_vals.iloc[i + 1]
                    )
                    freq_text = "" if pd.isna(freq_text) else str(freq_text).strip()
                    pairs.append((i, i + 1, freq_text))

            if not pairs:
                continue

            data_block = df_raw.iloc[header_row + 1 :, :].reset_index(drop=True)
            angle_series = pd.to_numeric(data_block.iloc[:, angle_idx], errors="coerce")
            pol = self._parse_polarization(str(pol_name))

            for amp_idx, phase_idx, freq_text in pairs:
                freq_ghz = self._parse_frequency_ghz(freq_text)
                freq_mhz = pd.to_numeric(freq_ghz, errors="coerce") * 1e3

                tmp = pd.DataFrame(
                    {
                        "polarization": pol,
                        "frequency": freq_mhz,
                        "angle": angle_series,
                        "amplitude": pd.to_numeric(data_block.iloc[:, amp_idx], errors="coerce"),
                        "phase": pd.to_numeric(data_block.iloc[:, phase_idx], errors="coerce"),
                    }
                )
                tmp = tmp.dropna(how="all", subset=["angle", "amplitude", "phase"])
                records.append(tmp)

        if not records:
            msg = (
                "Não foi possível identificar colunas Angle/Amplitude/Phase no arquivo "
                f"LIT: {self.path}"
            )
            raise ValueError(msg)

        beam_df = pd.concat(records, ignore_index=True)
        beam_df["frequency"] = pd.to_numeric(beam_df["frequency"], errors="coerce")
        beam_df["angle"] = pd.to_numeric(beam_df["angle"], errors="coerce")
        beam_df["amplitude"] = self._normalize_amplitude_by_frequency(
            beam_df["amplitude"],
            beam_df["frequency"],
        )
        beam_df = beam_df.dropna(subset=["polarization", "frequency", "angle"])

        ds = beam_df.set_index(["polarization", "frequency", "angle"]).to_xarray()

        ds = finalize_beam_dataset(ds)

        return Beam(ds)

    def _parse_polarization(self, key: str) -> str:
        key_u = key.upper()
        pol = None
        if "HORIZONTAL" in key_u:
            pol = "H"
        elif "VERTICAL" in key_u:
            pol = "V"
        elif key_u.strip() in {"H", "V"}:
            pol = key_u.strip()
        else:
            pol = key_u.strip() or "UNDEFINED"

        # Distingue copolar vs cross-polar
        if "CRUZADA" in key_u or "CROSS" in key_u:
            pol = f"{pol}_CROSS"
        elif "COPOLAR" in key_u or "COPOL" in key_u:
            pol = f"{pol}_COPOL"
        return pol

    def _normalize_text(self, x) -> str:
        return (
            str(x)
            .strip()
            .lower()
            .replace("â", "a")
            .replace("á", "a")
            .replace("ã", "a")
            .replace("ê", "e")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("õ", "o")
            .replace("ú", "u")
        )

    def _parse_frequency_ghz(self, freq_text: str):
        if not freq_text:
            return pd.NA
        match = re.search(r"([0-9]+(?:[\.,][0-9]+)?)\s*ghz", str(freq_text).lower())
        if not match:
            return pd.NA
        return float(match.group(1).replace(",", "."))

    def _is_angle_label(self, label: str) -> bool:
        normalized = self._normalize_text(label)
        return (
            ("θ" in str(label))
            or ("theta" in normalized)
            or ("angulo" in normalized)
            or ("angle" in normalized)
        )

    def _is_amplitude_label(self, label: str) -> bool:
        return "amplitude" in self._normalize_text(label)

    def _is_phase_label(self, label: str) -> bool:
        normalized = self._normalize_text(label)
        return ("fase" in normalized) or ("phase" in normalized)

    def _find_header_row(self, df_raw: pd.DataFrame) -> int:
        max_scan = min(len(df_raw), 40)
        for i in range(max_scan):
            vals = [v for v in df_raw.iloc[i].tolist() if pd.notna(v)]
            if not vals:
                continue
            has_angle = any(self._is_angle_label(v) for v in vals)
            has_amp = any(self._is_amplitude_label(v) for v in vals)
            has_phase = any(self._is_phase_label(v) for v in vals)
            if has_angle and has_amp and has_phase:
                return i
        return 1

    def _normalize_amplitude_by_frequency(
        self,
        amplitude: pd.Series,
        frequency: pd.Series,
    ) -> pd.Series:
        """Normaliza a amplitude em dB deslocando o pico de cada banda para 0 dB.

        O arquivo LIT armazena a amplitude em dB. A normalização padrão de um
        padrão de radiação NÃO é min-max (que distorce a forma relativa entre
        banda e banda), mas sim subtrair o máximo da banda: isso ajusta a
        amplitude máxima em dB para 0 dB, mantendo todos os demais valores
        relativos ao pico (≤ 0 dB).
        """

        def _peak_to_zero_db(group: pd.Series) -> pd.Series:
            s_max = group.max()
            if pd.isna(s_max):
                return group
            return group - s_max  # desloca o pico da banda para 0 dB

        return amplitude.groupby(frequency, dropna=False).transform(_peak_to_zero_db)
