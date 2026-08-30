"""
Módulo: src/radioskysims/application/beam_panel.py

Descrição:
    Implementa uma visualização interativa de um padrão de feixe gaussiano
    usando Panel, HoloViews e o modelo oficial Beam do domínio.
"""

from __future__ import annotations

import hvplot.xarray  # noqa: F401
import numpy as np
import panel as pn
import xarray as xr
from radioskysims_viz import PanelApplication

from radioskysims_beam.domain.beam.models import Beam

DEFAULT_FREQUENCY_HZ = 1.0e9
DEFAULT_POLARIZATION = "UNDEFINED"
DEFAULT_BEAM_NAME = "Gaussian Beam"


def gaussian_beam_dataset(
    frequency_hz: float,
    width_deg: float,
    amplitude: float,
    n_points: int = 181,
) -> xr.Dataset:
    """
    Cria um dataset de feixe gaussiano compatível com o domínio Beam.

    Args:
        frequency_hz:
            Frequência central do feixe em hertz.
        width_deg:
            Desvio padrão angular da gaussiana em graus.
        amplitude:
            Amplitude máxima do padrão.
        n_points:
            Quantidade de amostras angulares entre -90 e 90 graus.

    Returns:
        Dataset com dimensões ``polarization``, ``frequency`` e ``angle``,
        contendo as variáveis ``amplitude`` e ``phase``.

    Raises:
        ValueError:
            Se os parâmetros numéricos forem inválidos.
    """
    if frequency_hz <= 0:
        msg = "frequency_hz deve ser maior que zero."
        raise ValueError(msg)

    if width_deg <= 0:
        msg = "width_deg deve ser maior que zero."
        raise ValueError(msg)

    if amplitude < 0:
        msg = "amplitude não pode ser negativa."
        raise ValueError(msg)

    if n_points < 3:
        msg = "n_points deve ser maior ou igual a 3."
        raise ValueError(msg)

    angle_deg = np.linspace(-90.0, 90.0, n_points, dtype=float)

    beam_amplitude = amplitude * np.exp(-(angle_deg**2) / (2.0 * width_deg**2))

    amplitude_values = beam_amplitude[np.newaxis, np.newaxis, :]
    phase_values = np.zeros_like(amplitude_values)

    dataset = xr.Dataset(
        data_vars={
            "amplitude": (
                ("polarization", "frequency", "angle"),
                amplitude_values,
            ),
            "phase": (
                ("polarization", "frequency", "angle"),
                phase_values,
            ),
        },
        coords={
            "polarization": [DEFAULT_POLARIZATION],
            "frequency": [float(frequency_hz)],
            "angle": angle_deg,
        },
        attrs={
            "name": DEFAULT_BEAM_NAME,
            "model": "gaussian",
            "width_deg": float(width_deg),
        },
    )

    dataset["frequency"].attrs["units"] = "Hz"
    dataset["angle"].attrs["units"] = "deg"
    dataset["amplitude"].attrs["units"] = "dimensionless"
    dataset["phase"].attrs["units"] = "rad"

    return dataset


def create_gaussian_beam(
    frequency_hz: float,
    width_deg: float,
    amplitude: float,
    n_points: int = 181,
) -> Beam:
    """
    Cria um Beam de domínio com padrão gaussiano.

    Args:
        frequency_hz:
            Frequência central em hertz.
        width_deg:
            Desvio padrão angular em graus.
        amplitude:
            Amplitude máxima.
        n_points:
            Quantidade de pontos angulares.

    Returns:
        Beam de domínio.
    """
    dataset = gaussian_beam_dataset(
        frequency_hz=frequency_hz,
        width_deg=width_deg,
        amplitude=amplitude,
        n_points=n_points,
    )

    return Beam(dataset)


class BeamPanel(PanelApplication):
    """
    Painel interativo para visualização de um feixe gaussiano.

    A interface permite alterar a amplitude e a largura angular do padrão.
    Cada atualização cria um novo objeto Beam oficial do domínio.
    (Refatorado para herdar de ``PanelApplication`` — Fase 2.)
    """

    def build_sidebar(self):
        """Constrói os controles (sidebar) do painel."""
        self.amplitude = pn.widgets.FloatSlider(
            name="Amplitude",
            start=0.1,
            end=10.0,
            step=0.1,
            value=1.0,
        )

        self.width = pn.widgets.FloatSlider(
            name="Largura σ (graus)",
            start=1.0,
            end=30.0,
            step=0.5,
            value=10.0,
        )

        self.update_button = pn.widgets.Button(
            name="Atualizar Feixe",
            button_type="primary",
        )

        self.update_button.on_click(self._update_beam)

        return pn.Column(self.amplitude, self.width, self.update_button)

    def build_main(self):
        """Constrói a área principal (plot + metadados)."""
        self.beam = create_gaussian_beam(
            frequency_hz=DEFAULT_FREQUENCY_HZ,
            width_deg=self.width.value,
            amplitude=self.amplitude.value,
        )

        self.plot_pane = pn.pane.HoloViews(
            self._make_plot(),
            sizing_mode="stretch_width",
        )

        self.metadata_pane = pn.pane.Str(self._metadata_text())

        return pn.Column(self.plot_pane, self.metadata_pane)

    def _make_plot(self):
        """
        Gera o gráfico angular do feixe atual.

        Returns:
            Objeto HoloViews gerado a partir do DataArray de amplitude.
        """
        amplitude = self.beam.dataset["amplitude"].isel(
            polarization=0,
            frequency=0,
        )

        return amplitude.hvplot.line(
            x="angle",
            y="amplitude",
            title=f"Padrão do Feixe — {self.beam.dataset.attrs['name']}",
            xlabel="Ângulo (graus)",
            ylabel="Amplitude (a.u.)",
        )

    def _metadata_text(self) -> str:
        """
        Retorna uma representação textual dos parâmetros atuais.
        """
        frequency_hz = float(self.beam.dataset["frequency"].values[0])
        width_deg = float(self.beam.dataset.attrs["width_deg"])

        return (
            f"Beam: {self.beam.dataset.attrs['name']} | "
            f"Frequência: {frequency_hz / 1e9:.2f} GHz | "
            f"σ: {width_deg:.2f}° | "
            f"Amplitude: {self.amplitude.value:.2f}"
        )

    def _update_beam(self, event: object | None = None) -> None:
        """
        Recria o feixe quando os controles forem atualizados.

        Args:
            event:
                Evento emitido pelo botão do Panel. O valor não é utilizado.
        """
        del event

        self.beam = create_gaussian_beam(
            frequency_hz=DEFAULT_FREQUENCY_HZ,
            width_deg=self.width.value,
            amplitude=self.amplitude.value,
        )

        self.plot_pane.object = self._make_plot()
        self.metadata_pane.object = self._metadata_text()

    def __panel__(self) -> pn.Column:
        """Retorna o layout renderizável pelo Panel (herdado do base)."""
        return self.layout


if __name__ == "__main__":
    BeamPanel().serve(title="Beam Panel", show=True)
