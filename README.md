# radioskysims-beam

RadioBeam do ecossistema **RadioSkySim** — feixes de antena.

## Conteúdo

- `domain.beam` — `Beam`, `BeamOps`, `BeamProperties`, factory, value objects
- `infrastructure.beams.loaders` — `AnalyticalBeamLoader`, `CSVBeamLoader`,
  `CustomBeamLoader`, `LITBeamLoader`
- `infrastructure.db` — `beam_repository` + `beam_db_model`
- `application.ports.beam` — `BeamLoaderPort`/`BeamEvaluatorPort`/`BeamCreatorPort`
- `application.beam_panel` — `BeamPanel` (subclasse de `PanelApplication`)

## Dependências

Leves: `numpy`, `xarray`, `pandas`, `sqlmodel`, `panel`, `holoviews`, `hvplot`,
`radioskysims-core`, `radioskysims-viz`. **Sem** `dask`, `pyccl`, `camb`,
`psycopg2`.

## Testes

```bash
uv run --package radioskysims-beam pytest
```
