# DALA-Hydro

DALA-Hydro (Data-Anchored Lead-Adaptive Hydrology) is an
observation-governance layer for high-flow prediction. It retains a
meteorology-based protected anchor, admits issue-time discharge only when state
evidence clears that anchor by a positive margin, and contracts the admitted
increment as forecast lead increases. The rule is monotone, auditable and
portable above different forecasting models.

This repository accompanies the Journal of Hydrology manuscript
**“Data-anchored, lead-adaptive admission of river state improves extreme-flow
prediction across 981 basins.”**

## Important interpretation of the forcing experiment

The retrospective experiments use observed or reanalysis meteorology at the
target horizon as a perfect-forcing hindcast proxy. They do not include error
from an operational numerical-weather-prediction system. The scores therefore
test the incremental value and lead dependence of issue-time discharge under
controlled forcing information; they are not presented as an end-to-end
operational flood forecast.

## Persistent identifiers

- Repository: <https://github.com/lxxwgdxg/density-aware-event-memory>
- Software concept DOI: <https://doi.org/10.5281/zenodo.21277559>
- Previously archived software record: <https://doi.org/10.5281/zenodo.21277560>
- Derived-artifact concept DOI:
  <https://doi.org/10.5281/zenodo.21320182>
- Previously published derived-artifact version:
  <https://doi.org/10.5281/zenodo.21320183>

The concept DOI resolves to the latest software version after the v1.2.0 archive
is published. Zenodo will assign the new version-specific DOI during publishing.

## Repository structure

- `src/dememory/`: reusable data-catalogue, configuration and time-series
  utilities;
- `scripts/build_method_visuals.py`: reproduces manuscript Figure 1 and the
  graphical abstract;
- `scripts/build_main_figures.py`: reproduces result Figures 2–4;
- `scripts/`: policy evaluation and validation calculations;
- `configs/`: path-independent experiment and policy specifications;
- `metadata/`: public data-source manifests;
- `reproducibility/`: environment, command order, inputs and expected outputs;
- the separate derived-artifact archive supplies the numerical values for all
  four manuscript figures.

## Installation

```bash
conda env create -f environment.yml
conda activate dala_hydro_joh
```

or:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Figure-level reproduction

Assuming `code_repository/` and `derived_artifacts/` are extracted side by side:

```bash
python scripts/build_method_visuals.py --out-dir reproduced_figures
python scripts/build_main_figures.py \
  --data-dir ../derived_artifacts/supporting_tables \
  --out-dir reproduced_figures
```

The first command builds Figure 1 and the graphical abstract. The second builds
Figures 2–4 and `Result_Figure_Source_Data.xlsx`. The archived
`derived_artifacts/Source_Data.xlsx` is the authoritative integrated workbook
and additionally contains Figure 1 method constants.

## Data access

Raw CAMELS, Caravan, GRDC-Caravan and related source archives are not
redistributed. Provider URLs and persistent identifiers are listed in
`metadata/`. Users remain responsible for the original providers' licenses and
access terms. Figure-level reproduction is self-contained using the separate
derived-artifact archive; raw-data reconstruction remains provider-dependent.

## Citation and licenses

Use `CITATION.cff` for the software citation. Code is licensed under MIT. The
derived-artifact archive is licensed under CC BY 4.0; source datasets retain
their original licenses.
