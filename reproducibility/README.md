# Reproducibility guide

The release supports visual, figure-level and score-level reproduction.

## Method visual and graphical abstract

```bash
python scripts/build_method_visuals.py --out-dir reproduced_figures
```

This generates `Figure_1.pdf/png` and `Graphical_Abstract.pdf/png` directly
from fixed method definitions and evaluation inventory constants.

## Result-figure reproduction

```bash
python scripts/build_main_figures.py \
  --data-dir ../derived_artifacts/supporting_tables \
  --out-dir reproduced_figures
```

This regenerates manuscript Figures 2–4 from the supporting CSV tables without
requiring the raw source archives. It also writes
`Result_Figure_Source_Data.xlsx`. The integrated archived Source Data workbook
contains those result sheets plus Figure 1 method constants.

## Score-level reproduction

Reconstructing score-level results requires harmonized event frames produced
from the original hydrological archives. Relative paths are defined in
`configs/pilot_config.json`; the ordered calculations are listed in
`reproduction_commands.md`. The raw archives are not redistributed.

The experiments are retrospective perfect-forcing hindcasts: target-horizon
meteorological inputs come from observations or reanalysis, not an archived
operational weather forecast. Target discharge and event labels are excluded
from score construction.

All reported gains are computed relative to the protected anchor defined in
`configs/dala_policy.json`.
