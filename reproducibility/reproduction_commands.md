# Reproduction commands

Run commands from the repository root after installing the environment.

## 1. Validate configuration

```bash
python -c "from pathlib import Path; import json; p=json.loads(Path('configs/pilot_config.json').read_text()); print(p['project_root'], p['outputs'])"
```

## 2. Construct horizon-specific event-ranking inputs from harmonized data

The following command shows the Q98, H1 route after the harmonized train/test gauge lists and source archives have been placed under the relative paths in `configs/pilot_config.json`. Repeat with `--horizon 5` and `--horizon 7`, and with event quantiles 0.95 and 0.99 for the cross-threshold analysis.

```bash
python scripts/evaluate_horizon_q98_transfer.py --horizon 1
```

## 3. Evaluate the lead-adaptive policy

```bash
python scripts/evaluate_lead_adaptive_operational_policy.py --event-quantile 0.98 --horizons 1,5,7
```

## 4. Validate the archived overlap-excluded results and independent checks

```bash
python scripts/validate_primary_results.py --data-dir ../derived_artifacts/supporting_tables
```

This check verifies the final 981-gauge inventory, the protected-anchor comparison, nested leave-one-source-out stability, source-block intervals, episode-level gains and the post-lock LamaH-Ice confirmation set from the archived tables.

## 5. Rebuild the method figure and graphical abstract

```bash
python scripts/build_method_visuals.py --out-dir reproduced_figures
```

## 6. Rebuild result Figures 2–4

```bash
python scripts/build_main_figures.py --data-dir ../derived_artifacts/supporting_tables --out-dir reproduced_figures
```
