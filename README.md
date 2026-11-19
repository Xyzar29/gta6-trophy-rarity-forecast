# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Xyzar29/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Xyzar29/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for GTA VI's PlayStation 5 trophy earn rates on Exophase. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

Given a case file of the house's reads of Exophase's GTA VI trophy page
before a checkpoint, `main.py` forecasts five quantiles (5/25/50/75/95%)
of each of the game's 55 trophies' earn rate at the case target, scored by
pinball loss on the logit scale.

## How it forecasts

- **Per-trophy log-time curves.** For each trophy, logit(earn rate) is
  fitted by least squares against ln(1 + hours since release) over the
  last 36 hours of reads — the recent regime is what extrapolates, not
  the distorted launch night (a frozen page of 638 early players, then
  one big refresh). The forecast is anchored at the last observed rate
  plus the fitted slope's movement to the target, damped by 0.85,
  because real progression saturates faster than log-time once the
  early surge passes.
- **Shrinkage toward tier means.** Each trophy's slope is pulled toward
  its tier's mean slope (weight by how many distinct rates were seen),
  so noisy per-trophy fits from few refreshes do not wag the forecast.
- **Honest spread.** The logit-normal band's sigma combines a structural
  floor, a random-walk term growing with the horizon, the slope's
  standard error times the extrapolation distance, and the residual
  scatter around the fit (refresh jumps live there).
- **Frozen page? Expect a refresh.** A completely flat history means
  Exophase has not refreshed the page for hours while a just-released
  game is being played: a refresh is pending, and at a refresh the
  tracked-player denominator jumps while earners lag, so rates drop.
  The median shifts down and the band widens.
- **A trophy at 100% can only fall** as new players join who have not
  earned it yet, so its non-earner share is forecast instead, growing
  with the horizon.

## Calibration

Tuned on simulated launch-week trajectories (log-linear, saturating,
surging, late-drop and linear-time truths; practice refreshes of
several sizes), 40 runs each: mean pinball loss **0.28** vs the house
baseline's 0.36 and the house curve baseline's 0.43 on final-like
cases, and **0.52** vs the house baseline's 1.51 on practice-like
cases. Deterministic, standard library only, ~0.05 s per case.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`):
exactly `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}` per trophy.

---

<sub>Entered on [make.fun](https://make.fun/?ref=86zc8pnt&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
