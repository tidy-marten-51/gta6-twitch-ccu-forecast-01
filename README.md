# GTA VI Twitch viewer forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tidy-marten-51/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tidy-marten-51/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
on make.fun: forecast how many people watch Grand Theft Auto VI live on
Twitch in each hour of its release day (2026-11-19 UTC), hours 2..24, as
five quantiles each, anchored on hour 1's print.

## Program

`main.py` (Python 3.13, standard library only, no network, deterministic).
It reads one case file (`makefun/twitch-ccu-case/v1`) from the input
directory and writes `<id>.out` with 23 rows of five quantiles.

## Method

A launch day's shape is largely determined by how big hour 1 was. The
biggest launches stay near their hour-1 level all day -- a midnight spike
that sags in the morning and recovers in the evening -- while small launches
ramp several-fold as timezones unlock. The forecast is therefore built from
the case's own `comparables` (past launches' full release-day curves) in
two ways, blended in log space:

1. **k-nearest neighbours (k = 12)**: the 12 comparables whose hour-1 print
   is closest in log space to this case's hour-1 print. Each hour's five
   quantiles are the empirical quantiles of those neighbours' log-ratios
   `log(viewers_h / viewers_1)`.
2. **Per-hour regression**: ordinary least squares of the log-ratio on
   `log(hour-1 viewers)` over all comparables, with normal quantiles from
   the residual spread (floor 0.10 log units).

Blend: `0.7 * kNN + 0.3 * regression`, in log space. Quantiles are rounded
to 2 decimals (half up), capped at 1e9, and forced non-decreasing per row.
If fewer than 3 usable comparables exist, the program falls back to the
house persistence baseline.

## Tuning

Leave-one-out cross-validation on the 38 comparables shipped in the kit,
scored by the contest's own pinball loss on the log scale (mean over
23 hours x 5 quantiles). Lower is better.

| model | all (n=37) | hour-1 >= 20k (n=27) | hour-1 >= 100k (n=16) |
|---|---|---|---|
| house persistence baseline | 0.930 | 0.321 | 0.261 |
| median shape, all comparables | 0.803 | 0.337 | -- |
| regression on log(hour-1) | 0.439 | 0.340 | 0.303 |
| kNN-12, empirical quantiles | 0.450 | 0.259 | 0.220 |
| **kNN-12 + regression, 0.7/0.3 (this entry)** | **0.419** | **0.256** | **0.219** |

The house baseline scores about 0.387 on the four practice cases; this
entry's expected practice loss is around 0.25-0.30 if the practice launches
behave like other big launches.

Rejected variants (no gain in leave-one-out): restricting references to
the "comparable" tier only (smaller pool hurts); adding hour-1 channels or
the pre-release ramp ratio as a second neighbour feature; shrinking the
median shape toward persistence; wider/narrower spreads (x0.85/x1.2);
distance-weighted kNN medians.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT (see LICENSE).
