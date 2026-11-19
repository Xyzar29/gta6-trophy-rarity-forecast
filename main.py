#!/usr/bin/env python3
"""Forecast GTA VI PS5 trophy earn rates (Exophase) at a case target.

Reads every ``*.in`` case file in $MAKEFUN_INPUT_DIR (else /input) and writes
one ``<id>.out`` forecast per case to $MAKEFUN_OUTPUT_DIR (else /output).

Each case gives the house's reads of Exophase's GTA VI trophy page before a
checkpoint: per trophy, the percent of tracked players who have earned it.
The forecast is five quantiles (5/25/50/75/95%) of each trophy's earn rate
at the case target, scored by pinball loss on the logit scale.

Model (per trophy, all on the logit scale):
  * Fit logit(rate) against ln(1 + hours since release) by least squares
    over the last 36 h of reads (the recent regime is what extrapolates;
    launch-night reads are a different, distorted regime: a frozen page
    of 638 early players, then one big refresh). Anchor the
    forecast at the last observed rate plus the fitted slope's movement
    to the target, so a real level shift at the last refresh is kept.
  * Shrink each trophy's slope toward its tier's mean slope (empirical
    Bayes): noisy per-trophy slopes, especially from few distinct reads,
    are pulled toward the cross-trophy pattern; flat histories shrink
    to a zero slope.
  * Spread: a logit-normal predictive band whose sigma combines a
    structural floor, a random-walk term growing with the horizon, the
    slope's standard error times the extrapolation distance, and the
    residual scatter around the fit (refresh jumps live there). Erring
    wide costs less than erring narrow under pinball loss.
  * A completely flat history means Exophase has not refreshed the page
    for hours while a just-released game is being played: a refresh is
    pending, and at a refresh the tracked-player denominator jumps while
    earners lag, so rates drop. Shift the median down and widen the
    band. (Practice histories are exactly this shape; the final's
    48 h history will not be, so this branch cannot fire there.)
  * A trophy sitting at 100% can only fall as new players join who have
    not earned it yet: forecast its non-earner share instead, growing
    with the horizon.

Standard library only. Deterministic: no randomness, no wall-clock use,
no dict-order dependence. Runs in well under a second per case.
"""

import json
import math
import os
import sys

# Standard-normal quantiles at the five scored levels.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# --- Tuned constants (calibrated on simulated launch-week trajectories;
#     see the build notes in README.md) -----------------------------------
S0 = 0.30            # structural logit floor: answer-read staleness etc.
RW = 0.30            # sigma_rw = RW * sqrt(horizon in days)
SHRINK_K = 10.0      # slope weight w = n_distinct / (n_distinct + SHRINK_K)
WINDOW_HOURS = 36.0  # fit window: last N hours of reads
SLOPE_MIN = -1.0     # clamp on shrunk logit slope per log-hour
SLOPE_MAX = 5.0
FLAT_SHIFT = 1.00    # logit shift down when the history is fully flat
FLAT_SHIFT_HOURS = 6.0  # shift scales with freeze length up to this
FLAT_SIGMA = 1.30    # spread when the history is fully flat
DAMP = 0.85          # damping on the extrapolated slope movement: the
                     # log-time curve flattens, but real progression
                     # saturates faster than log-time once the early
                     # surge passes, so the raw extrapolation is damped
U_MED = 0.60         # median non-earner share (%) at full horizon, 100% trophy
U_MULTS = (0.06, 0.30, 1.0, 2.5, 6.0)  # non-earner-share quantile multiples
U_HORIZON_DAYS = 4.5  # horizon at which the 100% band reaches full width
TIER_FALLBACK = {"platinum": 0.5, "gold": 1.0, "silver": 2.0, "bronze": 5.0}


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(rate):
    """logit of a percent rate, clamped so 0% and 100% stay finite."""
    p = clamp(rate / 100.0, 0.0001, 0.9999)
    return math.log(p / (1.0 - p))


def sigmoid(logit):
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    e = math.exp(logit)
    return e / (1.0 + e)


def parse_time(text):
    """ISO 8601 UTC timestamp -> epoch seconds (float)."""
    return _parse_dt(text).timestamp()


def _parse_dt(text):
    from datetime import datetime, timezone
    s = text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _quantiles_from_logit(median_logit, sigma):
    out = []
    for z in Z:
        q = sigmoid(median_logit + z * sigma) * 100.0
        out.append(round(clamp(q, 0.01, 99.99), 4))
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def _ceiling_band(horizon_days):
    """Quantiles for a trophy at 100%: forecast the non-earner share."""
    f = min(1.0, horizon_days / U_HORIZON_DAYS)
    u = [U_MED * f * m for m in U_MULTS]
    qs = [round(clamp(100.0 - x, 0.01, 99.99), 4) for x in reversed(u)]
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def _ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_sd, se_b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return my, 0.0, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        resid_sd = math.sqrt(rss / (n - 2))
    else:
        resid_sd = 0.0
    se_b = resid_sd / math.sqrt(sxx) if sxx > 0.0 else 0.0
    return a, b, resid_sd, se_b


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history", [])
    horizon_days = case["horizon_minutes"] / 1440.0

    release = case.get("game", {}).get("release_date", "2026-11-19")
    t0 = parse_time(release + "T00:00:00Z") if len(release) == 10 else parse_time(release)

    target_t = parse_time(case["target"])
    x_target = math.log1p(max(0.0, (target_t - t0) / 3600.0))

    # Per-trophy series: (hours since release, rate) for every non-null read.
    series = []
    for pos, tr in enumerate(trophies):
        pts = []
        for read in history:
            rates = read.get("rates") or []
            if 0 <= pos < len(rates):
                r = rates[pos]
            else:
                r = None
            if r is None:
                continue
            t = parse_time(read["at"])
            pts.append((max(0.0, (t - t0) / 3600.0), float(r)))
        series.append(pts)

    # First pass: raw per-trophy slopes, for tier-mean shrinkage targets.
    raw_slopes = []
    for pts in series:
        if len(set(r for _, r in pts)) < 2 or len(pts) < 2:
            raw_slopes.append(None)
            continue
        xs = [math.log1p(t) for t, _ in pts]
        ys = [logit_pct(r) for _, r in pts]
        raw_slopes.append(_ols(xs, ys)[1])
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        vals = [b for b, tr in zip(raw_slopes, trophies)
                if b is not None and tr.get("tier") == tier]
        if vals:
            tier_slopes[tier] = sum(vals) / len(vals)
    global_slope = 0.0
    vals = [b for b in raw_slopes if b is not None]
    if vals:
        global_slope = sum(vals) / len(vals)

    rows = []
    for tr, pts in zip(trophies, series):
        tid = tr["id"]
        if not pts:
            # No reads at all: tier prior level, very wide band.
            level = TIER_FALLBACK.get(tr.get("tier"), 2.0)
            rows.append({"id": tid,
                         "q": _quantiles_from_logit(logit_pct(level), 1.5)})
            continue
        last_t, last_r = pts[-1]
        x_last = math.log1p(last_t)
        dx = max(0.0, x_target - x_last)

        if last_r >= 99.99:
            rows.append({"id": tid, "q": _ceiling_band(horizon_days)})
            continue

        distinct = len(set(r for _, r in pts))
        if distinct <= 1:
            # Frozen page: a refresh is pending, rates drop at refreshes.
            first_t = pts[0][0]
            target_h = max(0.0, (target_t - t0) / 3600.0)
            freeze = target_h - first_t
            shift = FLAT_SHIFT * min(1.0, freeze / FLAT_SHIFT_HOURS)
            med = logit_pct(last_r) - shift
            rows.append({"id": tid,
                         "q": _quantiles_from_logit(med, FLAT_SIGMA)})
            continue

        # Fit window: last WINDOW_HOURS of reads, else all reads.
        window = [(t, r) for (t, r) in pts if last_t - t <= WINDOW_HOURS]
        if len(window) < 3 or len(set(r for _, r in window)) < 2:
            window = pts
        xs = [math.log1p(t) for t, _ in window]
        ys = [logit_pct(r) for _, r in window]
        _a, b, resid_sd, se_b = _ols(xs, ys)

        # Shrink the slope toward the tier mean.
        w = distinct / (distinct + SHRINK_K)
        tier_b = tier_slopes.get(tr.get("tier"), global_slope)
        b_shrunk = clamp(w * b + (1.0 - w) * tier_b, SLOPE_MIN, SLOPE_MAX)

        med = logit_pct(last_r) + b_shrunk * dx * DAMP
        sigma = math.sqrt(
            S0 * S0
            + (RW * math.sqrt(horizon_days)) ** 2
            + (se_b * dx) ** 2
            + (resid_sd * math.sqrt(1.0 + dx)) ** 2
        )
        rows.append({"id": tid, "q": _quantiles_from_logit(med, sigma)})

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(forecast, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
