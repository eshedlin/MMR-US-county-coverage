# Latest findings and proposals (after HANDOFF.md), 2026-11-26

All paths are in /home/claude:
- proj/mmr_project/HANDOFF.md: full project summary (Hebrew). Sections 3-8 are the design, data issues, definitions and results.
- proj/mmr_project/analysis.py, robustness.py: the analysis code.
- data/: raw source files, freshly downloaded from https://raw.githubusercontent.com/CSSEGISandData/MMR_data/main/
  (mmr_data_us_counties_v2.csv = data, mmr_data_sources_v2.csv = per-state documentation; the repo README describes them the other way round).
- rerun/: outputs of a fresh rerun (python3 -I analysis.py data rerun; python3 -I robustness.py rerun).
- scripts/variants.py: definition variants and noise check.

## Rerun results
- All numbers in HANDOFF section 8 reproduced exactly, except:
  - States with classified counties: 22, not 23 (DC has one unit and is not classified). HANDOFF says "23 states".
  - Threshold-2 sensitivity: correct shares are stable 15.1 / up 9.9 (HANDOFF and in-memory analysis.py give 15.2 / 9.8). Cause: exact ties at the threshold (change exactly 2.0 points) evaluated as 1.9999... in floating point. Proposed fix: round differences to 6 decimals before comparing.

## Noise check (pre-COVID, one-year change 2018-19 -> 2019-20, all comparable states, n=806)
- Median absolute change 1.7 points; 70% of counties move >=1 point, 47% move >=2.
- Median absolute change by baseline: <90: 3.95; 90-95: 1.8; 95+: 1.1.

## Variants (shares %, n=1431)
| variant | stable | dip+recovered | down not recovered | up |
| main (thr 1, trough=min) | 5.4 | 10.4 | 70.7 | 13.5 |
| thr 1.5 | 10.2 | 12.3 | 65.6 | 11.9 |
| thr 2 | 15.1 | 14.4 | 60.7 | 9.9 |
| trough = mean of COVID years | 9.5 | 6.3 | 70.7 | 13.5 |
| trough = second lowest (n=1394) | 9.7 | 5.7 | 71.2 | 13.3 |
| baseline incl 2017-18 | 5.3 | 9.8 | 70.7 | 14.2 |
| thr 2 + trough mean | 22.3 | 7.2 | 60.7 | 9.9 |
- 76 of 193 "up" counties dipped >=1 point first.
- 69 of 149 "dip+recovered" counties depend on a single low year (not recovered under second-lowest trough).
- Repeal effect (baseline-adjusted regression): 6.99; with 2017-18 in baseline 7.74.
- CT under thr 1/1.5/2: never any "down not recovered".

## Proposals made to the user (not yet decided by him)
1. Threshold 2 points as primary (1 and 1.5 as sensitivity); report finding as range 61-71% not recovered.
2. Keep classification order; report "dipped then rose above baseline" as a sub-row of "up".
3. Keep trough = minimum, but flag "dip+recovered" as the least stable class.
4. Keep 2017-18 out of the main analysis.
5. Claim: repeal effect is independent of threshold choice because it is computed on continuous change.
