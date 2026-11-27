# Synthesis of review round 1 (as sent to the user, translated from Hebrew), 2026-11-26

Round-1 reports were from three reviewers: numeric reproduction (/home/claude/review/numeric/), methodology (/home/claude/review/method/), statistics (/home/claude/review/stats/). Their scripts are in those folders. Background: /home/claude/review/latest_findings.md and /home/claude/proj/mmr_project/HANDOFF.md (Hebrew). Data: /home/claude/data. County outputs: /home/claude/rerun/county_classes.csv.

## Claims the lead analyst says were personally verified
1. 635 of 1,431 classified counties (44%) have a baseline from 2019-20 only (TX 251, KS 104, TN 93, OK 77, AL 67, WA 38, others 5). Requiring both years: 796 counties, 16 states, shares stable 5.2 / dip+rec 8.7 / down 69.7 / up 16.5.
2. Shifted null: every county shifts -2.9 points plus noise and RTM -> down 76.5, up 10.7, dip+rec 10.7, stable 2.1 (observed 70.7 / 13.5 / 10.4 / 5.4). Conclusion stated to the user: "the four classes add little beyond one common decline".
3. Independent banding (band on 2018-19, change last minus 2019-20): <90 median -4.0, 90-95 -3.2, 95+ -3.0. Conclusion: "the 'up' in low counties is regression to the mean; no evidence of real recovery".
4. CA state-median county went 91.6 (2018-19) -> 96.6 (2019-20); CO +3.0; all other states <=1.5. Source documentation gives no explanation.
5. With 2019-20-only baseline: CA effect 7.1 -> 4.33; largest placebo 4.18; CT 6.54.
6. CT shorter follow-up: with 2023-24 as last year for all, repeal effect 6.27 instead of 6.99.
7. Median change -2.9, state-clustered bootstrap 95% CI -3.6 to -2.2; Conley-Taber-style repeal interval 3.1 to 13.1; CR1 CI 5.4 to 8.6 must not be reported.

## Recommendations given to the user (decisions still his)
- Withdrew "fixed 2-point threshold". Reason given: under pure noise thr 2 still puts 22% of counties in "down not recovered"; noise SD ~5 points below 90 and ~1.5 at 95+.
- Primary outcome: continuous change (median with CI) vs the pre-COVID one-year change; also "79% of counties declined" and "median county declined in 19 of 22 states".
- Classes secondary, with a threshold by baseline band and number of baseline years; fixed 1 and 2 as sensitivity; honest range for "down not recovered" 47-71%.
- "Dip+recovered" not distinguishable from noise (pure-noise null 18-20%); use mean of COVID years or require dip in >=2 of 3 years.
- Repeal chapter: drop causal language; CT as descriptive single-state case, timing consistent with Public Act 21-6 (preschool exemptions expired Sept 2022, rise from 2022-23); vs neighbours MA and VT, CT's margin shrinks to ~2-2.6 points. CA unusable as a repeal case (repealed 2016, unexplained jump).
- Literature: state-level staggered DiD (Bald, Gold & Yang, JAMA Pediatrics, online 27 Oct 2025, doi 10.1001/jamapediatrics.2025.4185) found total repeal associated with +2 to 4 points; county level adds distribution, not identification. Author order and issue not verified.
- Extra data issues: VT and WI cap at 95; Sherman and Wasco OR share one series; OK is an optional survey; TN's only baseline year was weighted by the dataset authors; Maine rise from 2020-21 holds for 10 of 14; 8 excluded states, not 7.

## Pending decisions for the user
Primary outcome; baseline rule (both years / 2019-20 only / mixed with disclosure); CA and CO handling; keep or drop the repeal chapter.
