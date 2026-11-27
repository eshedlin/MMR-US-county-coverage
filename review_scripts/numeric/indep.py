"""Independent reimplementation (pure csv + decimal-safe arithmetic, numpy only for OLS)."""
import csv, sys, statistics as st, collections, math
import numpy as np
D = sys.argv[1]
rows = list(csv.DictReader(open(f"{D}/mmr_data_us_counties_v2.csv", encoding="utf-8-sig")))
src = list(csv.DictReader(open(f"{D}/mmr_data_sources_v2.csv", encoding="utf-8-sig")))
YRS = [c for c in rows[0] if c.startswith("SY")]
print("rows", len(rows), "states", len({r['State'] for r in rows}), "years", YRS)

# --- comparable states, from documentation, my own reading
def norm(x): return " ".join((x or "").split()).lower()
two_dose = {"2 mmr", "2+ mmr", "up to date (2 mmr)", "2 measles"}
data_states = {r['State'] for r in rows}
comp, why = [], {}
for s in src:
    a = s['Abbreviation'].strip(); m = norm(s['Vaccination Type Provided']); ag = norm(s['Age Group']); su = norm(s['Spatial Unit'])
    if m in two_dose:
        if a not in data_states: why[a] = "2-dose but no data"; continue
        if ag != "kindergarten": why[a] = f"age {ag}"; continue
        if not su.startswith("county"): why[a] = f"unit {su}"; continue
        comp.append(a)
print("2-dose states with data:", len([s for s in src if norm(s['Vaccination Type Provided']) in two_dose and s['Abbreviation'] in data_states]))
print("comparable", len(comp), sorted(comp)); print("excluded 2-dose:", why)
print("data states without doc row:", data_states - {s['Abbreviation'].strip() for s in src})

# --- values
nstar = collections.Counter(); val = {}
fips_len = collections.Counter(); missing_fips = []
for i, r in enumerate(rows):
    f = r['FIPS'].strip()
    fips_len[len(f)] += 1
    if not f: missing_fips.append((r['State'], r['County']))
    v = {}
    for y in YRS:
        x = r[y].strip()
        if '*' in x: nstar[r['State']] += 1; v[y] = None; continue
        v[y] = float(x) * 100 if x else None
    r['_v'] = v; r['F5'] = f.zfill(5) if f else ""
print("starred", sum(nstar.values()), dict(nstar)); print("FIPS lengths", dict(fips_len), "missing", len(missing_fips), {s for s, _ in missing_fips})
print("4-digit by state", collections.Counter(r['State'] for r in rows if len(r['FIPS'].strip()) == 4))
dup = collections.Counter(r['F5'] for r in rows if r['F5']); print("dup FIPS5", {k: c for k, c in dup.items() if c > 1})
# state prefix check
bad = [(r['State'], r['F5']) for r in rows if r['F5'] and r['F5'][:2] not in PREF.get(r['State'], (r['F5'][:2],))] if (PREF := {}) else []
dupname = collections.Counter((r['State'], r['County']) for r in rows); print("dup state+county", {k: c for k, c in dupname.items() if c > 1})
# star raw check: values with star - what do they look like
print("sample starred", [ (r['State'], y, r[y]) for r in rows for y in YRS if '*' in r[y]][:5])
nonnum = [(r['State'], y, r[y]) for r in rows for y in YRS if r[y].strip() and '*' not in r[y] and not r[y].strip().replace('.', '', 1).isdigit()]
print("non-numeric non-star", nonnum[:5]); 
rng_ = [r['_v'][y] for r in rows for y in YRS if r['_v'][y] is not None]; print("value range", min(rng_), max(rng_))

# --- missing data % (whole file vs others)
def miss(rs, yrs):
    n = len(rs) * len(yrs); m = sum(r['_v'][y] is None for r in rs for y in yrs); return m, n
m, n = miss(rows, YRS); print(f"missing all rows x 8 yrs (stars as missing): {m}/{n} = {100*m/n:.1f}%")
m2 = sum(not r[y].strip() for r in rows for y in YRS); print(f"  blank only (stars not missing): {m2}/{n} = {100*m2/n:.1f}%")
print("  by year:", [round(100 * sum(r['_v'][y] is None for r in rows) / len(rows), 1) for y in YRS])
print("  by year blank-only:", [round(100 * sum(not r[y].strip() for r in rows) / len(rows), 1) for y in YRS])

# --- panel: CT planning regions out
def is_ct_region(r): return r['State'] == 'CT' and all(r['_v'][y] is None for y in YRS[:-1])
P = [r for r in rows if not is_ct_region(r)]
print("CT regions removed", sum(map(is_ct_region, rows)), "CT county rows with 2024-25 value:", [r['County'] for r in P if r['State']=='CT' and r['_v']['SY2024_25'] is not None])
C = [r for r in P if r['State'] in comp]
print("counties in comparable states", len(C))
m, n = miss(C, YRS); print(f"missing in comparable states, 8 yrs: {m}/{n}={100*m/n:.1f}%"); m, n = miss(C, YRS[1:]); print(f"  7 yrs: {m}/{n}={100*m/n:.1f}%")
PRE = ["SY2018_19", "SY2019_20"]; TR = ["SY2020_21", "SY2021_22", "SY2022_23"]
def mean(xs): xs = [x for x in xs if x is not None]; return sum(xs) / len(xs) if xs else None
def lastcol(r): return "SY2023_24" if r['State'] == 'CT' else "SY2024_25"
def feats(r, pre=PRE, trough="min"):
    b = mean([r['_v'][y] for y in pre]); tv = [r['_v'][y] for y in TR if r['_v'][y] is not None]
    t = (min(tv) if tv else None) if trough == "min" else (mean(tv) if trough == "mean" else (sorted(tv)[1] if len(tv) >= 2 else None))
    return b, t, r['_v'][lastcol(r)]
def cls(b, t, l, thr, rnd=None):
    if None in (b, t, l): return None
    d, dt = l - b, t - b
    if rnd is not None: d, dt = round(d, rnd), round(dt, rnd)
    if d >= thr: return "up"
    if d <= -thr: return "down"
    if dt <= -thr: return "rec"
    return "stable"
ORD = ["stable", "rec", "down", "up"]
def shares(cs):
    cs = [c for c in cs if c]; n = len(cs); cn = collections.Counter(cs)
    return n, {k: round(100 * cn[k] / n, 1) for k in ORD}
for r in C:
    r['b'], r['t'], r['l'] = feats(r); r['c'] = cls(r['b'], r['t'], r['l'], 1.0)
    r['npre'] = sum(r['_v'][y] is not None for y in PRE)
K = [r for r in C if r['c']]
print("classified", len(K), "states w/ classified", len({r['State'] for r in K}), sorted({r['State'] for r in K}))
print("comparable states w/o classified", sorted(set(comp) - {r['State'] for r in K}))
print("unclassified by state", collections.Counter(r['State'] for r in C if not r['c']))
print("classified baselines from ONE pre-year:", sum(r['npre'] == 1 for r in K), "of", len(K), collections.Counter(r['State'] for r in K if r['npre'] == 1))
print("MAIN", shares([r['c'] for r in K]))
for k_ in ORD:
    g = [r for r in K if r['c'] == k_]; print(f"  {k_}: n={len(g)} med base {st.median(r['b'] for r in g):.1f} med last {st.median(r['l'] for r in g):.1f} %base<95 {100*sum(r['b']<95 for r in g)/len(g):.0f}")
print("THR2 float", shares([cls(r['b'], r['t'], r['l'], 2.0) for r in C]))
print("THR2 rounded6", shares([cls(r['b'], r['t'], r['l'], 2.0, 6) for r in C]))
print("THR1 rounded6", shares([cls(r['b'], r['t'], r['l'], 1.0, 6) for r in C]))
print("THR1.5 r6", shares([cls(r['b'], r['t'], r['l'], 1.5, 6) for r in C]))
ties = [(r['State'], r['County'], r['b'], r['l'], r['t']) for r in C if None not in (r['b'], r['t'], r['l']) and (abs(abs(r['l']-r['b'])-2) < 1e-9 or abs(abs(r['t']-r['b'])-2) < 1e-9)]
print("exact |diff|=2 ties:", len(ties), ties[:8])
ties1 = [r for r in C if None not in (r['b'], r['t'], r['l']) and (abs(abs(r['l']-r['b'])-1) < 1e-9 or abs(abs(r['t']-r['b'])-1) < 1e-9)]
print("exact |diff|=1 ties:", len(ties1), "changed by rounding at thr1:", sum(cls(r['b'],r['t'],r['l'],1.0)!=cls(r['b'],r['t'],r['l'],1.0,6) for r in ties1))
s17 = [cls(*feats(r, ["SY2017_18"] + PRE), 1.0) for r in C]; print("INCL 2017-18", shares(s17))
print("  incl17 n changed class vs main:", sum(a != r['c'] for a, r in zip(s17, C)))
# strict baseline: both years required
strict = [cls(r['b'], r['t'], r['l'], 1.0) if r['npre'] == 2 else None for r in C]; print("STRICT both pre years", shares(strict), len({r['State'] for r, c in zip(C, strict) if c}))
print("TROUGH mean", shares([cls(*feats(r, trough="mean"), 1.0) for r in C]))
print("TROUGH 2nd", shares([cls(*feats(r, trough="2nd"), 1.0) for r in C]))
print("THR2+mean r6", shares([cls(*feats(r, trough="mean"), 2.0, 6) for r in C]), "float", shares([cls(*feats(r, trough="mean"), 2.0) for r in C]))
up = [r for r in K if r['c'] == 'up']; print("up dipped>=1:", sum(r['t'] - r['b'] <= -1 for r in up), "of", len(up))
rec = [r for r in K if r['c'] == 'rec']; print("rec still rec w/ 2nd lowest:", sum(cls(*feats(r, trough="2nd"), 1.0) == 'rec' for r in rec), "of", len(rec))

# by state
bys = collections.defaultdict(list)
for r in K: bys[r['State']].append(r['c'])
print("by state %down:", sorted(((round(100*c.count('down')/len(c)), s, len(c)) for s, c in bys.items()), reverse=True))
def band(b): return "<90" if b < 90 else ("90-95" if b < 95 else "95+")
bb = collections.defaultdict(list)
for r in K: bb[band(r['b'])].append(r['c'])
for k_, c in bb.items(): print("band", k_, len(c), {x: round(100*c.count(x)/len(c)) for x in ORD})
print("up with base<95:", round(100*sum(r['b']<95 for r in up)/len(up)))
# CT, ME
for r in K:
    if r['State'] == 'CT': print("CT", r['County'], round(r['b'],2), round(r['t'],2), round(r['l'],2), r['c'], round(r['l']-r['b'],2))
ME = [r for r in P if r['State'] == 'ME']
mec = []
for r in ME:
    b, t, l = feats(r); c = cls(b, t, l, 1.0); mec.append(c)
    first_up = next((y for y in TR + ["SY2023_24", "SY2024_25"] if r['_v'][y] is not None and b is not None and r['_v'][y] - b >= 1), None)
    print("ME", r['County'], None if b is None else round(b,1), None if t is None else round(t,1), l, c, "first>=+1:", first_up)
print("ME", collections.Counter(mec), len(ME))
medS = {s: (st.median(r['l'] for r in K if r['State']==s) - st.median(r['b'] for r in K if r['State']==s)) for s in bys}
print("CT median(last)-median(base)", round(medS['CT'],2), "others median of state medians", round(st.median(v for s, v in medS.items() if s != 'CT'),2))
print("CT median of county change", st.median(r['l']-r['b'] for r in K if r['State']=='CT'), "others county-level median change", round(st.median(r['l']-r['b'] for r in K if r['State']!='CT'),2))

# --- controlled comparison: on classified counties (my choice) and on base+last only (their choice)
REP = {"CT", "CA"}
def ctrl(S, label):
    print("== controlled:", label, "n", len(S))
    for bn in ["<90", "90-95", "95+"]:
        for rp in (True, False):
            g = [r['l'] - r['b'] for r in S if band(r['b']) == bn and (r['State'] in REP) == rp]
            print(f"  {bn} repeal={rp}: med {st.median(g):+.2f} n={len(g)}")
    def ols(S2, treat):
        X = np.array([[1, r['b'] - 95, 1.0 if r['State'] in treat else 0.0] for r in S2]); y = np.array([r['l'] - r['b'] for r in S2])
        return np.linalg.lstsq(X, y, rcond=None)[0]
    print("  repeal coef", round(ols(S, REP)[2], 3), "slope", round(ols(S, REP)[1], 3))
    for s in ["CT", "CA"]:
        print(f"  {s} alone", round(ols([r for r in S if r['State'] == s or r['State'] not in REP], {s})[2], 3))
    non = [r for r in S if r['State'] not in REP]
    pl = sorted((round(ols(non, {s})[2], 2), s) for s in {r['State'] for r in non})
    print("  placebo", len(pl), pl)
    return pl
BL = [r for r in C if r['b'] is not None and r['l'] is not None]
pl_their = ctrl(BL, "base & last non-missing (as robustness.py)")
pl_mine = ctrl(K, "classified only (base, trough, last)")
print("counties in BL not classified:", len(BL) - len(K), collections.Counter(r['State'] for r in BL if not r['c']))
# pre-COVID noise
pc = [r for r in C if r['_v']['SY2018_19'] is not None and r['_v']['SY2019_20'] is not None]
dd = [r['_v']['SY2019_20'] - r['_v']['SY2018_19'] for r in pc]
print("noise n", len(pc), "median|d|", round(st.median(map(abs, dd)), 2), ">=1", round(100*sum(abs(x)>=1 for x in dd)/len(dd)), ">=2", round(100*sum(abs(x)>=2 for x in dd)/len(dd)), ">=2 r6", round(100*sum(round(abs(x),6)>=2 for x in dd)/len(dd),1))
for bn in ["<90", "90-95", "95+"]:
    g = [abs(r['_v']['SY2019_20'] - r['_v']['SY2018_19']) for r in pc if band(mean([r['_v'][y] for y in PRE])) == bn]
    gs = [(r['_v']['SY2019_20'] - r['_v']['SY2018_19']) for r in pc if band(r['_v']['SY2018_19']) == bn]
    print("  noise band(mean)", bn, len(g), round(st.median(g), 2), "| signed by 2018-19 level band median", round(st.median(gs), 2))
pn = [r for r in pc if r['State'] not in REP]
x = np.array([r['_v']['SY2018_19'] for r in pn]); y = np.array([r['_v']['SY2019_20'] - r['_v']['SY2018_19'] for r in pn])
print("pre slope non-repeal", np.polyfit(x, y, 1).round(3), "n", len(pn), "corr", round(np.corrcoef(x, y)[0,1], 2))
xa = np.array([r['_v']['SY2018_19'] for r in pc]); ya = np.array(dd); print("pre slope all", np.polyfit(xa, ya, 1).round(3), "corr", round(np.corrcoef(xa, ya)[0,1],2))
# Repeal effect incl 2017-18
S17 = []
for r in C:
    b, t, l = feats(r, ["SY2017_18"] + PRE)
    if b is not None and l is not None: S17.append(dict(r, b=b, l=l))
X = np.array([[1, r['b'] - 95, 1.0 if r['State'] in REP else 0.0] for r in S17]); y = np.array([r['l'] - r['b'] for r in S17]); print("repeal incl17", round(np.linalg.lstsq(X, y, rcond=None)[0][2], 2))

# --- LOO
print("== LOO")
full = shares([r['c'] for r in K])[1]
mx = {k_: (0, None) for k_ in ORD}
for s in sorted({r['State'] for r in C}):
    sh = shares([r['c'] for r in K if r['State'] != s])[1]
    for k_ in ORD:
        dlt = sh[k_] - full[k_]
        if abs(dlt) > abs(mx[k_][0]): mx[k_] = (round(dlt, 1), s, sh[k_])
print("  max abs shift per class", mx)

# --- imputation
def scen(fn):
    out = []
    for r in C:
        v = fn(r)
        b = mean([v[y] for y in PRE]); tv = [v[y] for y in TR if v[y] is not None]; t = min(tv) if tv else None
        out.append(cls(b, t, v[lastcol(r)], 1.0))
    return shares(out)
def locf(r):
    v = dict(r['_v']); last = None
    for y in YRS:
        if v[y] is None: v[y] = last
        else: last = v[y]
    nxt = None
    for y in reversed(YRS):
        if v[y] is None: v[y] = nxt
        else: nxt = v[y]
    return v
def interp(r):
    v = dict(r['_v']); idx = [i for i, y in enumerate(YRS) if v[y] is not None]
    for i, y in enumerate(YRS):
        if v[y] is None and idx and idx[0] < i < idx[-1]:
            lo = max(j for j in idx if j < i); hi = min(j for j in idx if j > i)
            v[y] = v[YRS[lo]] + (v[YRS[hi]] - v[YRS[lo]]) * (i - lo) / (hi - lo)
    return v
med_sy = {}
for s in comp:
    for y in YRS:
        xs = [r['_v'][y] for r in C if r['State'] == s and r['_v'][y] is not None]; med_sy[(s, y)] = st.median(xs) if xs else None
def smed(r): return {y: r['_v'][y] if r['_v'][y] is not None else med_sy[(r['State'], y)] for y in YRS}
def bound(sign):
    def f(r):
        m = mean([r['_v'][y] for y in YRS])
        return {y: (r['_v'][y] if r['_v'][y] is not None else (None if m is None else min(100, m + 3 * sign))) for y in YRS}
    return f
for nm, fn in [("complete", lambda r: r['_v']), ("interp", interp), ("state-median", smed), ("locf", locf), ("low", bound(-1)), ("high", bound(1))]:
    print("  imp", nm, scen(fn))

# ---- extra checks
print("== extra")
print("TROUGH mean r6", shares([cls(*feats(r, trough="mean"), 1.0, 6) for r in C]))
def scen_r(fn, rnd):
    out = []
    for r in C:
        v = fn(r); b = mean([v[y] for y in PRE]); tv = [v[y] for y in TR if v[y] is not None]; t = min(tv) if tv else None
        out.append(cls(b, t, v[lastcol(r)], 1.0, rnd))
    return shares(out)
print("locf r6", scen_r(locf, 6))
# compare to rerun county classes
import csv as _c
HE = {"יציב": "stable", "ירידה והתאוששות": "rec", "ירידה שלא התאוששה": "down", "עלייה": "up"}
theirs = {(x['State'], x['County']): HE.get(x['cls']) for x in _c.DictReader(open("/home/claude/rerun/county_classes.csv", encoding="utf-8-sig"))}
diff = [(r['State'], r['County'], r['c'], theirs.get((r['State'], r['County']))) for r in C if r['c'] != theirs.get((r['State'], r['County']))]
print("county-level class mismatches vs rerun:", len(diff), diff[:5], "rerun rows", len(theirs))
# follow-up length sensitivity: everyone's last = 2023-24
S23 = [dict(r, l=r['_v']['SY2023_24']) for r in C if r['b'] is not None and r['_v']['SY2023_24'] is not None]
X = np.array([[1, r['b'] - 95, 1.0 if r['State'] in REP else 0.0] for r in S23]); y = np.array([r['l'] - r['b'] for r in S23])
print("repeal effect with last=2023-24 for all states", round(np.linalg.lstsq(X, y, rcond=None)[0][2], 2), "n", len(S23))
print("classes with last=2023-24 for all", shares([cls(r['b'], r['t'], r['_v']['SY2023_24'], 1.0) for r in C]))
# hand-check examples
for key in [("CT", "Windham"), ("ID", "Ada"), ("KS", "Lincoln"), ("OR", "Sherman"), ("OR", "Wasco")]:
    r = next(x for x in C if (x['State'], x['County']) == key); print("hand", key, r['b'], r['t'], r['l'], r['c'], theirs.get(key))
upx = [r for r in K if r['c'] == 'up' and r['State'] == 'TX'][:2]
for r in upx: print("up TX", r['County'], r['F5'], {y: r[y] for y in YRS}, r['b'], r['t'], r['l'], theirs.get((r['State'], r['County'])))
# precision of raw values per state (decimals)
prec = collections.defaultdict(collections.Counter)
for r in C:
    for y in YRS:
        x = r[y].strip().rstrip('*')
        if x: prec[r['State']][len(x.split('.')[1]) if '.' in x else 0] += 1
print("decimals in raw by state", {s: dict(c) for s, c in prec.items()})
# capped states VT/WI: share of classified with base>=95 or last==95
for s in ["VT", "WI"]:
    g = [r for r in K if r['State'] == s]; print(s, "classified", len(g), "last==95", sum(abs(r['l'] - 95) < 1e-9 for r in g), "any capped yr", sum(any(r['_v'][y] is not None and abs(r['_v'][y]-95) < 1e-9 for y in YRS) for r in g))
# placebo rank of CT/CA among 22 states (my own computation)
print("LOO without repeal states effect on down share", shares([r['c'] for r in K if r['State'] not in REP]))
