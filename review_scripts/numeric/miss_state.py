import csv, collections, sys
rows = list(csv.DictReader(open("/home/claude/data/mmr_data_us_counties_v2.csv", encoding="utf-8-sig")))
YRS = [c for c in rows[0] if c.startswith("SY")]
for star_missing in (False, True):
    out = {}
    for s in sorted({r['State'] for r in rows}):
        R = [r for r in rows if r['State'] == s]; n = m = 0
        for y in YRS:
            present = [bool(r[y].strip()) and not (star_missing and '*' in r[y]) for r in R]
            if any(present): n += len(R); m += present.count(False)
        out[s] = round(100*m/n, 1) if n else None
    print("stars missing" if star_missing else "stars present", {s: v for s, v in out.items() if v is None or v > 5})
print("WV rows", sum(r['State']=='WV' for r in rows), "WV nonblank", sum(1 for r in rows if r['State']=='WV' for y in YRS if r[y].strip()))
