"""Download the source files (not stored in this repository; the source repo has no licence file).

Usage: python get_data.py data
"""
import sys, pathlib, urllib.request

BASE = "https://raw.githubusercontent.com/CSSEGISandData/MMR_data/main/"
FILES = ["mmr_data_us_counties_v2.csv", "mmr_data_sources_v2.csv"]  # data, per-state documentation

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data")
out.mkdir(parents=True, exist_ok=True)
for f in FILES:
    urllib.request.urlretrieve(BASE + f, out / f)
    print("downloaded", f)
