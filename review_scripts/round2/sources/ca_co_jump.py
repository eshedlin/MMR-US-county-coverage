import csv, statistics as st, sys
path = sys.argv[1]
rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
yrs = ['SY2017_18','SY2018_19','SY2019_20','SY2020_21','SY2021_22','SY2022_23','SY2023_24','SY2024_25']
def f(v):
    try: return float(v)
    except: return None
for state in ['CA','CO']:
    sub = [r for r in rows if r['State']==state]
    print('=====', state, 'n counties', len(sub))
    for y in yrs:
        v=[f(r[y]) for r in sub if f(r[y]) is not None]
        if v: print(y, 'n',len(v),'median %.3f min %.3f max %.3f'%(st.median(v),min(v),max(v)))
    print('County, 17-18, 18-19, 19-20, 20-21, 24-25, delta(19-20 minus 18-19)')
    out=[]
    for r in sub:
        a,b,c=f(r['SY2017_18']),f(r['SY2018_19']),f(r['SY2019_20'])
        d = (c-b) if (b is not None and c is not None) else None
        out.append((d if d is not None else -9, r['County'], [r[y] for y in yrs]))
    out.sort()
    for d,c,v in out: print('%-16s'%c, v[0],v[1],v[2],v[3],v[7], '%.3f'%d if d!=-9 else 'NA')
    ds=[o[0] for o in out if o[0]!=-9]
    print('delta median %.3f, mean %.3f, n neg %d, n>0.03 %d, n>0.05 %d of %d'%(st.median(ds),st.mean(ds),sum(x<0 for x in ds),sum(x>0.03 for x in ds),sum(x>0.05 for x in ds),len(ds)))
    # correlation of delta with 2018-19 level
    pairs=[(f(r['SY2018_19']),f(r['SY2019_20'])-f(r['SY2018_19'])) for r in sub if f(r['SY2018_19']) is not None and f(r['SY2019_20']) is not None]
    if len(pairs)>2:
        x=[p[0] for p in pairs]; yv=[p[1] for p in pairs]
        mx,my=st.mean(x),st.mean(yv)
        cov=sum((a-mx)*(b-my) for a,b in pairs); sx=sum((a-mx)**2 for a in x)**.5; sy=sum((b-my)**2 for b in yv)**.5
        print('corr(level 18-19, delta) = %.2f'%(cov/(sx*sy)))
    # 17-18 vs 18-19 stability
    ds2=[f(r['SY2018_19'])-f(r['SY2017_18']) for r in sub if f(r['SY2018_19']) is not None and f(r['SY2017_18']) is not None]
    if ds2: print('17-18 -> 18-19 delta median %.3f'%st.median(ds2))
    ds3=[f(r['SY2020_21'])-f(r['SY2019_20']) for r in sub if f(r['SY2020_21']) is not None and f(r['SY2019_20']) is not None]
    if ds3: print('19-20 -> 20-21 delta median %.3f'%st.median(ds3))
