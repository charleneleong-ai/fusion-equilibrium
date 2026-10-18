import json, re, sys, pathlib, csv
D = pathlib.Path(sys.argv[1]); raw = D/'raw'
subs = [s for p in (1,2,3) for s in json.load(open(raw/f'page{p}.json'))['submissions']]
def cells(html):
    return [re.sub(r'<[^>]+>','',c).strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', html, re.S)]
def parse_machine(sec):
    c = cells(sec); out = {}
    for i in range(len(c)-1):
        k = c[i]
        m = {'R2_psi (flux, pooled)':'r2psi6','R2_qb (q95 + betaN mean)':'r2qb6','D_LCFS':'dlcfs6',
             'Consistency (7 psi-derived scalars)':'cons6','R2_psi per-shot mean':'r2psi_shot','flux sign applied':'sign',
             'LCFS fail fraction (prediction side)':'lcfs_fail','derivation fail fraction (prediction side)':'deriv_fail',
             'psi non-finite fraction':'nonfinite','shots scored / missing':'shots','S_model':'S6',
             'q95':'q95','betaN':'betaN','R_axis':'c_Raxis','Z_axis':'c_Zaxis','kappa':'c_kappa','tri_top':'c_tritop',
             'tri_bot':'c_tribot','volume':'c_vol','li':'c_li'}.get(k)
        if m and m not in out: out[m] = c[i+1]
    return out
rows = []
for s in subs:
    r = {'id': s['id'], 'owner': s['owner'], 'created': s['created_when'], 'org': (s.get('organization') or {}).get('name') if isinstance(s.get('organization'),dict) else s.get('organization'),
         'method': (s['fact_sheet_answers'] or {}).get('method_name',''), 'code_url': (s['fact_sheet_answers'] or {}).get('code_url',''),
         'compute_light': (s['fact_sheet_answers'] or {}).get('compute_light','')}
    for sc in s['scores']: r[sc['column_key']] = float(sc['score'])
    h = (raw/'details'/f"{s['id']}.html")
    if h.exists():
        html = h.read_text()
        parts = re.split(r'<h2>(DIII-D|MAST)</h2>', html)
        for i in range(1, len(parts), 2):
            pre = 'd3d_' if parts[i]=='DIII-D' else 'mast_'
            for k, v in parse_machine(parts[i+1]).items(): r[pre+'det_'+k] = v
        notes = re.findall(r'<p[^>]*>(.*?)</p>', html, re.S)
        r['notes'] = ' / '.join(re.sub(r'<[^>]+>','',n).strip() for n in notes)[:300]
    rows.append(r)
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(D/'leaderboard_all.csv','w',newline='') as f:
    w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
json.dump(subs, open(raw/'leaderboard_all_submissions.json','w'), indent=1)
print(len(rows), len({r['owner'] for r in rows})); print(keys)
