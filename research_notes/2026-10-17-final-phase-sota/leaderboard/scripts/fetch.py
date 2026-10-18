import json, re, subprocess, sys, pathlib, time
raw = pathlib.Path(sys.argv[1]); out = raw/'details'; out.mkdir(exist_ok=True)
subs = [s for p in (1,2,3) for s in json.load(open(raw/f'page{p}.json'))['submissions']]
def curl(u):
    for _ in range(3):
        r = subprocess.run(['curl','-s','--max-time','60',u],capture_output=True,text=True)
        if r.returncode==0 and r.stdout: return r.stdout
        time.sleep(2)
    return ''
for s in subs:
    f = out/f"{s['id']}.html"
    if f.exists(): continue
    try: url = json.loads(curl(f"https://www.codabench.org/api/submissions/{s['id']}/get_detail_result/"))
    except Exception: print('fail', s['id']); continue
    if not isinstance(url,str) or not url.startswith('http'): print('nourl', s['id'], url); continue
    html = re.sub(r"<img src='data:image/png;base64,[^']*'>", "", curl(url))
    f.write_text(html)
print(len(list(out.glob('*.html'))))
