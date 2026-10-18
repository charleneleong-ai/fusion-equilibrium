import pandas as pd, numpy as np
d=pd.read_pickle('/private/tmp/claude-502/scr/d.pkl')
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',200)
x=d[(d.d3d_S-d.d3d_Srec4).abs()>0.001][['owner','d3d_S','d3d_r2_psi','d3d_r2_qb','d3d_dlcfs','d3d_consistency','d3d_det_S6','notes']]; print(x.to_string()[:2000])
x=d[(d.mast_S-d.mast_Srec4).abs()>0.001][['owner','mast_S','mast_r2_psi','mast_r2_qb','mast_dlcfs','mast_consistency','mast_det_S6']]; print(x)
d['gcalc']=np.where(d.d3d_S6>=0.85, d.mast_S6/d.d3d_S6, 0)
print(d[(d.gcalc.fillna(0)-d.g_ratio).abs()>2e-4][['owner','d3d_S6','mast_S6','g_ratio','gcalc']])
# 4dp precision check
e=(d.d3d_S-d.d3d_Srec4).abs(); print('4dp recompute err quantiles', e[e<0.01].quantile([.5,.9,1]).to_dict())
# duplicates
key=['d3d_S6','d3d_r2psi6','d3d_cons6','mast_S6']
g=d[d.d3d_S6>0].groupby(key,dropna=False)['owner'].apply(list); print(g[g.apply(len)>1])
# clusters: count with identical rounded d3d_S
print(d.d3d_S.value_counts().head(15))
