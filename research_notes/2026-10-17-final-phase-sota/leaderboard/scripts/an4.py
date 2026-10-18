import pandas as pd, numpy as np
d=pd.read_pickle('/private/tmp/claude-502/scr/d.pkl')
pd.set_option('display.width',250)
print(d.iloc[30:36][['rank','owner','d3d_S6','d3d_r2qb6','d3d_dlcfs6','d3d_cons6']])
us=d[d.owner=='chaleong'].iloc[0]
pc=['d3d_det_c_Raxis','d3d_det_c_Zaxis','d3d_det_c_kappa','d3d_det_c_tritop','d3d_det_c_tribot','d3d_det_c_vol','d3d_det_c_li','d3d_det_q95','d3d_det_betaN']
t=d.head(10)[pc].apply(pd.to_numeric).mean(); u=pd.to_numeric(us[pc]); print(pd.DataFrame({'top10':t,'us':u,'gap':t-u, 'S_gain':(t-u)*np.array([.2/7]*7+[.075,.075])}).round(5))
m=d[d.g_ratio>0].copy()
m['mtr']=(m.mast_r2psi6>0.97)&(m.mast_cons6>0.5)
m['gate_edge']=m.d3d_S6<0.87
for name,sub in [('all',m),('excl likely-MAST-trained',m[~m.mtr]),('excl MT & gate-edge',m[~m.mtr&~m.gate_edge]),('mast r2psi<0.95',m[m.mast_r2psi6<0.95]),('mast r2psi<0.97',m[m.mast_r2psi6<0.97])]:
    s=sub.sort_values('g_ratio',ascending=False).reset_index(drop=True)
    r=s.index[s.owner=='chaleong']
    print(name,len(s),'our rank',r[0]+1 if len(r) else None,'G@10',s.g_ratio.iloc[9] if len(s)>9 else None,'G@20',s.g_ratio.iloc[19] if len(s)>19 else None)
# MAST honorable-mention leaders among r2psi<0.95
c=m[m.mast_r2psi6<0.95]
for k,asc in [('mast_r2psi6',False),('mast_r2qb6',False),('mast_cons6',False),('mast_dlcfs6',True)]:
    print(k, c.sort_values(k,ascending=asc)[['owner',k]].head(4).values.tolist())
# best per-term among all
for k,asc in [('mast_r2psi6',False),('mast_r2qb6',False),('mast_cons6',False),('mast_dlcfs6',True)]:
    print('ALL',k, m.sort_values(k,ascending=asc)[['owner',k]].head(4).values.tolist())
# what-if MAST: our G with individual term moves (denominator our d3d)
base=us.mast_S6; D3=us.d3d_S6
for lbl,dS in [('R2psi->0.95',.55*(0.95-us.mast_r2psi6)),('R2psi->0.98',.55*(0.98-us.mast_r2psi6)),('qb->0.41',.15*(0.41-us.mast_r2qb6)),('qb->0.56',.15*(0.56-us.mast_r2qb6)),('cons->0.21',.2*(0.21-us.mast_cons6)),('cons->0.35',.2*(0.35-us.mast_cons6)),('dlcfs->0.18',.1*(us.mast_dlcfs6-0.18))]:
    print(lbl, round(dS,4), 'G', round((base+dS)/D3,4))
print('mastS needed top20',0.7585*D3,'top10',0.8092*D3)
# date of top entries & sign
print(m.mast_det_sign.str.strip().str[:2].value_counts())
