import pandas as pd, numpy as np, sys
D=sys.argv[1]; df=pd.read_csv(D+'/leaderboard_all.csv')
for m in ('d3d','mast'):
    for k in ('S6','r2psi6','r2qb6','dlcfs6','cons6'): df[f'{m}_{k}']=pd.to_numeric(df[f'{m}_det_{k}'],errors='coerce')
    df[f'{m}_Srec']=0.55*df[f'{m}_r2psi6'].clip(lower=0)+0.15*df[f'{m}_r2qb6'].clip(lower=0)+0.10*(1-df[f'{m}_dlcfs6'].clip(upper=1))+0.20*df[f'{m}_cons6']
    df[f'{m}_Srec4']=0.55*df[f'{m}_r2_psi'].clip(lower=0)+0.15*df[f'{m}_r2_qb'].clip(lower=0)+0.10*(1-df[f'{m}_dlcfs'].clip(upper=1))+0.20*df[f'{m}_consistency']
    print(m,'max |S6-Srec6|',(df[f'{m}_S6']-df[f'{m}_Srec']).abs().max(),' max|S4-Srec4|',(df[f'{m}_S']-df[f'{m}_Srec4']).abs().max(), 'n',df[f'{m}_S6'].notna().sum())
df['g6']=df.mast_S6/df.d3d_S6
print('g check', (df.g6.where(df.d3d_S6>=0.85,0)-df.g_ratio).abs().max())
bad=df[(df.mast_S6-df.mast_Srec).abs()>1e-4][['owner','mast_S6','mast_Srec']]; print(bad)
d=df.sort_values(['d3d_S6','d3d_r2psi6'],ascending=False).reset_index(drop=True); d['rank']=d.index+1
us=d[d.owner=='chaleong'].iloc[0]; print('our rank ch1', int(us['rank']))
for t,w,c in (('psi',.55,'d3d_r2psi6'),('qb',.15,'d3d_r2qb6'),('lcfs',-.10,'d3d_dlcfs6'),('cons',.20,'d3d_cons6')): d['dS_'+t]=w*(d[c]-us[c])
d['dS']=d.d3d_S6-us.d3d_S6
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',200)
cols=['rank','owner','d3d_S6','d3d_r2psi6','d3d_r2qb6','d3d_dlcfs6','d3d_cons6','dS','dS_psi','dS_qb','dS_lcfs','dS_cons','created']
t=d.head(30)[cols]; print(t.round(5).to_string())
t.round(5).to_csv(D+'/ch1_top30_decomp.csv',index=False)
print('mean decomposition ranks1-10 above us', d[d.d3d_S6>us.d3d_S6][['dS','dS_psi','dS_qb','dS_lcfs','dS_cons']].mean().round(5).to_dict())
# per-scalar consistency of top 10
pc=['d3d_det_c_Raxis','d3d_det_c_Zaxis','d3d_det_c_kappa','d3d_det_c_tritop','d3d_det_c_tribot','d3d_det_c_vol','d3d_det_c_li','d3d_det_q95','d3d_det_betaN']
print(d.head(25)[['rank','owner']+pc].to_string())
# thresholds
for k in (10,20): print('ch1 S at rank',k, d.d3d_S6.iloc[k-1], 'gap',d.d3d_S6.iloc[k-1]-us.d3d_S6)
# distribution
print(d.d3d_S6.describe()); print('count >=0.99', (d.d3d_S6>=0.99).sum(), '>=0.98',(d.d3d_S6>=0.98).sum(),'>=0.85',(d.d3d_S6>=0.85).sum())
# correlations among valid d3d entries with S>0.85
v=d[d.d3d_S6>=0.85][['d3d_r2psi6','d3d_r2qb6','d3d_dlcfs6','d3d_cons6']]
print('pearson n',len(v)); print(v.corr().round(3)); print(v.corr('spearman').round(3))
v=d[d.d3d_S6>=0.97][['d3d_r2psi6','d3d_r2qb6','d3d_dlcfs6','d3d_cons6']]; print('S>=.97 n',len(v)); print(v.corr('spearman').round(3))
d.to_pickle('/private/tmp/claude-502/scr/d.pkl')
