import pandas as pd, numpy as np, sys
D=sys.argv[1]
d=pd.read_pickle('/private/tmp/claude-502/scr/d.pkl')
pd.set_option('display.width',260); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',200)
g=d[d.d3d_S6>0].groupby(['d3d_r2psi6','d3d_r2qb6','d3d_dlcfs6','d3d_cons6'])['owner'].apply(list); print('D3D-identical groups'); print(g[g.apply(len)>1].to_string())
m=d[d.mast_S6.notna()].copy()
m['msign']=m.mast_det_sign.str.strip().str[:2]
m['flag']=np.select([(m.mast_r2psi6>0.97)&(m.mast_cons6>0.5),(m.mast_r2psi6>0.97)|(m.mast_cons6>0.5),m.mast_r2psi6<0.95],['LIKELY-MAST-TRAINED','suspect','compliant-range'],'grey(0.95-0.97)')
m=m.sort_values('g_ratio',ascending=False).reset_index(drop=True); m['g_rank']=m.index+1
cols=['g_rank','owner','g_ratio','d3d_S6','mast_S6','mast_r2psi6','mast_r2qb6','mast_det_q95','mast_det_betaN','mast_dlcfs6','mast_cons6','msign','flag','method','created']
print(m[cols].head(60).to_string(max_colwidth=40))
m[cols+['mast_det_c_Raxis','mast_det_c_Zaxis','mast_det_c_kappa','mast_det_c_tritop','mast_det_c_tribot','mast_det_c_vol','mast_det_c_li','code_url']].to_csv(D+'/ch2_by_G.csv',index=False)
us=m[m.owner=='chaleong'].iloc[0]; print('our G rank',us.g_rank,'of',(m.g_ratio>0).sum())
print(m.flag.value_counts())
c=m[(m.mast_r2psi6<0.95)&(m.g_ratio>0)]; print('compliant-range top'); print(c[cols].head(15).to_string(max_colwidth=40))
for k in (10,20): print('G at rank',k,m.g_ratio.iloc[k-1],'mastS', m.mast_S6.iloc[k-1])
pc=['mast_det_c_Raxis','mast_det_c_Zaxis','mast_det_c_kappa','mast_det_c_tritop','mast_det_c_tribot','mast_det_c_vol','mast_det_c_li']
print(m[['g_rank','owner']+pc].head(30).to_string())
v=m[m.g_ratio>0][['mast_r2psi6','mast_r2qb6','mast_dlcfs6','mast_cons6','d3d_S6']]; print(v.corr('spearman').round(2))
