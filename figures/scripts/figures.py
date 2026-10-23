"""10-15 figures: adapt the supplied layouts; observations remain source-derived."""
from pathlib import Path
import sys,os,json,shutil,importlib.util
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];ROOT=R.parent;REV=ROOT/'provenance/2026-10-14'
spec=importlib.util.spec_from_file_location('styles',R/'code/original_style_reference.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
s.STYLE=R
plt=s.plt;Line2D=s.Line2D;Patch=s.Patch
tr=s.tr;ci=s.ci;err=s.err;tag=s.tag

def save(f,*args,**kwargs):
 for a in f.axes:
  if a.get_xscale()=='log' or a.get_yscale()=='log':a.tick_params(axis='both',labelsize=9.5)
 return s.save(f,*args,**kwargs)

POL=['frozen','frozen_trigger','current_coverage','current_residual','periodic','random']
COL=['#888888','#D28C45','#0072B2','#009E73','#9466A4','#AF725B']
MARK=['o','s','^','D','v','X']
def names():return [tr('不更新','Frozen'),tr('初始参照','Initial ref.'),tr('当前覆盖','Current cov.'),tr('当前残差','Current residual'),tr('周期','Periodic'),tr('随机','Random')]
def column(rows,height):
 f,a=plt.subplots(rows,1,figsize=(3.5,height),squeeze=False)
 f.subplots_adjust(left=.22,right=.95,bottom=.11,top=.91,hspace=.66)
 return f,a[:,0]
def width():
 ma=pd.read_csv(REV/'tables/matched.csv');f,(a,b)=column(2,4.9)
 cols=['#9299A4','#009E73'];ns=[tr('匹配 Base','Matched Base'),'Residual'];x=np.linspace(0,1,100)
 for j,arm in enumerate(['base','residual']):
  ends=[]
  for p in sorted((REV/'raw').glob('matched_*_'+arm+'.npz')):
   with np.load(p) as z:
    pr=z['prediction'];q=z['radius'];lo=np.maximum(0,np.expm1(np.log1p(pr)-q));hi=np.expm1(np.log1p(pr)+q)
    ends.append([np.mean((1+lo)/(1+pr)),np.mean((1+hi)/(1+pr))])
  ends=np.asarray(ends);assert len(ends)==10
  for low,high in ends:
   a.plot(x,np.exp(x*np.log(high)),color=cols[j],alpha=.12,lw=.5);a.plot(x,np.exp(x*np.log(low)),color=cols[j],alpha=.12,lw=.5)
  for k in [0,1]:
   m,l,h=ci(ends[:,k]);a.fill_between(x,np.exp(x*np.log(l)),np.exp(x*np.log(h)),color=cols[j],alpha=.15,lw=0)
   a.plot(x,np.exp(x*np.log(m)),color=cols[j],ls='--' if j==0 else '-',lw=1.4)
   a.plot(1,m,marker='s' if j==0 else 'o',color=cols[j],ms=4)
  g=ma[(ma.arm==arm)&(ma.factor==1)]
  a.text([.02,.29][j],[.98,.22][j],f'{ns[j]}: W={g.width.mean():.2f}\n'+tr('覆盖 ','Coverage ')+f'{100*g.coverage.mean():.2f}%',transform=a.transAxes,color=cols[j],fontsize=7,va='top')
  curve=ma[ma.arm==arm].groupby('factor').coverage.mean()
  b.plot(curve.index,curve*100,marker='s' if j==0 else 'o',ls='--' if j==0 else '-',color=cols[j],label=ns[j],ms=4)
  for fac,g in ma[ma.arm==arm].groupby('factor'):err(b,fac,g.coverage*100,fmt='none',ecolor=cols[j])
 a.set_yscale('log');a.set_xlim(-.02,1.04);a.set_ylim(.003,900);a.set_xticks([0,1],[tr('中心','Center'),tr('区间端点','Bounds')])
 a.set_ylabel(tr('平移响应 / 平移点预测','Shifted response / shifted prediction'));a.set_xlabel(tr('展开位置（几何示意）','Expansion (geometry only)'))
 tag(a,'a',tr('相近覆盖下的区间几何','Geometry at similar coverage'))
 b.set_ylim(55,96);b.axhline(90,color='#555',ls=':',lw=.7);b.set_xlabel(tr('相关性漂移因子','Correlation multiplier'));b.set_ylabel(tr('覆盖率 (%)','Coverage (%)'));b.set_xticks([1,2,4,7.39],['1','2','4','7.39']);tag(b,'b',tr('冻结区间的覆盖退化','Frozen coverage under drift'))
 f.legend(handles=[Line2D([],[],marker='s',ls='--',color=cols[0],label=ns[0]),Line2D([],[],marker='o',color=cols[1],label=ns[1])],loc='upper center',ncol=2,bbox_to_anchor=(.57,.995),fontsize=7)
 save(f,'ch4_width_fan','paper/figures/ch4_width_fan_stacked.pdf')
def scan():
 d=pd.read_csv(REV/'tables/width.csv');f,(a,b)=column(2,4.4)
 for j,ds in enumerate(['stats','imdb']):
  g=d[(d.dataset==ds)&(d.factor==4.48)]
  for ax,key in [(a,'negative_alarm_rate'),(b,'positive_alarm_rate')]:
   means=[]
   for alpha,gg in g.groupby('alpha'):
    x=100*(1-alpha);means.append((x,gg[key].mean()*100));err(ax,x,gg[key]*100,fmt=['o','s'][j],color=['#0072B2','#009E73'][j],ms=3)
   ax.plot(*zip(*sorted(means)),color=['#0072B2','#009E73'][j],label=ds,lw=1)
 for ax in [a,b]:ax.set_xticks([70,80,90,95,99]);ax.set_xlabel(tr('名义覆盖率 (%)','Nominal coverage (%)'))
 a.axhline(10,color='#888',ls=':',lw=.7);a.set_ylabel(tr('独立负窗误报 (%)','Negative-window alarms (%)'));a.legend(fontsize=7)
 b.set_ylabel(tr('4.48倍位移检出 (%)','Detection at 4.48x (%)'))
 tag(a,'a',tr('实际误报工作点','Achieved false-alarm rates'));tag(b,'b',tr('位移检出工作点','Shift-detection rates'))
 save(f,'width_scan','paper/figures/S1_width_alarm.pdf')
def feedback():
 d=pd.read_csv(REV/'tables/feedback_unified.csv');d=d[(d.pool=='pool_age.pkl')&(d.state=='age_purge50_posts')&(d.feedback_budget==60)]
 f,(a,b,c)=column(3,5.9);f.subplots_adjust(bottom=.09,top=.94,hspace=.75)
 order=['explore_00','explore_05','explore_10','explore_20','uniform'];ticks=['0%','5%','10%','20%',tr('均匀','Uniform')]
 for j,st in enumerate(order):
  g=d[d.strategy==st];err(a,j,g.feedback_coverage_bias*100,fmt='o',color='#252525',ms=4);err(b,j,g.zero_inclusive_coverage*100,fmt='D',color='#16946E',ms=4)
  vals=g.median_normalized_length.to_numpy();rng=np.random.default_rng(61014+j)
  c.scatter(vals,j+rng.uniform(-.13,.13,len(vals)),s=7,color='#159778',alpha=.32,edgecolors='none')
  c.boxplot([vals],positions=[j],orientation='horizontal',widths=.42,patch_artist=True,showfliers=False,manage_ticks=False,boxprops={'facecolor':'#ADDACB','edgecolor':'#72BFA8','linewidth':.7},medianprops={'color':'#127E64'},whiskerprops={'color':'#72BFA8'},capprops={'color':'#72BFA8'})
 for ax in [a,b]:ax.set_xticks(range(5),ticks);ax.set_xlabel(tr('均匀探索比例','Uniform exploration'))
 a.axhline(0,color='#888',ls=':',lw=.7);a.set_ylabel(tr('反馈覆盖偏差 (pp)','Feedback coverage bias (pp)'))
 b.set_ylabel(tr('覆盖率 (%)','Coverage (%)'));b.set_ylim(60,102);b.axhline(90,color='#777',ls=':',lw=.7)
 c.set_xscale('log');c.set_yticks(range(5),ticks);c.invert_yaxis();c.set_xlabel(tr('审计中位归一化长度 W','Audit median normalized length W'));c.set_ylabel(tr('均匀探索比例','Uniform exploration'))
 for ax,l,t in zip([a,b,c],'abc',[tr('反馈覆盖偏差','Feedback coverage bias'),tr('总体审计覆盖','Overall audit coverage'),tr('区间长度分布','Interval-length distribution')]):tag(ax,l,t)
 save(f,'ch4_feedback_compact','paper/figures/ch4_feedback_compact.pdf')
 f,aa=column(1,2.5);f.subplots_adjust(top=.77,bottom=.26);a=aa[0]
 for j,st in enumerate(order):err(a,j,d[d.strategy==st].zero_subgroup_coverage*100,fmt='D',color='#216AA5',ms=4)
 a.set_xticks(range(5),ticks);a.set_ylim(0,103);a.axhline(90,color='#777',ls=':',lw=.7)
 a.set_xlabel(tr('均匀探索比例','Uniform exploration'));a.set_ylabel(tr('零结果子群覆盖 (%)','Zero-result subgroup coverage (%)'))
 tag(a,'a',tr('统一评分下的子群审计','Subgroup audit under unified score'))
 save(f,'feedback_zero_subgroup','paper/figures/ch4_feedback_compact.pdf panel d')
def realdata():
 # Adapt the original boxplot source without changing its observations.
 import inspect
 src=inspect.getsource(s.realdata).replace('def realdata():','def adapted():')
 src=src.replace("f,axs=grid(height=3.0);f.subplots_adjust(wspace=.35)","f,aa=column(2,4.8);axs=np.array([aa]);f.subplots_adjust(top=.77,hspace=.72)")
 src=src.replace("ncol=3,loc='upper center'","ncol=1,loc='upper center'")
 src=src.replace("1.16","1.16")
 env=dict(s.__dict__);env.update(column=column,save=save)
 exec(src,env);env['adapted']()
def utility():
 data=ROOT/'artifacts/paper_revision_20261009_ch4_ch5/source_data';rows=[]
 for cond in ['correlation','workload']:
  d=pd.read_csv(data/('utility_'+cond+'.csv'))
  for st,model in [('before','frozen'),('after','frozen'),('after','active')]:
   r=d[(d.state==st)&(d.model==model)].iloc[0].to_dict()
   title=tr('相关','Corr.') if cond=='correlation' else tr('联合','Combined')
   state=tr('基线','Baseline') if st=='before' else (tr('漂移冻结','Frozen') if model=='frozen' else tr('修复后','Repaired'))
   rows.append(dict(label=title+' / '+state,**r))
 f,(a,b)=column(2,4.1);f.subplots_adjust(left=.36,right=.96,hspace=.66)
 for j,r in enumerate(rows):
  y=5-j;v=r['changed_plan_queries']/r['queries']*100
  a.barh(y,v,height=.45,color='#1B8A65',alpha=.7);a.text(v+.5,y,f'{int(r["changed_plan_queries"])}/{int(r["queries"])}',va='center',fontsize=7)
  m=(r['geomean_point_over_interval_runtime']-1)*100;lo=(r['ci_low']-1)*100;hi=(r['ci_high']-1)*100
  b.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt='o',color='#2569A8',capsize=2,ms=4,lw=1)
 for ax in [a,b]:ax.set_ylim(-.6,5.6);ax.set_yticks(range(6),[r['label'] for r in rows][::-1])
 a.set_xlim(0,22);a.set_xticks([0,10,20]);a.set_xlabel(tr('改变计划的比例 (%)','Queries changing plan (%)'))
 b.axvline(0,color='#8A8F98',ls=':',lw=.8);b.set_xlim(-5.2,1.2);b.set_xticks([-4,-2,0]);b.set_xlabel(tr('点 / 区间耗时比 − 1 (%)','Point / interval runtime − 1 (%)'))
 tag(a,'a',tr('改变计划的查询比例','Queries changing plan'));tag(b,'b',tr('执行耗时的配对比较','Paired execution times'))
 save(f,'ch4_plan_verified','paper/figures/ch4_plan_verified.pdf')
def continuous():
 wins=pd.read_csv(R/'tables/offline_windows.csv');runs=pd.read_csv(R/'tables/offline_runs.csv')
 dbs=['stats','imdb','ergastf1','genome']
 f,axs=plt.subplots(3,2,figsize=(7,5.8));f.subplots_adjust(left=.09,right=.98,bottom=.16,top=.88,wspace=.35,hspace=.72)
 for ax,ds,letter in zip(axs[:2].ravel(),dbs,'abcd'):
  for pol,c,mk,nm in zip(POL,COL,MARK,names()):
   g=wins[(wins.dataset==ds)&(wins.pattern=='cycles')&(wins.policy==pol)]
   v=g.pivot(index='seed',columns='window',values='coverage').to_numpy()*100
   ix=np.random.default_rng(61012).integers(0,len(v),(2000,len(v)));lo,hi=np.quantile(v[ix].mean(1),[.025,.975],axis=0)
   x=np.arange(1,81);mean=v.mean(0);ax.plot(x,mean,color=c,lw=.8,marker=mk,markevery=8,ms=2.2)
   take=np.arange(0,80,8);ax.errorbar(x[take],mean[take],yerr=np.array([mean-lo,hi-mean])[:,take],fmt='none',color=c,capsize=1,lw=.45,alpha=.75)
  for start in [16,48]:ax.axvline(start+.5,color='#999',ls=':',lw=.6);ax.axvline(start+16.5,color='#999',ls=':',lw=.6)
  ax.axhline(90,color='#777',ls=':',lw=.6);ax.set_ylim(0,102);ax.set_xticks([1,16,32,48,64,80])
  ax.set_xlabel(tr('窗口（更新下一窗生效）','Window (repair active next window)'));ax.set_ylabel(tr('审计覆盖率 (%)','Audit coverage (%)'));tag(ax,letter,ds)
 for ax,subset,letter in zip(axs[2],[dbs[:2],dbs[2:]],'ef'):
  for j,(pol,c,nm) in enumerate(zip(POL,COL,names())):
   for k,ds in enumerate(subset):
    g=runs[(runs.dataset==ds)&(runs.pattern=='cycles')&(runs.policy==pol)];m,lo,hi=ci(g.deployment_labels);x=j*3+k
    ax.bar(x,1600,color='#D5D9DF',width=.70);ax.bar(x,m-1600,bottom=1600,color=c,width=.70)
    ax.errorbar(x,m,yerr=[[m-lo],[hi-m]],fmt='none',ecolor=c,capsize=1.5,lw=.7)
   ax.text(j*3+.5,-.16,[tr('不更新','Frozen'),tr('初始','Initial'),tr('覆盖','Coverage'),tr('残差','Residual'),tr('周期','Periodic'),tr('随机','Random')][j],ha='right',va='top',rotation=40,transform=ax.get_xaxis_transform(),fontsize=6.5)
  ax.set_xticks([]);ax.set_ylim(0,10300);ax.set_yticks([0,4000,8000]);ax.set_ylabel(tr('实际在线标签','Consumed online labels'))
  tag(ax,letter,' / '.join(subset))

 f.legend(handles=[Line2D([],[],color=c,marker=m,lw=.8,label=n,ms=3) for c,m,n in zip(COL,MARK,names())],ncol=3,loc='upper center',bbox_to_anchor=(.53,.995),fontsize=7)
 save(f,'continuous_repair','paper/figures/07_temporal.pdf; actual four-database 80-window results')

def detection():
 z=np.load(REV/'detection/detection_ci.npz',allow_pickle=True);v=np.asarray(z['recs'],float).reshape(50,44,7)
 f=plt.figure(figsize=(3.5,4.6))
 gs=f.add_gridspec(2,2,left=.24,right=.95,bottom=.15,top=.78,height_ratios=[1,1.15],hspace=.75,wspace=.38)
 a=f.add_subplot(gs[0,:]);b=f.add_subplot(gs[1,0]);c=f.add_subplot(gs[1,1])
 factors=[1.,1.35,2.,2.72,4.48,7.39]
 for det,col,mk,ls,name in [(0,'#2166AC','o','-',tr('覆盖（有标签）','Coverage (labeled)')),(1,'#A65E83','^','--',tr('残差（有标签）','Residual (labeled)')),(3,'#7283AD','s',':',tr('特征均值（无标签）','Feature mean (unlabeled)'))]:
  means=[]
  for fac in factors:
   ix=np.flatnonzero((v[0,:,0]==0)&np.isclose(v[0,:,1],fac)&(v[0,:,2]==det))[0];arr=v[:,ix,3]*100
   err(a,fac,arr,fmt=mk,color=col,ms=3);means.append(arr.mean())
  a.plot(factors,means,color=col,ls=ls,lw=1,label=name)
 a.set_ylim(0,107);a.set_xscale('log');a.minorticks_off();a.set_xticks([1,2,4,7.39],['1','2','4','7.39'])
 a.set_xlabel(tr('相关性漂移因子','Correlation multiplier'));a.set_ylabel(tr('报警率 (%)','Alarm rate (%)'))
 tag(a,'a',tr('监控报警率','Monitoring alarms'))
 labels=[]
 for fam,faclist in [(0,[1.35,2,2.72,4.48,7.39]),(1,[2,5,10,50])]:
  bi=np.flatnonzero((v[0,:,0]==fam)&(v[0,:,1]==1)&(v[0,:,2]==3))[0]
  for fac in faclist:
   ix=np.flatnonzero((v[0,:,0]==fam)&np.isclose(v[0,:,1],fac)&(v[0,:,2]==3))[0];j=len(labels)
   labels.append((tr('相关×','Corr×') if fam==0 else tr('尺度×','Scale×'))+f'{fac:g}')
   loss=(v[:,bi,5]-v[:,ix,5])*100;inc=(v[:,ix,3]-v[:,bi,3])*100
   for ax,arr,col in [(b,loss,'#D8A477'),(c,inc,'#91A0C4')]:
    m,l,h=ci(arr);ax.barh(j,m,color=col,height=.57);ax.errorbar(m,j,xerr=[[m-l],[h-m]],fmt='none',ecolor=col,capsize=1.4,lw=.8)
 for ax in [b,c]:ax.set_yticks(range(len(labels)),labels if ax==b else []);ax.invert_yaxis();ax.tick_params(axis='y',labelsize=6.8)
 b.set_xlabel(tr('损失 (pp)','Loss (pp)'));c.set_xlabel(tr('增量 (pp)','Increase (pp)'))
 tag(b,'b',tr('审计覆盖','Audit coverage'));tag(c,'c',tr('特征报警','Feature alarms'))
 h,l=a.get_legend_handles_labels();f.legend(h,l,loc='upper center',ncol=1,bbox_to_anchor=(.56,.995),fontsize=7)
 save(f,'ch4_detection_compact','paper/figures/ch4_detection_compact.pdf')

def main():
 for lang in ['cn','en']:
  s.LANG=lang;s.OUT=R/('figures_'+lang)
  for fn in [width,scan,feedback,realdata,detection,utility,continuous]:
   print(lang,fn.__name__,flush=True);fn()
  for p in s.OUT.glob('*.pdf'):shutil.copy2(p,R/('manuscript_'+lang)/'figures'/p.name)
 (R/'qa/figure_manifest.json').write_text(json.dumps(s.MANIFEST,indent=2,ensure_ascii=False),encoding='utf-8')
 # The latest user preference requires the actual original Figure 4 renderer.
 import subprocess,sys
 subprocess.run([sys.executable,str(R/'code/figure4_original.py')],check=True)
 subprocess.run([sys.executable,str(R/'code/annotation_figures.py')],check=True)

if __name__=='__main__':main()

