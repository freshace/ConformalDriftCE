"""Restore supplied visual designs while retaining the verified unified-score evidence."""
from pathlib import Path
import sys,os,json,shutil,subprocess,hashlib,importlib.util
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];STYLE=Path(__file__).resolve().parents[1];REV=ROOT/'provenance/2026-10-14'
os.environ['MPLCONFIGDIR']=str(REV/'tmp/mpl');os.environ['MPLBACKEND']='Agg'
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0,str(REV/'code'))
import unified_core as uc
sys.path.insert(0,'E:/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
mpl.rcParams.update({'font.family':['Arial','Microsoft YaHei','DejaVu Sans'],'font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7,'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'dejavusans'})
ARMS=['frozen','recal_only','repair','garbage'];COLORS=['#8A8A8A','#0072B2','#009E73','#E57827'];MARK=['o','s','^','D']
LANG='cn'
OUT=None
MANIFEST=[]
def tr(cn,en):return cn if LANG=='cn' else en
def ci(v):
 a=np.asarray(v,float);a=a[np.isfinite(a)]
 boot=np.random.default_rng(61014).choice(a,(10000,len(a)),replace=True).mean(1)
 return np.array([a.mean(),*np.quantile(boot,[.025,.975])])
def err(ax,x,v,axis='y',**kw):
 m,lo,hi=ci(v);delta=np.array([[m-lo],[hi-m]])
 ax.errorbar(x,m,yerr=delta,capsize=2,lw=.8,**kw) if axis=='y' else ax.errorbar(m,x,xerr=delta,capsize=2,lw=.8,**kw)
 return m
def tag(ax,letter,title):
 ax.text(0,1.08,letter,transform=ax.transAxes,fontweight='bold',fontsize=9,va='bottom')
 ax.text(.065,1.08,title,transform=ax.transAxes,fontsize=8,va='bottom')
def grid(rows=1,cols=2,height=2.75):
 f,a=plt.subplots(rows,cols,figsize=(7.0,height),squeeze=False)
 f.subplots_adjust(left=.105,right=.97,bottom=.20 if rows==1 else .13,top=.82,wspace=.43,hspace=.70)
 return f,a
def save(f,name,reference,groups=None,exemptions=None):
 for ax in f.axes:
  if ax.get_xscale()=='log' or ax.get_yscale()=='log':ax.minorticks_off()
 f.canvas.draw()
 kw={}
 if groups is not None:kw['comparable_groups']=groups
 if exemptions is not None:kw['exemptions']=exemptions
 require_matplotlib_panel_alignment(f,json_out=str(STYLE/'qa'/f'{LANG}_{name}.alignment.json'),overlay_svg=str(STYLE/'qa'/f'{LANG}_{name}.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True,**kw)
 for ext in ['pdf','svg','png']:f.savefig(OUT/f'{name}.{ext}',dpi=300,facecolor='white')
 plt.close(f)
 MANIFEST.append(dict(language=LANG,figure=name,action='original-style adaptation',reference=reference,data='verified unified-score results',sha256=hashlib.sha256((OUT/f'{name}.pdf').read_bytes()).hexdigest()))
def mechanism():
 rows=[]
 for arm in ARMS:
  for p in sorted((REV/'raw').glob('audit_*_'+arm+'.npz')):
   with np.load(p) as z:
    m=uc.metrics(z['truth'],z['prediction'],z['radius']);m.update(arm=arm,self_coverage=uc.coverage(z['calibration_scores'],z['radius']));rows.append(m)
 d=pd.DataFrame(rows)
 f,aa=plt.subplots(2,1,figsize=(3.65,6.0));f.subplots_adjust(left=.19,right=.94,bottom=.09,top=.94,hspace=.59)
 a,b=aa
 names=[tr('冻结','Frozen'),tr('仅重校准','Recalibrated'),tr('修复','Repaired'),tr('放大预测','Inflated')]
 for j,arm in enumerate(ARMS):
  g=d[d.arm==arm];m=[g.self_coverage.mean()*100,g.coverage.mean()*100]
  a.plot([0,1],m,marker=MARK[j],ms=4,color=COLORS[j],ls='-' if j==0 else '--',lw=.9,mfc='white')
  for x,key in enumerate(['self_coverage','coverage']):err(a,x,g[key]*100,fmt='none',ecolor=COLORS[j])
  a.text(1.14,[65,97,80,87][j],names[j]+'\n'+f'{m[1]:.1f}%',color=COLORS[j],fontsize=7,va='center')
  b.scatter(g.width,g.q50,color=COLORS[j],marker=MARK[j],s=9,alpha=.28,edgecolors='none')
  x,y=g.width.mean(),g.q50.mean();xc=ci(g.width);yc=ci(g.q50)
  b.errorbar(x,y,xerr=[[x-xc[1]],[xc[2]-x]],yerr=[[y-yc[1]],[yc[2]-y]],fmt=MARK[j],ms=5,color=COLORS[j],ecolor=COLORS[j],capsize=2,mec='white',mew=.6)
  off=[(5,15),(-4,17),(12,-3),(-36,-27)][j]
  b.annotate(names[j]+'\n'+f'{x:.1f} / {y:.2f}',(x,y),xytext=off,textcoords='offset points',fontsize=6.8,color=COLORS[j])
 a.axhline(90,color='#888',ls=':',lw=.7);a.set_xlim(-.15,1.75);a.set_ylim(59,102);a.set_xticks([0,1],[tr('同批自测','Self-evaluation'),tr('独立审计','Independent audit')]);a.set_ylabel(tr('覆盖率 (%)','Coverage (%)'));tag(a,'a',tr('自测接近目标，审计结果分化','Self-coverage matches target; audit differs'))
 b.set_xscale('log');b.set_yscale('log');b.set_xlim(5,3000);b.set_ylim(1,180);b.set_xlabel(tr('平均归一化长度 W（越小越紧）','Mean normalized length W'));b.set_ylabel(tr('中位 q-error（越低越好）','Median q-error'));tag(b,'b',tr('区间代价与点误差','Interval cost and point error'))
 save(f,'03_failure_modes_stacked','paper/figures/03_failure_modes_stacked.pdf')
def motivation():
 d=pd.read_csv(REV/'tables/motivation.csv');f=plt.figure(figsize=(3.65,2.95));a=f.add_axes([.15,.23,.66,.64])
 # Single waterfall with the original right-hand GMQ axis.
 b=a.twinx()
 fac=sorted(d.factor.unique());cov=np.array([d[d.factor==x].coverage.mean()*100 for x in fac]);gm=np.array([d[d.factor==x].gmq.mean() for x in fac])
 a.bar(0,cov[0],color='#AAAAAA',width=.56)
 for i in range(1,len(fac)):
  a.bar(i,cov[i-1]-cov[i],bottom=cov[i],color='#E78132',width=.62)
  a.plot([i-.65,i+.12],[cov[i-1]]*2,color='#888',ls='--',lw=.75)
  a.text(i,101,f'{cov[i]-cov[i-1]:.1f}',ha='center',fontsize=8,color='#D55E00')
 a.bar(6,cov[-1],color='#E78132',width=.62)
 a.axhline(cov[0],color='#777',ls=':',lw=.8);a.text(-.3,cov[0]+4,f'{cov[0]:.1f}%',color='#777',fontweight='bold')
 a.text(6,cov[-1]+3,f'{cov[-1]:.1f}%',ha='center',color='#D55E00',fontweight='bold')
 b.plot(range(6),gm,'o-',color='#20282D',ms=4,lw=1.5)
 for i,x in enumerate(fac):
  err(a,i,d[d.factor==x].coverage*100,fmt='none',ecolor='#B25F25')
  err(b,i,d[d.factor==x].gmq,fmt='none',ecolor='#20282D')
 for i in [0,5]:b.annotate(f'{gm[i]:.2f}',(i,gm[i]),xytext=(8,-11) if i==0 else (-5,-28),textcoords='offset points',fontsize=8)
 a.set_ylim(0,107);a.set_yticks([0,30,60,90]);a.set_xticks(range(7),[f'{v:g}' if v in [1,2] else f'{v:.2f}' for v in fac]+[tr('最终','Final')]);a.set_ylabel(tr('覆盖率 (%)','Coverage (%)'),color='#D55E00');a.set_xlabel(tr('相关性漂移档位\n条上数字 = 覆盖损失 (pp)','Correlation multiplier\nlabels = coverage loss (pp)'))
 b.set_ylim(1,6.8);b.set_ylabel(tr('GMQ（越低越好）','GMQ (lower is better)'));b.spines['right'].set_visible(True)
 # A twin axis is one plot area, with no comparable panel alignment claim.
 f.canvas.draw()
 for ext in ['pdf','svg','png']:f.savefig(OUT/f'02_necessity.{ext}',dpi=300,facecolor='white')
 plt.close(f);MANIFEST.append(dict(language=LANG,figure='02_necessity',action='original-style adaptation',reference='paper/figures/02_necessity.pdf',data='tables/motivation.csv'))
def width():
 ma=pd.read_csv(REV/'tables/matched.csv');d=pd.read_csv(REV/'tables/width.csv');f,axs=grid(2,2,4.95);a,b,c,e=axs.ravel()
 cols=['#9299A4','#009E73'];names=[tr('匹配 Base','Matched Base'),'Residual'];x=np.linspace(0,1,100)
 for j,arm in enumerate(['base','residual']):
  endpoints=[]
  for p in sorted((REV/'raw').glob('matched_*_'+arm+'.npz')):
   with np.load(p) as z:
    pr=z['prediction'];q=z['radius'];lo=np.maximum(0,np.expm1(np.log1p(pr)-q));hi=np.expm1(np.log1p(pr)+q)
    endpoints.append([np.mean((1+lo)/(1+pr)),np.mean((1+hi)/(1+pr))])
  ends=np.array(endpoints);assert len(ends)==10
  for low,high in ends:
   a.plot(x,np.exp(x*np.log(high)),color=cols[j],alpha=.12,lw=.5);a.plot(x,np.exp(x*np.log(low)),color=cols[j],alpha=.12,lw=.5)
  for k in [0,1]:
   m,l,h=ci(ends[:,k]);a.fill_between(x,np.exp(x*np.log(l)),np.exp(x*np.log(h)),color=cols[j],alpha=.15,lw=0);a.plot(x,np.exp(x*np.log(m)),color=cols[j],ls='--' if j==0 else '-',lw=1.4)
  for k in [0,1]:a.plot(1,ends[:,k].mean(),marker='s' if j==0 else 'o',color=cols[j],ms=4)
  g=ma[(ma.arm==arm)&(ma.factor==1)]
  a.text(.45,[.96,.21][j],f'{names[j]}: W={g.width.mean():.2f}\n'+tr('覆盖 ','Coverage ')+f'{100*g.coverage.mean():.2f}%',transform=a.transAxes,color=cols[j],fontsize=7,va='top')
  curve=ma[ma.arm==arm].groupby('factor').coverage.mean()
  b.plot(curve.index,curve*100,marker='s' if j==0 else 'o',ls='--' if j==0 else '-',color=cols[j],label=names[j],ms=4)
  for fac,g in ma[ma.arm==arm].groupby('factor'):err(b,fac,g.coverage*100,fmt='none',ecolor=cols[j])
 a.set_yscale('log');a.set_xlim(-.02,1.04);a.set_ylim(.003,900);a.set_xticks([0,1],[tr('中心','Center'),tr('区间端点','Endpoints')]);a.set_ylabel(tr('平移响应 / 平移点预测','Shifted response / shifted prediction'));a.set_xlabel(tr('展开位置（中间仅为几何示意）','Expansion (geometry only)'));tag(a,'a',tr('相近覆盖下的区间几何','Interval geometry at similar coverage'))
 b.set_ylim(55,96);b.axhline(90,color='#555',ls=':',lw=.7);b.set_xlabel(tr('相关性漂移因子','Correlation multiplier'));b.set_ylabel(tr('覆盖率 (%)','Coverage (%)'));tag(b,'b',tr('冻结区间的覆盖退化','Frozen interval coverage under drift'))
 for j,ds in enumerate(['stats','imdb']):
  g=d[(d.dataset==ds)&(d.factor==4.48)]
  for ax,key in [(c,'negative_alarm_rate'),(e,'positive_alarm_rate')]:
   means=[]
   for alpha,gg in g.groupby('alpha'):
    xx=100*(1-alpha);means.append((xx,gg[key].mean()*100));err(ax,xx,gg[key]*100,fmt='o' if j==0 else 's',color=['#0072B2','#009E73'][j],ms=3)
   means.sort();ax.plot(*zip(*means),color=['#0072B2','#009E73'][j],lw=1,label=ds)
 for ax in [c,e]:ax.set_xlabel(tr('名义覆盖率 (%)','Nominal coverage (%)'));ax.set_xticks([70,80,90,95,99])
 c.set_ylabel(tr('独立负窗误报 (%)','Independent negative alarms (%)'));c.axhline(10,color='#555',ls=':',lw=.7);c.legend(fontsize=7);e.set_ylabel(tr('4.48倍位移检出率 (%)','Detection at 4.48x (%)'));tag(c,'c',tr('实际误报工作点','Actual false-alarm operating points'));tag(e,'d',tr('位移检出工作点','Shift-detection operating points'))
 f.legend(handles=[Line2D([],[],marker='s',ls='--',color=cols[0],label=names[0]),Line2D([],[],marker='o',color=cols[1],label=names[1]),Line2D([],[],ls=':',color='#555',label=tr('90%目标','90% target'))],loc='upper center',ncol=3,bbox_to_anchor=(.53,.995))
 save(f,'ch4_width_fan','paper/figures/ch4_width_fan.pdf + S1_width_alarm.pdf')
def feedback():
 d=pd.read_csv(REV/'tables/feedback_unified.csv');d=d[(d.pool=='pool_age.pkl')&(d.state=='age_purge50_posts')&(d.feedback_budget==60)]
 f,axs=grid(2,2,4.8);a,b,c,e=axs.ravel();order=['explore_00','explore_05','explore_10','explore_20','uniform'];ticks=['0%','5%','10%','20%',tr('均匀','Uniform')]
 for j,st in enumerate(order):
  g=d[d.strategy==st];err(a,j,g.feedback_coverage_bias*100,fmt='o',color='#252525',ms=4)
  err(b,j,g.zero_inclusive_coverage*100,fmt='D',color='#16946E',ms=4)
  err(e,j,g.zero_subgroup_coverage*100,fmt='D',color='#216AA5',ms=4)
  vals=g.median_normalized_length.to_numpy();rng=np.random.default_rng(61014+j)
  c.scatter(vals,j+rng.uniform(-.13,.13,len(vals)),s=7,color='#159778',alpha=.32,edgecolors='none')
  box=c.boxplot([vals],positions=[j],orientation='horizontal',widths=.42,patch_artist=True,showfliers=False,manage_ticks=False,boxprops={'facecolor':'#ADDACB','edgecolor':'#72BFA8','linewidth':.7},medianprops={'color':'#127E64'},whiskerprops={'color':'#72BFA8'},capprops={'color':'#72BFA8'})
 for ax in [a,b,e]:ax.set_xticks(range(5),ticks)
 a.axhline(0,color='#888',ls=':',lw=.7);a.set_ylabel(tr('反馈减总体覆盖 (pp)','Feedback minus overall coverage (pp)'));b.set_ylabel(tr('覆盖率 (%)','Coverage (%)'));b.set_ylim(60,102);b.axhline(90,color='#777',ls=':',lw=.7)
 c.set_xscale('log');c.set_yticks(range(5),ticks);c.invert_yaxis();c.set_xlabel(tr('审计中位归一化长度 W','Audit median normalized length W'));c.set_ylabel(tr('均匀探索比例','Uniform exploration'))
 e.set_ylabel(tr('零结果覆盖率 (%)','Zero-result coverage (%)'));e.set_ylim(0,103);e.set_xlabel(tr('均匀探索比例','Uniform exploration'));e.axhline(90,color='#777',ls=':',lw=.7)
 for ax,l,title in zip(axs.ravel(),'abcd',[tr('反馈覆盖偏差','Feedback coverage bias'),tr('总体审计覆盖','Overall audit coverage'),tr('区间长度：越小越紧','Interval length: smaller is tighter'),tr('零结果子群覆盖','Zero-result subgroup coverage')]):tag(ax,l,title)
 f.legend(handles=[Line2D([],[],marker='D',ls='',color='#16946E',label=tr('统一评分：总体覆盖','Unified score: overall coverage')),Line2D([],[],marker='D',ls='',color='#216AA5',label=tr('统一评分：零结果子群','Unified score: zero-result subgroup'))],loc='upper center',ncol=2,bbox_to_anchor=(.53,.995))
 save(f,'ch4_feedback_compact','paper/figures/ch4_feedback_compact.pdf')
def cross():
 d=pd.read_csv(REV/'tables/cross_all.csv');val=np.zeros((4,6));texts={}
 for i,ds in enumerate(['stats','imdb','ergastf1','genome']):
  for j,cond in enumerate(['corr','work','schema']):
   for k,st in enumerate(['frozen','repair']):
    x=ci(d[(d.dataset==ds)&(d.condition==cond)&(d.stage==st)].coverage*100);val[i,2*j+k]=x[0];texts[i,2*j+k]=x
 f,ax=plt.subplots(figsize=(7,2.9));f.subplots_adjust(left=.10,right=.91,bottom=.15,top=.83)
 im=ax.imshow(val,cmap='Blues',vmin=0,vmax=100,aspect='auto')
 for (i,j),v in texts.items():
  ax.text(j,i-.08,f'{v[0]:.1f}',ha='center',va='center',fontweight='bold',fontsize=9,color='white' if v[0]>75 else '#24333D')
  ax.text(j,i+.22,f'{v[1]:.1f}–{v[2]:.1f}',ha='center',va='center',fontsize=7,color='white' if v[0]>75 else '#24333D')
 ax.set_yticks(range(4),['stats','imdb','ergastf1','genome']);ax.set_xticks(range(6),[tr('冻结','Frozen'),tr('适应后','Adapted')]*3)
 ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True);ax.grid(which='minor',color='white',lw=1);ax.tick_params(which='minor',bottom=False,left=False)
 for j,title in enumerate([tr('相关性漂移','Correlation drift'),tr('工作负载漂移','Workload drift'),tr('模式漂移','Schema drift')]):ax.text((2*j+1)/6,1.16,title,ha='center',transform=ax.transAxes,fontweight='bold',fontsize=9)
 f.colorbar(im,ax=ax,fraction=.025,pad=.045,label=tr('覆盖率 (%)','Coverage (%)'))
 save(f,'06_cross_dataset','paper/figures/06_cross_dataset.pdf')
def realdata():
 data=json.loads((REV/'tables/realdata.json').read_text(encoding='utf-8'));f,axs=grid(height=3.0);f.subplots_adjust(wspace=.35)
 for ax,kind,base,l,title in zip(axs[0],['','_sens'],['base_cov','base_cov_sens'],'ab',[tr('全部查询','All queries'),tr('相关性敏感子集','Sensitive subgroup')]):
  baseline=np.mean([r['results'][base] for r in data])*100;ax.axhline(baseline,color='#353A40',ls='--',lw=.8);ax.axhline(90,color='#888',ls=':',lw=.5)
  for j,(st,color) in enumerate([('f','#AAAAAA'),('r','#159C80')]):
   vals=[np.array([r['results'][str(rho)]['cov_'+st+kind] for r in data])*100 for rho in [.3,.45,.6,.75,.9]]
   pos=np.array([.3,.45,.6,.75,.9])+(j-.5)*.023
   ax.boxplot(vals,positions=pos,widths=.018,patch_artist=True,showfliers=True,flierprops={'marker':'.','markersize':1.6,'markeredgecolor':color,'markerfacecolor':color,'alpha':.5},manage_ticks=False,boxprops={'facecolor':'white','edgecolor':color,'linewidth':.7},medianprops={'color':color,'linewidth':.7},whiskerprops={'color':color,'linewidth':.7},capprops={'color':color,'linewidth':.7})
   for xx,vs in zip(pos,vals):ax.plot(xx,vs.mean(),'o',mfc='white',mec=color,ms=3)
   ax.text(.94,vals[-1].mean()+((11 if vals[-1].mean()<15 else -11) if st=='f' else 6),f'{vals[-1].mean():.1f}%',color='#777777' if st=='f' else '#009E73',fontsize=8,ha='left',va='center')
  ax.set_ylim(0,105);ax.set_xlim(.26,.99);ax.set_xticks([.3,.45,.6,.75,.9],['0.30','0.45','0.60','0.75','0.90']);ax.set_xlabel(tr('组内相关系数 ρ','Within-group correlation ρ'));ax.set_ylabel(tr('覆盖率 (%)','Coverage (%)'));tag(ax,l,title)
 f.legend(handles=[Patch(fc='#BBBBBB',label=tr('冻结','Frozen')),Patch(fc='#A3DACA',label=tr('适应后','Adapted')),Line2D([],[],ls='--',color='#353A40',label=tr('该组无漂移基线','Group no-drift baseline'))],ncol=3,loc='upper center',bbox_to_anchor=(.52,.995))
 save(f,'09_data_level','paper/figures/09_data_level.pdf')
def legacy():
 wins=pd.read_csv(REV/'legacy/monitoring_windows.csv');runs=pd.read_csv(REV/'legacy/monitoring_policies.csv')
 f=plt.figure(figsize=(7,4.65));gs=f.add_gridspec(2,2,height_ratios=[1,1],left=.10,right=.97,bottom=.20,top=.83,wspace=.35,hspace=.70);aa=f.add_subplot(gs[0,0]);bb=f.add_subplot(gs[0,1]);cc=f.add_subplot(gs[1,:])
 pol=['frozen','trigger_recalibrate','trigger_retrain','every_window_recalibrate'];cols=['#888888','#0072B2','#009E73','#9466A4'];marks=['o','s','^','D'];names=[tr('冻结','Frozen'),tr('触发重校准','Triggered recal.'),tr('触发重训','Triggered retrain'),tr('每窗重校准','Every-window recal.')]
 for ax,pat,l,title in [(aa,'abrupt','a',tr('突变相关性回放','Abrupt correlation replay')),(bb,'gradual','b',tr('渐变相关性回放','Gradual correlation replay'))]:
  for policy,col,mk,name in zip(pol,cols,marks,names):
   g=wins[(wins.pattern==pat)&(wins.policy==policy)];v=g.pivot(index='seed',columns='window',values='coverage').to_numpy()*100;x=np.arange(1,13)
   m=v.mean(0);ix=np.random.default_rng(61014).integers(0,len(v),(10000,len(v)));lo,hi=np.quantile(v[ix].mean(1),[.025,.975],axis=0)
   ax.errorbar(x,m,yerr=np.vstack([m-lo,hi-m]),fmt=mk+'-',ms=2.5,color=col,capsize=1.5,lw=.8,label=name)
  ax.set_ylim(60,102);ax.set_xticks([1,4,5,8,12]);ax.axvline(4.5,color='#777',ls=':',lw=.7);ax.axhline(90,color='#888',ls=':',lw=.6);ax.set_ylabel(tr('覆盖率 (%)','Coverage (%)'));ax.set_xlabel(tr('窗口（更新下一窗生效）','Window (updates active next window)'));tag(ax,l,title)
 xs=[];ticks=[]
 for j,(policy,col,name) in enumerate(zip(pol,cols,names)):
  for k,pat in enumerate(['abrupt','gradual']):
   g=runs[(runs.pattern==pat)&(runs.policy==policy)];x=j*3+k;v=g.deployment_feedback_labels.to_numpy();m,l,h=ci(v);cc.bar(x,1200,color='#D5D9DF',width=.7);cc.bar(x,m-1200,bottom=1200,color=col,width=.7);cc.errorbar(x,m,yerr=[[m-l],[h-m]],fmt='none',capsize=2,ecolor=col,lw=.8);cc.text(x,h+650,f'{m:.0f}',ha='center',fontsize=7);xs.append(x);ticks.append(tr('突变','Abrupt') if k==0 else tr('渐变','Gradual'))
  cc.text(j*3+.5,-.37,name,ha='center',va='top',transform=cc.get_xaxis_transform(),fontsize=7)
 cc.set_xticks(xs,ticks);cc.set_ylim(0,7200);cc.set_ylabel(tr('在线标签数量','Online labels'));tag(cc,'c',tr('监控与更新的实际消耗','Consumed monitoring and update labels'))
 f.legend(handles=[Line2D([],[],color=c,marker=m,lw=.8,label=n) for c,m,n in zip(cols,marks,names)],ncol=4,loc='upper center',bbox_to_anchor=(.52,.995),fontsize=7)
 save(f,'07_temporal_compact','paper/figures/07_temporal_compact.pdf')
def detection():
 z=np.load(REV/'detection/detection_ci.npz',allow_pickle=True);v=np.asarray(z['recs'],float).reshape(50,44,7);f,axs=grid(1,3,2.9);f.subplots_adjust(left=.08,right=.98,wspace=.50,top=.73,bottom=.25);a,b,c=axs[0]
 factors=[1.,1.35,2.,2.72,4.48,7.39]
 for det,col,mk,ls,name in [(0,'#2166AC','o','-',tr('覆盖（有标签）','Coverage (labeled)')),(1,'#A65E83','^','--',tr('残差（有标签）','Residual (labeled)')),(3,'#7283AD','s',':',tr('特征均值（无标签）','Feature mean (unlabeled)'))]:
  means=[]
  for fac in factors:
   ix=np.flatnonzero((v[0,:,0]==0)&np.isclose(v[0,:,1],fac)&(v[0,:,2]==det))[0];arr=v[:,ix,3]*100;err(a,fac,arr,fmt=mk,color=col,ms=3);means.append(arr.mean())
  a.plot(factors,means,color=col,ls=ls,lw=1,label=name)
 a.set_ylim(0,107);a.set_xscale('log');a.minorticks_off();a.set_xticks([1,2,4,7.39],['1','2','4','7.39']);a.set_xlabel(tr('相关性漂移因子','Correlation multiplier'));a.set_ylabel(tr('报警率 (%)','Alarm rate (%)'));tag(a,'a',tr('监控报警率','Monitoring alarms'))
 labels=[]
 for fam,faclist in [(0,[1.35,2,2.72,4.48,7.39]),(1,[2,5,10,50])]:
  bi=np.flatnonzero((v[0,:,0]==fam)&(v[0,:,1]==1)&(v[0,:,2]==3))[0]
  for fac in faclist:
   ix=np.flatnonzero((v[0,:,0]==fam)&np.isclose(v[0,:,1],fac)&(v[0,:,2]==3))[0];j=len(labels);labels.append((tr('相关 ×','Corr ×') if fam==0 else tr('尺度 ×','Scale ×'))+f'{fac:g}')
   loss=(v[:,bi,5]-v[:,ix,5])*100;inc=(v[:,ix,3]-v[:,bi,3])*100
   for ax,arr,col in [(b,loss,'#D8A477'),(c,inc,'#91A0C4')]:
    m,l,h=ci(arr);ax.barh(j,m,color=col,height=.57);ax.errorbar(m,j,xerr=[[m-l],[h-m]],fmt='none',ecolor=col,capsize=1.4,lw=.8)
 for ax in [b,c]:ax.set_yticks(range(len(labels)),labels if ax==b else []);ax.invert_yaxis()
 b.set_xlabel(tr('覆盖损失 (pp)','Coverage loss (pp)'));c.set_xlabel(tr('报警率增量 (pp)','Alarm increase (pp)'));tag(b,'b',tr('审计覆盖损失','Audit coverage loss'));tag(c,'c',tr('特征报警增量','Feature-alarm increase'))
 h,l=a.get_legend_handles_labels();f.legend(h,l,loc='upper center',ncol=3,bbox_to_anchor=(.53,.995),fontsize=7)
 save(f,'ch4_detection_compact','paper/figures/ch4_detection_compact.pdf')
def architecture():
 original=ROOT/('paper/revision_20261013_focus/authoring/01_architecture_focus.tex' if LANG=='cn' else 'paper/english_restructured_20261014/authoring/01_architecture_focus.tex')
 text=original.read_text(encoding='utf-8')
 text=text.replace('正数 / 含零评分',r'统一 $\log(1+y)$ 评分').replace('Positive / zero-inclusive',r'Unified $\log(1+y)$ score')
 text=text.replace(r'\begin{document}',r'\DeclareMathSizes{8}{8}{6}{5.2}'+'\n'+r'\DeclareMathSizes{8.5}{8.5}{6}{5.2}'+'\n'+r'\DeclareMathSizes{9}{9}{6}{5.2}'+'\n'+r'\DeclareMathSizes{10}{10}{7}{5.2}'+'\n'+r'\begin{document}')
 text=text.replace('(4.54,1.80) rectangle','(4.26,1.80) rectangle')
 text=text.replace(r'{报警、预算\\与冷却条件}',r'{报警\\预算\\冷却}').replace(r'{Alarm, budget\\and cooldown}',r'{Alarm\\budget\\cooldown}')
 text=text.replace(r'^{(v_t)}',r'^{(v)}').replace(r'本窗版本 $v_t$',r'本窗版本 $v$').replace(r'Window version $v_t$',r'Window version $v$')
 text=text.replace(r'I_{z_t^{\rm dep}}(x)',r'I_t(x)')


 d=STYLE/'code'/('architecture_'+LANG);d.mkdir(exist_ok=True);(d/'architecture.tex').write_text(text,encoding='utf-8')
 cmd=['D:/TeXlive/texlive/2026/bin/windows/xelatex.exe','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','architecture.tex']
 t=subprocess.run(cmd,cwd=d,stdout=subprocess.PIPE,stderr=subprocess.PIPE);(STYLE/'qa'/('architecture_'+LANG+'.log')).write_bytes(t.stdout+t.stderr);assert t.returncode==0
 shutil.copy2(d/'architecture.pdf',OUT/'01_architecture_focus.pdf')
 MANIFEST.append(dict(language=LANG,figure='01_architecture_focus',action='original TikZ layout with unified-score label',reference=str(original),data='no numerical observations'))
def reuse():
 pairs=[('continuous_repair',ROOT/'provenance/2026-10-12/figures/continuous_repair.pdf'),('real_cost',ROOT/'provenance/2026-10-12/figures/real_cost.pdf'),('S5_real_online_1000',REV/'backup/source_cn/figures/S5_real_online_1000.pdf'),('ch4_plan_verified',REV/('backup/source_cn/figures/ch4_plan_verified.pdf' if LANG=='cn' else 'backup/source_en/figures/ch4_plan_verified.pdf'))]
 for name,src in pairs:
  shutil.copy2(src,OUT/(name+'.pdf'));MANIFEST.append(dict(language=LANG,figure=name,action='exact original PDF reuse',reference=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),data='unchanged verified log1p evidence'))
def main():
 global LANG,OUT
 for LANG in ['cn','en']:
  OUT=STYLE/('figures_'+LANG);OUT.mkdir(exist_ok=True)
  for fn in [mechanism,motivation,width,feedback,cross,realdata,legacy,detection,architecture,reuse]:
   print(LANG,fn.__name__,flush=True);fn()
 (STYLE/'qa/figure_reuse_manifest.json').write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
