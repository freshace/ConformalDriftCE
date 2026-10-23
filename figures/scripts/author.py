"""Build the standalone 10-15 sources from frozen 10-13 sources and verified tables."""
from pathlib import Path
import re,json,shutil
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];ROOT=R.parent;Z=ROOT/'provenance/2026-10-14'
S=json.loads((Z/'tables/paper_numbers.json').read_text(encoding='utf-8'))
C=json.loads((R/'tables/summary.json').read_text(encoding='utf-8'))
POL=['frozen','frozen_trigger','current_coverage','current_residual','periodic','random']
DB=['stats','imdb','ergastf1','genome']
LANG='cn'
def tr(a,b):return a if LANG=='cn' else b
def table(caption,label,headers,rows,fmt=None,continued=False):
 fmt=fmt or 'l'+'r'*(len(headers)-1)
 if continued:caption+=tr('（续）',' (continued)')
 return '\n\\begin{table}[!htbp]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{2pt}\n'+('\\ContinuedFloat\n' if continued else '')+'\\caption{'+caption+'}\n'+('\\label{'+label+'}\n' if label else '')+'\\begin{tabular}{@{}'+fmt+'@{}}\\toprule\n'+' & '.join(headers)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(map(str,row))+r'\\' if isinstance(row,list) else row for row in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
def group(text,n):return '\\multicolumn{'+str(n)+'}{@{}l}{\\textit{'+text+'}}\\\\\n'
def block(text,label,kind):
 for m in re.finditer(r'\\begin\{'+kind+r'\*?\}.*?\\end\{'+kind+r'\*?\}',text,re.S):
  if '\\label{'+label+'}' in m.group():return m.group()
 raise ValueError(label)
def replaceblock(text,label,new,kind='table'):return text.replace(block(text,label,kind),new)
def fblock(text,label):return block(text,label,'figure')
def replace_caption(b,caption):
 # Captions/descriptions here contain no literal newlines.
 b=re.sub(r'\\Description\{[^\n]*\}',lambda m:'\\Description{'+caption+'}',b)
 return re.sub(r'\\caption\{[^\n]*\}',lambda m:'\\caption{'+caption+'}',b)
def single(b,placement='!htbp'):
 b=b.replace('figure*','figure').replace('\\textwidth','\\linewidth')
 return re.sub(r'(\\begin\{figure\})\[[^\]]*\]',lambda m:m[1]+'['+placement+']',b)
def fig(name,caption,label,star=False):
 env='figure*' if star else 'figure';unit='\\textwidth' if star else '\\linewidth'
 return '\n\\begin{'+env+'}[!htbp]\n\\centering\n\\includegraphics[width='+unit+']{figures/'+name+'}\n\\Description{'+caption+'}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{'+env+'}\n'
def stat(ds,pattern,policy):
 return next(x for x in C['offline'] if x['dataset']==ds and x['pattern']==pattern and x['policy']==policy)
def ciwrap(v,scale=100):
 return '\\shortstack{'+f'{v["mean"]*scale:.1f}'+r'\\'+'{['+f'{v["lower"]*scale:.1f},{v["upper"]*scale:.1f}'+']}}'
def policy(p):
 return dict(zip(POL,[tr('不更新','Frozen'),tr('初始参照','Initial'),tr('当前覆盖','Coverage'),tr('当前残差','Residual'),tr('周期','Periodic'),tr('随机','Random')]))[p]
def budget_tables(pattern):
 text=''
 for j,dslist in enumerate([DB[:2],DB[2:]]):
  rows=[]
  for ds in dslist:
   rows.append(group(ds,7))
   for p in POL:
    g=stat(ds,pattern,p)
    rows.append([policy(p),ciwrap(g['coverage']),*[f'{g[k]["mean"]:.2f}' for k in ['width','q50','q90','q99']],f'{g["deployment_labels"]["mean"]:.0f}'])
  cap=tr('连续维护完整对照：','Continuous maintenance: ')+{'stable':tr('稳定','stable'),'gradual':tr('渐变','gradual'),'cycles':tr('重复漂移','repeated drift')}[pattern]+tr('。覆盖为均值及95\\%种子CI，其余为轨迹池化指标的种子均值；标签为实际在线消费。','; coverage is mean and 95\\% seed CI; other metrics average seed-level pooled trajectories. Labels are actual online use.')
  text+=table(cap,'tab:continuous-'+pattern if j==0 else 'tab:continuous-'+pattern+'-continued',[tr('策略','Policy'),tr('覆盖\\%及CI','Cov.\\% / CI'),'$W$','q50','q90','q99',tr('标签','Labels')],rows,continued=j>0)
 return text
def main():
 global LANG
 for LANG in ['cn','en']:
  dest=R/('manuscript_'+LANG);src=ROOT/'paper'/('revision_20261014_zero_'+LANG)
  tx={p.stem:p.read_text(encoding='utf-8') for p in src.glob('*.tex') if p.stem in ['main','intro','related_work','method','experiments','conclusion','appendix']}
  # Standalone abstract: the agreed rhetorical template with current evidence/budgets.
  abstract=tr(
r'''基数估计（Cardinality Estimation, CE）是查询优化器选择执行计划的基础，部署后的数据与工作负载变化要求持续检验估计器的可靠性。本文提出Conformal-DriftCE，一种面向有限、选择性反馈的模型无关可靠性包装层，将校准、监控、修复与独立审计按用途分离，并联合评价覆盖率、区间宽度和点误差。在这一流程中，轻量级残差修正器学习底模的系统性偏差，共形校准在平移后的基数尺度上构造乘性预测区间；监控在进入窗口前固定的既有区间上评估新反馈，修复则更新外挂层，并由独立审计评价效果。我们在真实数据库负载、受控漂移与模拟在线查询流上评估该设计，揭示三类结构性失效机制：同批样本校准与自测使覆盖读数受到秩规则约束，过宽区间掩盖估计偏移，有偏反馈池内的独立划分仍可能偏离部署分布。在stats工作负载中，相关性漂移下500条修复训练标签与500条独立校准标签将经验覆盖率从63.2\%恢复至90.3\%；工作负载漂移下200条训练标签与200条校准标签将覆盖率从52.8\%恢复至89.3\%。四库80窗重复漂移中，当前覆盖参照相对周期更新提高平均审计覆盖2.32--13.30个百分点；真实30窗联合变化中，覆盖为92.7\%、使用830条在线标签，周期更新为88.9\%、1150条。结果表明，用途分离、版本一致的监控与联合质量评价能够支持有限反馈下可检验的按需维护。''',
r'''Cardinality estimation (CE) supports query-plan selection, while changing data and workloads require continuing reliability assessment. We present Conformal-DriftCE, a model-agnostic wrapper for limited, selective feedback that separates calibration, monitoring, repair and independent audit and jointly evaluates coverage, interval width and point error. A lightweight residual corrector learns systematic base-model bias; conformal calibration constructs multiplicative intervals on the shifted-cardinality scale. Monitoring evaluates new feedback against intervals fixed before the window, repair updates the wrapper, and independent audit evaluates the outcome. Real database workloads, controlled shifts and simulated online streams expose three structural failures: same-sample calibration and self-evaluation constrain coverage through ranks; wide intervals hide estimation shifts; and independent splits within biased feedback pools can still differ from deployment traffic. On stats, 500 repair-training and 500 independent calibration labels restore correlation-shift coverage from 63.2\% to 90.3\%; 200 training and 200 calibration labels restore workload-shift coverage from 52.8\% to 89.3\%. Across four databases, 80-window repeated shifts yield 2.32--13.30 percentage-point gains over periodic updating. A real 30-window combined-change replay reaches 92.7\% with 830 online labels, versus 88.9\% and 1150 for periodic updating. Evidence-role separation, version-consistent monitoring and joint quality assessment support testable on-demand maintenance under limited feedback.''')
  tx['main']=re.sub(r'(?<=\\begin\{abstract\}\n).*?(?=\\end\{abstract\})',lambda m:abstract+'\n',tx['main'],flags=re.S)
  tx['main']=tx['main'].replace('\\usepackage{placeins}','\\usepackage{placeins}\n\\usepackage{flafter}')
  # Summary describes the evidence hierarchy, independent of manuscript history.
  oldstart='我们以真实查询负载' if LANG=='cn' else 'Experiments span mechanism controls'
  paragraphs=tx['intro'].split('\n\n')
  for j,p in enumerate(paragraphs):
   if p.startswith(oldstart):
    paragraphs[j]=tr(
r'我们从机制、适用范围和持续维护三个层次评价该设计。四个数据库（stats、imdb、ergastf1和genome）的相关性、工作负载与模式变化检验跨环境修复，四个底模接口检验包装层的适用性；四库各80窗回放检验稳定、渐变和重复漂移下的维护质量与标签支出。真实数据库上的30窗执行反馈实验进一步计量监控、修复与独立审计成本。重复漂移中，当前覆盖参照相对周期更新的平均覆盖增益为2.32--13.30个百分点；真实联合变化中，以830条在线标签达到92.7\%覆盖，周期更新为1150条、88.9\%。覆盖率、宽度、点误差和计划效用共同界定这些维护收益。',
r'We evaluate mechanisms, scope and continuous maintenance. Correlation, workload and schema changes on stats, imdb, ergastf1 and genome test repair across environments, while four base-estimator interfaces test wrapper applicability. Eighty-window replays on all four databases compare maintenance quality and label expenditure under stable, gradual and repeated shifts. A real-database 30-window replay measures feedback acquisition, repair and audit costs. Under repeated shifts, current coverage monitoring gains 2.32--13.30 percentage points over periodic updates; under real combined change, it reaches 92.7\% with 830 online labels, versus 88.9\% and 1150 for periodic updates. Coverage, width, point error and plan utility jointly characterize the outcomes.')
  tx['intro']='\n\n'.join(paragraphs)
  tx['method']=tx['method'].replace('早期严格越界协议保留为补充控制。','固定预测器扫描采用严格越界，并报告独立负窗误报。').replace('Earlier strict-exceedance protocols are supplemental controls.','The fixed-predictor scan uses strict exceedance and reports independent negative-window alarms.')
  tx['method']=tx['method'].replace('早期真实8窗采用全局中位对数残差中心更新。','计划效用审计中的反馈实验采用全局中位对数残差中心更新。').replace('The earlier real eight-window replay uses a global median log-residual center update.','The feedback experiment supporting the plan audit uses a global median log-residual center update.')
  exp=tx['experiments'];app=tx['appendix']
  # Remove superseded controls, extract labeled/feature experiment for adjacent main-text display.
  detstart=app.index('\\paragraph{'+tr('有标签与特征信号.','Labeled and feature signals.')+'}')
  end=app.index('\\section{',detstart);detection=app[detstart:end]
  legacy=app.index('\\section{'+tr('早期在线协议的补充控制','Earlier Online Controls')+'}')
  app=app[:legacy]+app[end:]
  # Place feature-signal experiment directly after data-level figure.
  datafig=fblock(exp,'fig:realdata-correlation')
  exp=exp.replace(datafig,datafig+'\n'+detection+'\n\\FloatBarrier\n')
  # Narrow figures with readable panel arrangements.
  caps={
   'fig:ablation-corrector':tr(r'相近覆盖下的区间几何与冻结覆盖，10次划分。(a) 端点为审计内平均$(1+L)/(1+\hat y)$及$(1+U)/(1+\hat y)$；细线为全部划分，带为端点均值的95\% CI，中间曲线为几何示意。(b) 冻结后的覆盖及95\%划分CI。宽度采用归一化长度$W$；固定预测器扫描见附录。',r'Interval geometry and frozen coverage at similar initial coverage, ten splits. (a) Endpoints average $(1+L)/(1+\hat y)$ and $(1+U)/(1+\hat y)$ within audit; thin curves show splits and bands are 95\% CIs for mean endpoints. Interior curves illustrate geometry. (b) Frozen coverage and 95\% split CIs. Width is normalized length $W$; Appendix reports the fixed-predictor scan.'),
   'fig:feedback-replay':tr(r'反馈选择的整体影响：删旧数据池、60标签、每策略30次划分。(a) 反馈减冻结总体审计覆盖。(b) 修复后总体覆盖。a、b为均值及95\%划分CI。(c) 各划分审计中位$W$及箱体，中线为中位数、箱为IQR，须至1.5 IQR围栏内最远观测；横轴对数。统一评分下的零结果子群审计见附录。',r'Feedback selection in the deletion pool: 60 labels, 30 splits per strategy. (a) Feedback minus frozen overall audit coverage. (b) Repaired overall coverage; a,b show means and 95\% split CIs. (c) Split-level audit-median $W$ and boxes: median, IQR and extreme observations within 1.5-IQR fences, on a log x-axis. Appendix reports the zero-result subgroup under the same score.')}
  for label in ['fig:necessity','fig:failure-modes','fig:ablation-corrector','fig:feedback-replay','fig:realdata-correlation','fig:detection','fig:plan-regret']:
   b=fblock(exp,label);nb=single(b,'H' if label in ['fig:necessity','fig:failure-modes'] else '!htbp')
   if label in caps:nb=replace_caption(nb,caps[label])
   exp=exp.replace(b,nb)
  # Requeue plan figure after its introduction, including the English source's earlier float queue.
  planfig=fblock(exp,'fig:plan-regret');exp=exp.replace(planfig,'');exp=exp.rstrip()+'\n'+planfig
  # Keep point-error/width tradeoffs in the main text and shift subgroup detail to Appendix.
  exp=exp.replace('均匀反馈修复后的总体覆盖为 90.9\\%，零结果子群覆盖为 48.1\\%。覆盖、区间长度和零子群表现需要同时解释，统一评分不会自动消除选择偏差。','均匀反馈修复后的总体覆盖为90.9\\%。覆盖与区间长度共同揭示反馈选择的质量与代价；统一评分下的零结果子群审计见附录。')
  exp=exp.replace('Repaired coverage with uniform feedback is 90.9\\% overall and 48.1\\% for zero results.','Repaired overall coverage with uniform feedback is 90.9\\%; Appendix reports the zero-result subgroup under the same score.')
  # Position section boundaries after preceding floats.
  exp=exp.replace('\\FloatBarrier\n','')
  # Cross-model metrics: same audit, two states per estimator.
  rows=[]
  for base in ['PG','ALECE','PRICE','MSCN']:
   for st in ['frozen','repair']:
    g=S['model_'+base+'_'+st]
    rows.append([base if st=='frozen' else '',tr('冻结','Frozen') if st=='frozen' else tr('修复','Repaired'),f'{g["coverage"][0]*100:.1f}',f'{g["width"][0]:.2f}',f'{g["q90"][0]:.2f}'])
  exp=replaceblock(exp,'tab:crossmodel',table(tr(r'跨底模修复前后联合质量：50次划分、500训练加500独立校准标签，同一审计集。$W$为平均归一化长度；q90为q-error的90分位数。',r'Joint quality before and after repair: 50 splits, 500 training plus 500 calibration labels, shared audit. $W$ is mean normalized length; q90 is the 90th q-error percentile.'),'tab:crossmodel',[tr('底模','Base'),tr('状态','State'),tr('覆盖\\%','Cov.\\%'),'$W$','q90'],rows))
  # Native main table includes all joint metrics; paired error ratios remain an Appendix display.
  ratios=block(exp,'tab:native-matched','table');rows=[]
  for ds in ['stats','imdb']:
   for con in ['correlation','workload']:
    rows.append(group(ds+' / '+(tr('相关','correlation') if con=='correlation' else tr('负载','workload')),5))
    for arm in ['wrapper_repair','native_finetune_calibrated']:
     g=S['native_'+ds+'_'+con+'_'+arm]
     rows.append([tr('残差修复','Residual') if arm=='wrapper_repair' else 'PRICE',f'{100*g["coverage"][0]:.1f}',*[f'{g[k][0]:.2f}' for k in ['width','q50','q90']]])
  exp=exp.replace(ratios,table(tr(r'同预算残差修复与原生PRICE适应。每种设置10次划分，每臂500训练加500校准标签；覆盖、平均$W$及q50/q90均在共享独立审计上评价。配对误差比及95\% CI见附录。',r'Equal-budget residual repair and native PRICE adaptation: ten splits per setting, 500 training plus 500 calibration labels per arm. Coverage, mean $W$ and q50/q90 use a shared independent audit. Appendix reports paired error ratios and 95\% CIs.'),'tab:native-matched',[tr('方法','Method'),tr('覆盖\\%','Cov.\\%'),'$W$','q50','q90'],rows))
  # Continuous section: exact protocol, all four databases, no version comparison.
  start=exp.index('\\subsection{'+tr('连续漂移下的维护效果与预算','Continuous Maintenance and Budget')+'}')
  stop=exp.index('\\subsection{',start+12)
  sub=exp[start:stop];subfig=fblock(sub,'fig:continuous-repair')
  cap=tr(r'四库连续维护，各10个种子。(a--d) 80窗重复漂移的平均独立审计覆盖，误差棒为点态95\%种子CI（2000次重采样，每8窗显示），曲线保留全部80窗；竖线表示分布切换。(e,f) 实际在线标签，灰段为监控、彩段为更新及阈值维护，总高误差棒为95\%种子CI。完整六策略和稳定、渐变对照见附录。',r'Continuous maintenance on four databases, ten seeds each. (a--d) Mean audit coverage over all 80 repeated-shift windows; error bars show pointwise 95\% seed CIs (2000 resamples) every eight windows. Vertical lines mark distribution switches. (e,f) Consumed online labels: gray is monitoring, color is repair and threshold maintenance; total-height bars are 95\% seed CIs. Appendix reports all six policies and stable/gradual controls.')
  subfig=replace_caption(subfig,cap)
  rows=[];summary=[]
  for ds in DB:
   a=stat(ds,'cycles','current_coverage');b=stat(ds,'cycles','periodic')
   p=next(x for x in C['offline_paired'] if x['dataset']==ds and x['pattern']=='cycles' and x['comparator']=='periodic')['coverage_difference']
   rows.append([ds,f'{a["coverage"]["mean"]*100:.1f}',f'{b["coverage"]["mean"]*100:.1f}',f'{p["mean"]*100:.2f}',f'{a["deployment_labels"]["mean"]:.0f}'])
   summary.append(f'{ds}: {p["mean"]*100:.2f} [{p["lower"]*100:.2f}, {p["upper"]*100:.2f}]')
  # Main text holds the additional numbers; Appendix provides full tables.
  detail=tr(
r'''四个数据库各进行80窗回放、10个种子，唯一查询数依次为stats 50,000、imdb 46,672、ergastf1 49,999和genome 49,773。每窗20条监控和50条独立审计；初始化使用4000训练、1000校准和2000阈值验证标签。固定底模和特征，仅将多列过滤真值乘以$e^{\rho_t}$；重复轨迹每16窗使用$\rho=0,2,0,2,0$，另设稳定与渐变对照。修正器为100棵树、31叶、学习率0.1，时间漂移为合成注入。

六策略共享查询身份、初始化、分组中心修复器和8000条更新上限。每次修复使用200训练、200校准和400阈值验证标签，最多10次；触发策略冷却3窗，更新下一窗生效，末窗不更新。对照为不更新、初始参照、当前覆盖、当前残差、周期与预先随机更新，报告实际标签消费。

图~\ref{fig:continuous-repair}中，当前覆盖参照在stats、imdb、ergastf1和genome的平均审计覆盖依次为88.5\%、89.0\%、87.1\%和86.8\%，周期更新为86.2\%、86.4\%、84.5\%和73.5\%。对应的配对覆盖增益及95\%种子CI依次为2.32 [1.58,3.12]、2.58 [2.08,3.07]、2.56 [0.64,4.12]和13.30 [12.28,14.19]个百分点。当前覆盖消耗8800、9120、9040和8160条在线标签，周期策略各为9600条。

当前参照持续测量已更新的部署状态，在分布往返变化时提供按需维护依据。完整稳定、渐变与重复轨迹同时报告覆盖、宽度、点误差及实际标签，用于比较不同反馈条件下的质量与成本（附录~\ref{app:continuous-protocol}）。残差触发作为同标签可用性下的强对照，与覆盖触发共同评价。

覆盖计数与平均残差采用共同10\%单窗误报目标及验证并列随机化。独立负窗实测覆盖/残差误报依次为''',
r'''Each of four databases has 80 windows and ten seeds: stats, imdb, ergastf1 and genome contain 50,000, 46,672, 49,999 and 49,773 unique queries. Each window uses 20 monitoring and 50 independent audit labels; initialization uses 4000 training, 1000 calibration and 2000 threshold-validation labels. Base models and features are fixed; multiple-column responses are multiplied by $e^{\rho_t}$. Repeated shifts use $\rho=0,2,0,2,0$ in 16-window segments, alongside stable and gradual controls. The corrector has 100 trees, 31 leaves and learning rate 0.1; temporal drift is synthetically injected.

Six policies share query identities, initialization, grouped center repair and an 8000-label update cap. Each repair uses 200 training, 200 calibration and 400 threshold-validation labels, with at most ten repairs. Triggered policies have a three-window cooldown; updates apply next window and the final window cannot update. Frozen, initial-reference, current-coverage, current-residual, periodic and prescheduled random policies report actual label use.

In Figure~\ref{fig:continuous-repair}, current coverage monitoring reaches 88.5\%, 89.0\%, 87.1\% and 86.8\% on stats, imdb, ergastf1 and genome; periodic updating reaches 86.2\%, 86.4\%, 84.5\% and 73.5\%. Paired coverage gains and 95\% seed CIs are 2.32 [1.58,3.12], 2.58 [2.08,3.07], 2.56 [0.64,4.12] and 13.30 [12.28,14.19] percentage points. Current coverage uses 8800, 9120, 9040 and 8160 online labels, versus 9600 per database for periodic updating.

The current reference measures the repaired deployment through recurring distribution changes. Complete stable, gradual and repeated-shift results report coverage, width, point error and consumed labels to compare maintenance quality and cost (Appendix~\ref{app:continuous-protocol}). Residual triggering supplies a strong control with the same label availability.

Coverage counts and mean residuals share a 10\% window false-alarm target with validation tie randomization. Independent negative-window coverage/residual alarm rates are ''')
  fprs=[]
  for ds in DB:
   vals=[next(x for x in C['negative_fpr'] if x['dataset']==ds and x['nominal_fpr']==.1 and x['detector']==det)['actual_negative_fpr']['mean']*100 for det in ['coverage','residual']]
   fprs.append(ds+' '+f'{vals[0]:.1f}\\%/{vals[1]:.1f}\\%')
  detail+=tr('、',', ').join(fprs)+tr('。实际误报另行报告，单窗目标不构成时间一致误报保证。','; achieved rates are reported separately, and a per-window target is not a time-uniform guarantee.')
  subhead=sub[:sub.index('\n')];newsub=subhead+'\n'+detail+'\n'+subfig+'\n'
  exp=exp[:start]+newsub+exp[stop:]
  for lab in ['tab:crossmodel','tab:native-matched']:
   old=block(exp,lab,'table');exp=exp.replace(old,old.replace('[!htbp]','[H]'))
  # Eliminate historical framing while retaining the downstream protocol.
  exp=exp.replace('真实8窗反馈闭环及其计划审计采用统一的区间评分','计划效用审计采用统一的区间评分，反馈状态由独立的8窗执行实验建立')
  exp=exp.replace('The earlier real eight-window feedback loop and plan audit use the same interval score','The plan audit uses the common interval score and feedback states established in a separate eight-window execution experiment')
  # Appendix config is anchored after Appendix A heading, in a single column.
  conf=block(app,'tab:experiment-index','table');rows=[]
  entries=[
   (tr('静态机制','Static mechanism'),'20,000 / 3000','500 / 500','50'),
   (tr('宽度匹配','Width matching'),'20,000 / 3000 / 2000 val','4000 test','10'),
   (tr('固定预测器','Fixed predictor'),'20,000 / 3000 / 5000 val','5000 neg / 10,000 pos','30/DB'),
   (tr('跨库相关','Cross-DB corr.'),'20,000 / 3000','500 / 500','20/DB'),
   (tr('跨库负载/模式','Workload/schema'),'1500 / 300; up to 20,000 / 3000','200 / 200','20/DB'),
   ('PRICE','2000 / 500','500 / 500; 2000 audit','10/case'),
   (tr('80窗维护','80-window'),'4000 / 1000 / 2000 val','200 / 200 / 400 val','10/DB'),
   (tr('真实30窗','Real 30-window'),'400 / 200 / 200 val','40 / 80 / 80 val','5')]
  for name,init,repair,reps in entries:
   rows.extend([[name,tr('初始：','Initial: ')+init,reps],['',tr('后续标签：','Subsequent: ')+repair,'']])
  newconf=table(tr('主要实验配置。未另标注时，斜杠依次表示训练、校准、阈值验证标签；val/test/audit分别注明验证、测试和审计。重复以划分或种子计。','Main configurations. Unless marked otherwise, slash-separated counts are training, calibration and threshold-validation labels. Explicit val/test/audit entries denote validation, test and audit. Repetitions are splits or seeds.'),'tab:experiment-index',[tr('实验','Experiment'),tr('标签配置','Label configuration'),tr('重复','Rep.')],rows,fmt='p{2.0cm}p{4.7cm}r').replace('[!htbp]','[H]')
  app=app.replace(conf,newconf)
  app=app.replace('静态实验、12窗控制与80/30窗连续协议分别记录其校准、监控和更新操作。','静态实验、80窗回放、真实30窗维护与计划效用审计分别记录其校准、监控和更新操作。')
  app=app.replace('Static experiments, 12-window controls and 80/30-window continuous protocols record their respective calibration, monitoring and update operations.','Static experiments, 80-window replays, real 30-window maintenance and plan audits record their calibration, monitoring and update operations.')
  app=app.replace('Static, 12-window and continuous 80/30-window protocols document their distinct calibration, monitoring and repair operations.','Static experiments, 80-window replays, real 30-window maintenance and plan audits document their calibration, monitoring and repair operations.')
  app=app.replace('历史静态缓存','静态查询缓存').replace('Historical static caches','Static query caches')
  app=app.replace('早期真实8窗审计','计划效用实验的独立审计').replace('earlier real eight-window audit','independent audit in the plan-utility experiment')
  # All appendix tables are reflowed; no whole-table scaling.
  app=app.replace('table*','table').replace('\\begin{table}[t]','\\begin{table}[!htbp]').replace('{3pt}','{2pt}')
  # Reflow existing source tables while preserving every printed value.
  def parsed(label):
   b=block(app,label,'table');head=b.split('\\midrule\n',1)[0];body=b.split('\\midrule\n',1)[1].split('\\bottomrule',1)[0]
   rows=[x.strip().removesuffix('\\\\').strip().split(' & ') for x in body.strip().splitlines()]
   cap=re.search(r'\\caption\{([^\n]*)\}',b)[1]
   return b,cap,rows
  b,cap,rr=parsed('tab:self-audit-all')
  for row in rr:
   row[0]={'Frozen':tr('冻结','Frozen'),'Recalibrated':tr('重校准','Recal.'),'Repaired':tr('修复','Repaired'),'Inflated':tr('放大预测','Inflated')}[row[0]]
   mean,rest=row[2].split(' ',1);row[2]='\\shortstack{'+mean+r'\\'+'{'+rest+'}}'
  app=app.replace(b,table(cap,'tab:self-audit-all',[tr('状态','State'),tr('自测','Self'),tr('审计及CI','Audit / CI'),'$W$','q50'],rr))
  b,cap,rr=parsed('tab:width-alarm-all');rows=[];last=None
  for row in rr:
   if row[0]!=last:rows.append(group(row[0],6));last=row[0]
   rows.append(row[1:])
  app=app.replace(b,table(cap,'tab:width-alarm-all',[tr('目标\\%','Target\\%'),'$W$',tr('误报\\%','FPR\\%'),tr('检出\\%','Detect\\%'),tr('延迟','Delay'),tr('未检出\\%','Miss\\%')],rows))
  # Keep scanner graphic with the mechanism tables, not a duplicate main figure.
  scan_cap=tr(r'固定预测器的名义覆盖扫描：每库30次划分，均值与95\%划分CI。两面板分别为独立负窗误报和4.48倍位移检出。阈值只使用验证窗口，实际误报另行测量。',r'Fixed-predictor nominal-coverage scan: 30 splits per database, means and 95\% split CIs. Panels show independent negative-window alarms and detection at 4.48x. Thresholds use validation windows only; achieved false alarms are measured separately.')
  app=app.replace('\\subsection{'+tr('反馈预算与零结果子群','Feedback Budget and Zero-result Subgroups')+'}',fig('width_scan',scan_cap,'fig:fixed-scan')+'\n\\FloatBarrier\n\\subsection{'+tr('反馈预算与零结果子群','Feedback Budget and Zero-result Subgroups')+'}')
  # If English subsection capitalization differs, insert via label.
  if '\\label{fig:fixed-scan}' not in app:
   pos=app.rfind('\\subsection{',0,app.index('\\label{app:feedback-budget-supplement}'))
   app=app[:pos]+fig('width_scan',scan_cap,'fig:fixed-scan')+'\n\\FloatBarrier\n'+app[pos:]
  zcap=tr(r'统一评分下的零结果子群审计：删旧数据池、60标签、每策略30次划分。均值及95\%划分CI，虚线为90\%总体目标；所有查询采用相同区间定义。',r'Zero-result subgroup audit under the unified score: deletion pool, 60 labels and 30 splits per strategy. Means and 95\% split CIs; the dashed line marks the global 90\% target. All queries use the same interval definition.')
  fb=block(app,'tab:feedback-budget-all','table')
  app=app.replace(fb,fb+fig('feedback_zero_subgroup',zcap,'fig:feedback-zero'))
  b,cap,rr=parsed('tab:cross-all-quality');rows=[]
  for ds in DB:
   for con in ['corr','work','schema']:
    rows.append(group(ds+' / '+{'corr':tr('相关','correlation'),'work':tr('负载','workload'),'schema':tr('模式','schema')}[con],5))
    for row in rr:
     if row[0]==ds and row[1]==con:rows.append([tr('冻结','Frozen') if row[2]=='frozen' else tr('修复','Repaired')]+row[3:])
  app=app.replace(b,table(cap,'tab:cross-all-quality',[tr('状态','State'),tr('覆盖\\%','Cov.\\%'),'$W$','q50','q90'],rows))
  b,cap,rr=parsed('tab:native-all-arms');rows=[];last=None
  for row in rr:
   key=' / '.join(row[:2])
   if key!=last:rows.append(group(key,5));last=key
   name={'PRICE + calibration':'PRICE','Recalibration':tr('重校准','Recal.'),'Residual repair':tr('残差修复','Residual')}[row[2]]
   rows.append([name]+row[3:])
  app=app.replace(b,table(cap,'tab:native-all-arms',[tr('方法','Method'),tr('覆盖\\%','Cov.\\%'),'$W$','q50','q90'],rows))
  ratio_rows=[]
  for ds in ['stats','imdb']:
   for con in ['correlation','workload']:
    ratio_rows.append(group(ds+' / '+con,2))
    for metric in ['q50','q90']:
     v=S['native_ratio_'+ds+'_'+con+'_'+metric]
     ratio_rows.append([metric,f'{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]'])
  paired=table(tr(r'残差/PRICE的配对几何平均误差比及95\%划分bootstrap CI，沿用正文相同的10份配对划分。',r'Paired geometric-mean residual/PRICE error ratios and 95\% split-bootstrap CIs, using the same ten paired splits as the main comparison.'),'tab:native-paired-ratios',[tr('指标','Metric'),tr('配对比率及CI','Paired ratio / CI')],ratio_rows)
  idx=app.index('\\section{',app.index('\\label{tab:native-all-arms}'))
  app=app[:idx]+paired+'\n\\FloatBarrier\n'+app[idx:]
  for pat in ['stable','gradual','cycles']:
   app=replaceblock(app,'tab:continuous-'+pat,budget_tables(pat))
  b,cap,rr=parsed('tab:real-all-quality');rows=[];last=None
  for row in rr:
   if row[0]!=last:rows.append(group(row[0],7));last=row[0]
   row[1]={'initial':policy('frozen_trigger'),'coverage':policy('current_coverage'),'residual':policy('current_residual')}.get(row[1],policy(row[1]) if row[1] in POL else row[1])
   mean,rest=row[2].split(' ',1);row[2]='\\shortstack{'+mean+r'\\'+'{'+rest+'}}'
   rows.append(row[1:])
  app=app.replace(b,table(cap,'tab:real-all-quality',[tr('策略','Policy'),tr('覆盖\\%及CI','Cov.\\% / CI'),'$W$','q50','q90','q99',tr('标签','Labels')],rows))
  app=app.replace('\\section{'+tr('真实8窗反馈与计划审计','Real Eight-window Feedback and Plan Audit')+'}','\\section{'+tr('计划效用审计协议','Plan-utility Audit Protocol')+'}')
  app=app.replace('早期真实实验在独立PostgreSQL 16.6中','计划效用审计的反馈实验在PostgreSQL 16.6中').replace('The earlier PostgreSQL 16.6 experiment','The feedback experiment supporting plan utility in PostgreSQL 16.6')
  app=app.replace('监控图上的点来自20条反馈，不是独立审计覆盖。','每窗监控使用20条反馈，独立审计另设。')
  app=app.replace('Monitoring points are not independent audit coverage.','Each window uses 20 monitoring labels; independent audit queries are separate.')
  rf=fblock(app,'fig:real-online-supplement');app=app.replace(rf,'')
  app=re.sub(r'图~\\ref\{fig:real-online-supplement\}[^\n]*','',app)
  app=re.sub(r'Figure~\\ref\{fig:real-online-supplement\}[^\n]*','',app)
  # Update scope wording in conclusion without inventing new real SQL databases.
  tx['conclusion']=tx['conclusion'].replace('双数据库80窗','四数据库80窗').replace('two-database 80-window','four-database 80-window')
  app=app.replace('\\section{','\\FloatBarrier\n\\section{')
  
  # Queue broad displays while their section still has text to fill the page.
  for lab,sec_label in [('fig:continuous-repair','sec:exp-labelbudget'),('fig:real-cost','sec:exp-real-online')]:
   b=fblock(exp,lab);exp=exp.replace(b,'')
   pos=exp.index('\\label{'+sec_label+'}');pos=exp.index('\n\n',pos)
   exp=exp[:pos]+'\n'+b+exp[pos:]
  b=fblock(exp,'fig:plan-regret');exp=exp.replace(b,'')
  pos=exp.index('\\label{sec:exp-downstream}');pos=exp.index('\n\n',pos)
  exp=exp[:pos]+'\n'+b+exp[pos:]
  if LANG=='en':
   exp=exp.replace('Uniform-feedback repair reaches 90.9\\% overall coverage and 48.1\\% zero-result subgroup coverage.','Uniform-feedback repair reaches 90.9\\% overall coverage; the Appendix reports the zero-result subgroup under the same score.')

  exp=exp.replace('\\subsection{'+tr('真实执行反馈与维护成本','Real Execution Feedback and Maintenance Cost')+'}', '\\FloatBarrier\n\\subsection{'+tr('真实执行反馈与维护成本','Real Execution Feedback and Maintenance Cost')+'}')
  
  exp=exp.replace('Uniform-feedback repair reaches 90.9\\% overall coverage and 48.1\\% zero-subgroup coverage. Sampling, interval cost and subgroup results must be read together.','Uniform-feedback repair reaches 90.9\\% overall coverage. Coverage and interval length jointly characterize sampling quality and cost; the Appendix reports the zero-result subgroup under the same score.')
  exp=exp.replace('audits feedback coverage bias, repair quality and zero-result coverage.','audits feedback coverage bias, overall repair coverage and interval length.')
  dcap=tr(r'有标签可靠性与特征均值信号的受控对照，50次划分。(a) 有标签覆盖、残差与无标签特征均值的报警率。(b) 独立审计覆盖损失。(c) 特征均值报警率增量。均值及95\%划分CI；相关性注入保持特征固定，尺度控制刷新特征。',r'Controlled labeled-reliability and feature-mean signals, 50 splits. (a) Alarm rates for labeled coverage/residual and unlabeled feature means. (b) Independent-audit coverage loss. (c) Feature-mean alarm increase. Means and 95\% split CIs; correlation injection fixes features, while scale controls refresh them.')
  exp=replaceblock(exp,'fig:detection',replace_caption(fblock(exp,'fig:detection'),dcap),'figure')

  
  # Place the two broad maintenance displays in the text flow, with their lead paragraphs.
  tx['main']=tx['main'].replace('\\usepackage{flafter}','\\usepackage{flafter}\n\\usepackage{cuted}')
  exp=exp.replace('\\FloatBarrier\n\\subsection{'+tr('真实执行反馈与维护成本','Real Execution Feedback and Maintenance Cost')+'}', '\\subsection{'+tr('真实执行反馈与维护成本','Real Execution Feedback and Maintenance Cost')+'}')
  exp=exp.replace('\\subsection{'+tr('连续漂移下的维护效果与预算','Continuous Maintenance and Budget')+'}', '\\FloatBarrier\n\\subsection{'+tr('连续漂移下的维护效果与预算','Continuous Maintenance and Budget')+'}')
  for lab in ['fig:continuous-repair','fig:real-cost']:
   b=fblock(exp,lab)
   nb=re.sub(r'\\begin\{figure\*\}\[[^\]]*\]',lambda m:'\\begin{strip}\\centering',b).replace('\\end{figure*}','\\end{strip}')
   nb=nb.replace('\\caption{','\\captionsetup{hypcap=false}\\captionof{figure}{')
   exp=exp.replace(b,nb)

  
  # Keep the setup compact enough for the motivation figure at the column foot.
  start=exp.index('\\label{sec:exp-setup}')+len('\\label{sec:exp-setup}')
  stop=exp.index('\\subsection{',start)
  setup=tr(r'''
静态主负载为stats的50,000条查询（8表、43列、622维特征）。LightGBM修正器使用300棵树、63叶、学习率0.06；初始训练/校准20,000/3000条，目标覆盖90\%。底模包括PostgreSQL、ALECE、PRICE和MSCN。各实验预先固定查询身份、划分、种子与预算，分离标签用途，统一采用本文评分；执行反馈保留真实零结果。

覆盖按原始真值计算，宽度为归一化长度$W$；q50/q90/q99和GMQ分别为截断q-error分位数及几何均值，pp为百分点。除另述外，CI采用10,000次划分/种子bootstrap，逐窗为点态区间。完整配置见附录~\ref{app:revision-protocol}。

''',r'''
The main static workload has 50,000 stats queries (eight tables, 43 columns, 622 features). LightGBM uses 300 trees, 63 leaves and learning rate 0.06; initial training/calibration uses 20,000/3000 labels and targets 90\% coverage. Bases include PostgreSQL, ALECE, PRICE and MSCN. Experiments predeclare query identities, splits, seeds and budgets, separate label roles and use the common score. Execution feedback retains actual zero results.

Coverage uses original responses and width is normalized length $W$. q50/q90/q99 and GMQ are clipped-q-error quantiles and geometric mean; pp denotes percentage points. Unless specified, 95\% CIs use 10,000 split/seed bootstrap resamples; trajectory CIs are pointwise. Appendix~\ref{app:revision-protocol} gives configurations.

''')
  exp=exp[:start]+setup+exp[stop:]
  b=fblock(exp,'fig:necessity');exp=exp.replace(b,b.replace('width=1.0\\linewidth','width=0.88\\linewidth'))

  
  # Put detailed setup/statistical definitions in A/B and retain a concise main setup.
  a=exp.index('\\label{sec:exp-setup}')+len('\\label{sec:exp-setup}');b=exp.index('\\subsection{',a)
  exp=exp[:a]+tr(r'''
静态主负载为stats的50,000条查询，底模包括PostgreSQL、ALECE、PRICE和MSCN。默认目标覆盖90\%，所有区间采用本文统一评分，执行反馈保留真实零结果。各实验预先固定查询身份、划分、种子与预算，分离标签用途。覆盖率、归一化长度$W$和点误差联合报告；配置与统计口径见附录~\ref{app:revision-protocol}、\ref{app:conformal-proof}。

''',r'''
The main static workload contains 50,000 stats queries; bases include PostgreSQL, ALECE, PRICE and MSCN. Nominal coverage is 90\%. All intervals use the common score, and execution feedback retains actual zeros. Experiments predeclare query identities, splits, seeds and budgets and separate label roles. Coverage, normalized length $W$ and point error are evaluated jointly; Appendices~\ref{app:revision-protocol} and~\ref{app:conformal-proof} give configurations and statistical definitions.

''')+exp[b:]
  intro=tr(r'静态stats负载覆盖8表、43列，采用622维特征。LightGBM使用300棵树、63叶、学习率0.06；初始训练/校准为20,000/3000条。除另述外，95\% CI采用10,000次划分或种子bootstrap，逐窗轨迹为点态区间。q50/q90/q99为截断q-error的分位数，GMQ为其几何均值，pp表示百分点。',r'The static stats workload covers eight tables and 43 columns with 622 features. LightGBM uses 300 trees, 63 leaves and learning rate 0.06; initial training/calibration use 20,000/3000 labels. Unless specified, 95\% CIs use 10,000 split/seed bootstrap resamples; trajectory CIs are pointwise. q50/q90/q99 are clipped-q-error quantiles, GMQ is its geometric mean and pp denotes percentage points.')
  pos=app.index('\n',app.index('\\label{sec:impl}'))+1;app=app[:pos]+intro+'\n\n'+app[pos:]
  cap=tr(r'冻结覆盖与点误差，50次划分。瀑布条为覆盖下降，黑线为GMQ；误差棒为95\%划分bootstrap CI。条上数字为相邻档位覆盖变化（pp），横轴为位移档位。',r'Frozen coverage and point error, 50 splits. Waterfall bars show coverage loss; the black line is GMQ. Error bars are 95\% split-bootstrap CIs. Top labels give adjacent coverage changes (pp); the x-axis denotes shift level.')
  exp=replaceblock(exp,'fig:necessity',replace_caption(fblock(exp,'fig:necessity'),cap),'figure')
  # Compact the plan protocol to avoid an almost-empty trailing page.
  pos=app.index('\\section{'+tr('计划效用审计协议','Plan-utility Audit Protocol')+'}')
  head=app[pos:app.index('\n',pos)+1]
  protocol=tr(r'''
反馈实验在PostgreSQL 16.6中置换过滤列，保留边际值多重集、连接键及旧统计。每条件8窗、每窗20条监控；第3窗切换快照，联合条件同时切换查询形态。首次报警使用500训练加500校准标签作全局中心修复，下一窗生效，每条件至多一次。2320个监控与修复标签均来自完整三表计划一次执行的两表子计划行数，并经独立COUNT核验。

每条件另有100条独立审计SQL。相关对照冻结/修复覆盖为94\%/91\%、中位$W$为2.77/4.07、q50为1.15/1.26；联合变化为76\%/95\%、1.32/4.94、1.34/1.38。

\paragraph{计划选择与计时.}
两表CE查询增加users连接，固定两表Hash子计划和连接顺序，外层候选为Hash、Merge与Nested Loop。Rows提示注入基数，区间内在$\log(1+c)$空间取5个等距场景并加入点预测，规划行数下界为1。请求与达成基数相对误差须不超过5\%；同一候选在各场景中的算子、索引、连接条件及排序须一致。对估计代价$C(p,c)$、候选$\mathcal P$和场景$\mathcal S$，选择规则为
\[
\begin{aligned}p_{\rm point}&=\arg\min_{p\in\mathcal P}C(p,\hat y),\\
p_{\rm int}&=\arg\min_{p\in\mathcal P}\max_{c\in\mathcal S}\frac{C(p,c)}{\min_{p'\in\mathcal P}C(p',c)}.
\end{aligned}
\]
计时前结构核验剔除799项中的37项，保留762项，不读取耗时。每项3候选各按随机顺序串行执行3次，7191条原始记录中6858次计时有效，5秒上限下无超时。每候选取中位耗时；两规则选同一计划时共享计时。最快候选仅用于事后regret评价。CI为状态内10,000次查询配对bootstrap；各状态分母不同，状态间差异不估计总体因果修复收益。

\paragraph{子群评价.}\label{app:groupcov}\label{sec:exp-coverage-boundaries}\label{tab:real-online-audit}
零结果与相关性敏感子群预先定义，分别记录有效查询数及跨划分聚合方式。总体覆盖不推出子群覆盖。
''',r'''
PostgreSQL 16.6 filter-column permutations preserve marginal multisets, join keys and old statistics. Each condition has eight windows of 20 monitoring labels. Window three switches snapshots; combined change also switches query structure. The first alarm triggers at most one 500-training/500-calibration global-center repair, effective next window. All 2320 monitoring/repair labels count two-table subplans from complete three-table executions, independently COUNT-verified.

Each condition also has 100 independent audit queries. Frozen/repaired correlation-control coverage, median $W$ and q50 are 94\%/91\%, 2.77/4.07 and 1.15/1.26; combined-change values are 76\%/95\%, 1.32/4.94 and 1.34/1.38.

\paragraph{Plan selection and timing.}
Queries add a users join while fixing the two-table Hash subplan/order. Outer candidates are Hash, Merge and Nested Loop. Rows hints inject five equally spaced $\log(1+c)$ interval scenarios and the point estimate (planner minimum one). Achieved cardinalities must match requests within 5\%; operators, indexes, join conditions and sorting must match per candidate across scenarios. For cost $C(p,c)$, candidates $\mathcal P$ and scenarios $\mathcal S$,
\[
\begin{aligned}p_{\rm point}&=\arg\min_{p\in\mathcal P}C(p,\hat y),\\
p_{\rm int}&=\arg\min_{p\in\mathcal P}\max_{c\in\mathcal S}\frac{C(p,c)}{\min_{p'\in\mathcal P}C(p',c)}.
\end{aligned}
\]
Structural checks exclude 37 of 799 items before timing, retaining 762. Three candidates each receive three randomized-order serial trials: 7191 raw records yield 6858 valid timings, with no five-second timeout. Runtime comparisons use each candidate\'s median trial time; unchanged decisions share timings. The fastest candidate is a post-hoc regret reference. CIs use 10,000 within-state paired-query bootstrap resamples; differing state denominators preclude overall causal repair comparisons.

\paragraph{Subgroup interpretation.}\label{app:groupcov}\label{sec:exp-coverage-boundaries}\label{tab:real-online-audit}
Predefined zero-result and correlation-sensitive groups record valid counts and aggregation. Overall coverage does not imply subgroup coverage.
''')
  app=app[:pos]+head+protocol

  
  # Every supplementary display and continuation has an explicit textual reference.
  reftext=tr(r'图~\ref{fig:fixed-scan}补充名义覆盖扫描，图~\ref{fig:feedback-zero}报告统一评分下的零结果子群审计。',r'Figure~\ref{fig:fixed-scan} supplements the nominal-coverage scan; Figure~\ref{fig:feedback-zero} reports the zero-result subgroup under the unified score.')
  pos=app.index('\n',app.index('\\label{app:training-budget}'))+1
  app=app[:pos]+reftext+'\n\n'+app[pos:]
  text=tr(r'两种修复臂的配对误差比与置信区间见表~\ref{tab:native-paired-ratios}。',r'Table~\ref{tab:native-paired-ratios} gives paired error ratios and confidence intervals for the two repair arms.')
  pos=app.index('\n',app.index('\\label{app:realdata-full}'))+1
  app=app[:pos]+text+'\n\n'+app[pos:]
  text=tr(r'各库续表分别见表~\ref{tab:continuous-stable-continued}、\ref{tab:continuous-gradual-continued}和\ref{tab:continuous-cycles-continued}。',r'Tables~\ref{tab:continuous-stable-continued}, \ref{tab:continuous-gradual-continued} and~\ref{tab:continuous-cycles-continued} continue the corresponding conditions for the remaining databases.')
  pos=app.index('\n',app.index('\\label{tab:repair-policies}'))+1
  app=app[:pos]+text+'\n\n'+app[pos:]

  
  # Layout follow-up: continue normal body text through former float-only pages.
  exp=exp.replace('\\FloatBarrier\n\\subsection{'+tr('连续漂移下的维护效果与预算','Continuous Maintenance and Budget')+'}', '\\subsection{'+tr('连续漂移下的维护效果与预算','Continuous Maintenance and Budget')+'}')
  for lab,following in [('fig:continuous-repair','sec:exp-real-online'),('fig:real-cost','sec:exp-downstream')]:
   bb=next(m.group() for m in re.finditer(r'\\begin\{strip\}.*?\\end\{strip\}',exp,re.S) if '\\label{'+lab+'}' in m.group())
   exp=exp.replace(bb,'')
   position=exp.rfind('\\subsection{',0,exp.index('\\label{'+following+'}'))
   exp=exp[:position]+bb+'\n'+exp[position:]
  # Restore the actual 10-10 Figure 4 horizontal original design.
  bb=fblock(exp,'fig:ablation-corrector')
  nb=bb.replace('\\begin{figure}[!htbp]','\\begin{figure*}[!t]').replace('\\end{figure}','\\end{figure*}').replace('1.0\\linewidth','1.0\\textwidth')
  cap=tr(r'相近覆盖下的区间余量与失效可见性，10次划分。(a) 喇叭口在平移响应尺度上示意截断前的$[1/w,w]$，$w=e^q$为乘性半宽；实际区间按$y=0$截断。细线和浅色点保留各划分，误差棒变换平均半宽95\% CI的界限，横向展开和填色均为几何示意。(b) 冻结预测器与分位数后的覆盖，均值及95\% CI。图中CI采用5,000次划分bootstrap；覆盖匹配只读取验证集。',r'Interval slack and failure visibility at similar coverage, ten splits. (a) The original fan illustrates pre-clipping $[1/w,w]$ on the shifted-response scale, with multiplicative half-width $w=e^q$; actual intervals clip at $y=0$. Thin curves and faint points retain splits. Error bars transform the 95\% CI of mean half-width; horizontal expansion and shading illustrate geometry. (b) Frozen-predictor/quantile coverage, means and 95\% CIs. Figure CIs use 5,000 split-bootstrap resamples; coverage matching uses validation only.')
  nb=replace_caption(nb,cap);exp=exp.replace(bb,nb)

  
  # Use ordinary floats so the following prose fills available page space.
  # The strip output routine cannot safely coexist with queued full-width floats.
  tx['main']=tx['main'].replace('\\usepackage{cuted}\n','')
  for lab,sec_label in [('fig:continuous-repair','sec:exp-labelbudget'),('fig:real-cost','sec:exp-real-online')]:
   bb=next(m.group() for m in re.finditer(r'\\begin\{strip\}.*?\\end\{strip\}',exp,re.S) if '\\label{'+lab+'}' in m.group())
   exp=exp.replace(bb,'')
   bb=bb.replace('\\begin{strip}\\centering','\\begin{figure*}[!t]').replace('\\end{strip}','\\end{figure*}').replace('\\captionsetup{hypcap=false}\\captionof{figure}','\\caption')
   position=exp.index('\n\n',exp.index('\\label{'+sec_label+'}'))
   exp=exp[:position]+'\n'+bb+exp[position:]
  tx['main']=tx['main'].replace('\\raggedbottom','\\raggedbottom\n\\setcounter{topnumber}{3}\\setcounter{dbltopnumber}{3}\\setcounter{totalnumber}{6}\n\\renewcommand{\\topfraction}{0.95}\\renewcommand{\\dbltopfraction}{0.95}\\renewcommand{\\textfraction}{0.05}')

  



  # Let body text, conclusion and references fill pages around the queued displays.
  tx['main']=tx['main'].replace('\\input{experiments}\n\\FloatBarrier\n\\input{conclusion}\n\\FloatBarrier','\\input{experiments}\n\\input{conclusion}')
  bb=fblock(exp,'fig:feedback-replay')
  exp=exp.replace(bb,bb.replace('[!htbp]','[!t]').replace('width=1.0\\linewidth','width=0.90\\linewidth'))
  bb=fblock(exp,'fig:cross-dataset');exp=exp.replace(bb,'')
  position=exp.index('\n\n',exp.index('\\label{sec:exp-crossmodel}'))
  exp=exp[:position]+'\n'+bb+exp[position:]





  # Continue existing text through the appendix boundary instead of flushing floats.
  tx['main']=tx['main'].replace('\\clearpage\n\\input{appendix}','\\input{appendix}')
  app=app.replace('\\FloatBarrier\n','')

  tx['experiments']=exp;tx['appendix']=app
  # Each source is self-contained; remove manuscript-history phrases still present.
  for name,text in tx.items():
   text=text.replace('earlier real eight-window','plan-audit eight-window').replace('Earlier real eight-window','Plan-audit eight-window')
   (dest/(name+'.tex')).write_text(text,encoding='utf-8')
  shutil.copy2(src/'references.bib',dest/'references.bib')
 print('Standalone bilingual 10-15 sources written.')
if __name__=='__main__':
 main()
 import subprocess,sys
 subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'apply_annotations.py')],check=True)


