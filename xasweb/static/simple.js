'use strict';
const $ = id => document.getElementById(id);
const state = {datasets:[],supports:[],files:[],index:0,busy:false,templates:[]};
const current = () => state.datasets[state.index];
const modeNames = {light:'轻元素配位',heavy:'硫 / 卤素配位',metal:'金属近邻',combined:'组合模型'};
function analysisState(d) {
  const attempt=d.previous_attempt || d.result;
  if(d.lastError)return 'retryable';
  if(attempt?.required?.length)return 'needs_input';
  // HTTP 200 can contain completed spectra but no usable numerical fit.
  if(attempt && !attempt.structure?.fit)return 'retryable';
  return attempt?'completed':'pending';
}
function error(message='') { $('error').textContent=message; $('error').hidden=!message; }
function options(el, entries) { el.replaceChildren(...entries.map(([v,t])=>new Option(t,v))); }
async function api(url,options={}) {
  const response=await fetch(url,options);
  if(!response.ok) { const body=await response.json().catch(()=>({})); throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail || response.status)); }
  return response.json();
}
const post=(url,data)=>api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
function controls() {
  $('file').disabled=state.busy;
  $('import-label').setAttribute('aria-disabled',String(state.busy));
  $('export').disabled=state.busy || !state.datasets.some(d=>d.result);
  for(const el of document.querySelectorAll('#context-form input,#context-form select,#file-list select')) el.disabled=state.busy;
  const step=state.busy?'process':current()?.result?'results':state.datasets.length?'process':'upload';
  for(const name of ['upload','process','results']) { const el=$('step-'+name);el.classList.toggle('active',name===step);if(name===step)el.setAttribute('aria-current','step');else el.removeAttribute('aria-current'); }
}
function supportPayload() {
  return [...state.supports.filter(s=>s.text).map(({name,role,text})=>({name:name.slice(0,200),role,text})),
    ...state.datasets.filter(d=>d.purpose==='reference').map(d=>({name:d.filename.slice(0,200),role:'reference_spectrum',text:d.raw_text}))];
}
function renderFiles() {
  $('uploads').hidden=!state.files.length;$('file-list').replaceChildren();
  for(const s of state.supports) { const row=document.createElement('p');row.textContent=s.name+' · '+({cif:'晶体结构',xyz:'原子坐标',feff_input:'FEFF 输入',feff_path:'FEFF 路径',sample_info:'样品说明',support:'支持材料'}[s.role] || s.role);$('file-list').append(row); }
  for(const d of state.datasets) {
    const row=document.createElement('label');row.className='file-row';const text=document.createElement('span');text.textContent=d.label;
    const select=document.createElement('select');select.setAttribute('aria-label',d.label+' 文件用途');options(select,[['spectrum','实验光谱'],['reference','参考谱']]);select.value=d.purpose;
    select.addEventListener('change',async()=>{d.purpose=select.value;await runAll();});row.append(text,select);$('file-list').append(row);
  }
}
async function upload(files) {
  if(state.busy || !files.length)return;
  if(state.files.length+files.length>100 || state.files.reduce((n,f)=>n+f.size,0)+files.reduce((n,f)=>n+f.size,0)>100000000) {error('当前会话最多 100 个文件、100 MB。');return;}
  state.busy=true;error();controls();$('status').textContent='正在识别光谱、路径和支持材料…';const failures=[];
  try {
    for(const file of files) {
      try {
        const form=new FormData();form.append('file',file);const imported=await api('/api/import',{method:'POST',body:form});state.files.push(file);
        if(imported.support)state.supports.push({...imported.support,file});
        for(const d of imported.datasets || []) {
          if(d.kind==='support' || !d.rows) {state.supports.push({name:file.name,role:'support',text:d.source_preview || '',file});continue;}
          d.file=file;d.purpose=d.role==='reference' || /#\s*Data.role:\s*reference/i.test(d.raw_text)?'reference':'spectrum';
          d.id=String(state.datasets.length);state.datasets.push(d);
        }
      } catch(err) {failures.push(file.name+'：'+err.message);}
    }
    if(state.datasets.length)state.index=state.datasets.length-1;
    renderFiles();render();
  } finally {state.busy=false;$('file').value='';controls();}
  if(failures.length)error(failures.join('\n'));
  await runAll();
}
$('file').addEventListener('change',event=>upload([...event.target.files]));
$('import-label').addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();if(!state.busy)$('file').click();}});
for(const name of ['dragenter','dragover'])$('upload-panel').addEventListener(name,event=>{event.preventDefault();$('upload-panel').classList.add('dragging');});
for(const name of ['dragleave','drop'])$('upload-panel').addEventListener(name,event=>{event.preventDefault();$('upload-panel').classList.remove('dragging');});
$('upload-panel').addEventListener('drop',event=>upload([...event.dataTransfer.files]));
async function runOne(d) {
  const request={raw_text:d.raw_text,filename:d.filename.slice(0,200),assessment_token:d.assessment_token,supports:supportPayload(),...(d.context || {})};
  try {
    const result=await post('/api/workflow',request);
    const oldArrays=arraysOf(d.result),newArrays=arraysOf(result);
    if(d.result && oldArrays.chi && !newArrays.chi) {
      d.previous_attempt=result;d.lastError='本次处理未完成，保留此前已完成结果。'+(result.stop_reasons || []).join('；');
    } else {d.result=result;d.selectedFit=result.structure?.selected_fit_id;d.lastError='';d.previous_attempt=null;}
  } catch(err) {d.lastError='本次分析未完成：'+err.message+'。已保留此前结果与原始输入。';}
}
async function runAll(only=null) {
  if(state.busy)return;
  const targets=only?[only]:state.datasets.filter(d=>d.purpose==='spectrum');
  state.busy=true;controls();
  try {
    for(const [i,d] of targets.entries()) {
      const started=Date.now();
      const update=()=>{$('status').textContent=`正在分析 ${i+1}/${targets.length}：${d.label} · 光谱、四类候选与稳定性检查 · ${Math.floor((Date.now()-started)/1000)} 秒`;};
      update();const timer=setInterval(update,1000);
      try {await runOne(d);} finally {clearInterval(timer);}
      if(current()===d)render();
    }
  } finally {
    state.busy=false;render();controls();
    const outcomes=state.datasets.filter(d=>d.purpose==='spectrum').map(analysisState);
    $('status').textContent=!targets.length?'支持材料已保留，请上传实验光谱。':outcomes.includes('needs_input')?'自动推断未完成，请手动补齐元素或吸收边；填写后会自动拟合。':outcomes.includes('retryable')?'系统已自动尝试拟合；未获得可用结果时请查看停止原因。修改元素或吸收边后会自动重试。':'本轮分析完成；候选用于初筛，未强行确定唯一结构。';
  }
}
$('spectrum-select').addEventListener('change',()=>{state.index=Number($('spectrum-select').value);render();});
function arraysOf(result) { const p=result?.processing;return p?.kind==='mu'?p.data.result?.arrays || {}:p?.data || {}; }
let contextTimer;
function scheduleContext(immediate=false,field=null) {
  clearTimeout(contextTimer);const d=current();if(!d || state.busy || d.purpose!=='spectrum')return;
  const element=$('element').value.trim(),edge=$('edge').value;
  d.context={...(d.context || {})};
  // Displayed inferred/header values are not manual overrides. Only the edited
  // field (or both fields on explicit form submission) becomes a manual value.
  for(const key of field?[field]:['element','edge']) {
    const value=key==='element'?element:edge;
    if(value)d.context[key]=value;else delete d.context[key];
  }
  if(element && !/^[A-Za-z]{1,2}$/.test(element))return;
  // Confirmation is an explicit retry, including identical element/edge values.
  if(immediate){runAll(d);return;}
  contextTimer=setTimeout(()=>{if(current()===d)runAll(d);},250);
}
$('context-form').addEventListener('submit',event=>{event.preventDefault();scheduleContext(true);});
$('context-form').addEventListener('keydown',event=>{
  if(event.key==='Enter' && !event.isComposing){event.preventDefault();scheduleContext(true);}
});
$('element').addEventListener('input',()=>scheduleContext(false,'element'));$('edge').addEventListener('change',()=>scheduleContext(true,'edge'));
$('candidate-select').addEventListener('change',()=>{const d=current();d.selectedFit=$('candidate-select').value;renderStructure(d.result?.structure);});
function appendText(parent,text,tag='p') {const el=document.createElement(tag);el.textContent=text;parent.append(el);return el;}
function parameter(p,digits=4) {
  if(p?.value==null)return '未确定'+(p?.estimate==null?'':`（诊断估计 ${Number(p.estimate).toPrecision(4)}${p.estimate_stderr==null?'':` ± ${Number(p.estimate_stderr).toPrecision(2)}`}）`);
  if(p.status==='assumed')return Number(p.value).toFixed(digits)+'（假设值，固定）';
  const precision=p.stderr>0?Math.min(8,Math.max(digits,1-Math.floor(Math.log10(p.stderr)))):digits;
  const range=Array.isArray(p.assumption_range)&&p.assumption_range.length===2&&p.assumption_range.every(Number.isFinite)?`；假设范围 ${p.assumption_range.map(v=>Number(v).toPrecision(4)).join('–')}`:'';
  return Number(p.value).toFixed(precision)+(p.stderr==null?'':` ± ${Number(p.stderr).toPrecision(2)}`)+(p.status==='quick_estimate'?`（快速估计值${range}）`:range);
}
function render() {
  const d=current();$('workspace').hidden=!d;if(!d){controls();return;}
  options($('spectrum-select'),state.datasets.map((d,i)=>[String(i),d.label]));$('spectrum-select').value=String(state.index);$('selector-wrap').hidden=state.datasets.length<2;
  $('spectrum-name').textContent=d.label;
  const r=d.result,attempt=d.previous_attempt || r,outcome=analysisState(d),context={...(attempt?.context || {}),...(d.context || {})};
  const identification=r?.input_identification || d.input_identification,s=identification?.selection;
  $('spectrum-info').textContent=`${d.rows} 点 · ${d.columns} 列 · ${d.purpose==='reference'?'参考谱':'实验光谱'}${s?` · ${s.kind==='chi'?'χ(k) / Å⁻¹':(s.mode==='mu'?'μ(E)':s.mode==='transmission'?'透射强度':'荧光强度')+' / '+s.energy_unit}`:''}`;
  $('element').value=context.element || '';$('edge').value=context.edge || '';
  $('context-details').hidden=d.purpose==='reference';
  const automatic=['element','edge'].some(key=>!d.context?.[key] && context.provenance?.[key]?.type==='automatic');
  $('context-hint').textContent=attempt?.required?.length?'自动推断未完成 · 请手动补齐':outcome==='retryable'?'可重试 · 修改后重新分析':context.element?`${context.element} ${context.edge || ''}${automatic?' · 自动推断':''}`:'缺失时自动推断';
  $('context-reason').textContent=(automatic?'已根据能量谱自动补齐元素或吸收边并继续初筛。':'')+'可修改元素或吸收边，修改后自动重新分析；手动值优先，推断来源随结果导出。保持相同选择时，可按回车重试。';
  if(attempt?.required?.length || outcome==='retryable')$('context-details').open=true;
  $('results').hidden=!r;$('stage-summary').textContent=d.lastError || (outcome==='retryable'?'系统已自动尝试拟合，但未获得可用结果；已保留光谱处理和停止原因。修改元素或吸收边后会自动重试。':r?identification?.assumptions?.length?'输入识别包含假设，已随结果记录。':'自动识别完成，处理范围与候选结果如下。':d.purpose==='reference'?'此文件作为参考材料参与实验光谱的分析。':'准备自动分析');
  if(!r){controls();return;}
  const a=arraysOf(r),p=r.processing,v=r.spectral_views || {};
  const stages=[a.energy?'吸收谱':null,a.norm?'归一化':null,a.chi?'χ(k)':null,a.chir_mag?'FT':null,v.wavelet?.status==='completed'?'小波图':null,r.structure?.fit?'候选拟合与检查':'模型评估'].filter(Boolean);
  $('processing-summary').textContent=stages.join(' → ')+'。R 空间图未校正相移，峰位不能直接作为键长。';
  $('plots').replaceChildren();
  if(a.energy && a.mu)$('plots').append(chart(a.energy,a.mu,'能量空间吸收谱 μ(E)','能量 / eV'));
  if(a.energy && a.norm)$('plots').append(chart(a.energy,a.flat || a.norm,'能量空间归一化吸收谱','能量 / eV'));
  for(const [weight,title] of [[0,'k 空间 EXAFS 振荡谱 χ(k)（未加权）'],[1,'k 空间一阶加权 EXAFS 振荡谱 kχ(k)'],[2,'k 空间二阶加权 EXAFS 振荡谱 k²χ(k)'],[3,'k 空间三阶加权 EXAFS 振荡谱 k³χ(k)']]) {
    const y=v.weighted?.[weight] || (a.chi && a.k?.map((k,i)=>a.chi[i]*k**weight));
    if(y)$('plots').append(chart(a.k,y,title,'k / Å⁻¹'));
  }
  for(const [key,title] of [['mag','R 空间傅里叶变换模值 |χ(R)|'],['re','R 空间傅里叶变换实部 Re χ(R)'],['im','R 空间傅里叶变换虚部 Im χ(R)']])if(a.r && a['chir_'+key])$('plots').append(chart(a.r,a['chir_'+key],title+' · 未校正相移','R / Å',null,6));
  if(v.wavelet?.status==='completed')$('plots').append(heatmap(v.wavelet));
  else if(v.wavelet?.reason)appendText($('plots'),'小波图未完成：'+v.wavelet.reason);
  $('processing-record').textContent=JSON.stringify({identification,parameters:p?.kind==='mu'?p.data.result?.resolved_calls || p.data.request?.parameters:p?.data?.parameters,
    assessment:p?.kind==='mu'?p.data.result?.assessment:p?.data?.assessment,record:p?.data?.result?.processing_record,wavelet:v.wavelet?{parameters:v.wavelet.parameters,notes:v.wavelet.notes}:null,references:r.reference_checks},null,2);
  renderStructure(r.structure);
  const stops=[...(r.stop_reasons || []),...(d.lastError?[d.lastError]:[])];$('stops').hidden=!stops.length;$('stop-list').replaceChildren();for(const stop of stops)appendText($('stop-list'),stop,'li');
  controls();
}
function renderStructure(structure={}) {
  structure=structure || {};const d=current(),fits=structure.fits || (structure.fit?[structure.fit]:[]);
  const fit=fits.find(f=>f.id===d.selectedFit) || structure.fit;
  $('candidate-wrap').hidden=!fits.length;
  $('candidate-select').replaceChildren();
  for(const [mode,name] of Object.entries(modeNames)) {
    const group=document.createElement('optgroup');group.label=name;
    for(const f of fits.filter(f=>f.mode===mode)) group.append(new Option(`${f.model} · ${f.source?.type==='hypothetical_reference'?'参考簇假设':'上传模型'} · R-factor ${f.statistics.rfactor.toPrecision(3)}${f.unresolved_parameters.length?' · 待复核':''}`,f.id));
    if(group.childElementCount)$('candidate-select').append(group);
  }
  if(fit)$('candidate-select').value=fit.id;
  $('mode-summary').textContent=(structure.modes || []).map(m=>`${m.name}：${m.status==='fitted'?m.fitted+' 个候选':m.status==='stopped'?'未获得可用拟合':'等待有效光谱与吸收边'}`).join(' · ');
  $('structure-result').replaceChildren();$('fit-plots').replaceChildren();
  const quick=Boolean(structure.first_shell_window);
  $('fit-badge').textContent=fit?fit.unresolved_parameters.length?'快速初估 · 部分未确定':quick?'第一壳层快速初估':'条件结果':'未确定';
  $('structure-status').textContent=fit?`${fit.model} · ${fit.source?.type==='hypothetical_reference'?'假设参考簇':'用户提供的参考环境'}。N 依赖 S₀² 和模型假设；候选排名不代表唯一结构。`:'每条实验光谱均进入模型评估。缺少证据或计算未完成时，结构参数保持未确定。';
  if(fit) {
    const wrapper=document.createElement('div');wrapper.className='table-scroll';const table=document.createElement('table');const head=document.createElement('tr');
    for(const t of ['路径 / 壳','N（条件）','R / Å','σ² / Å²','ΔE₀ / eV'])appendText(head,t,'th');table.append(head);
    for(const shell of fit.shells){const tr=document.createElement('tr');for(const t of [shell.pair,parameter(shell.N,2),parameter(shell.R),parameter(shell.sigma_squared,5),parameter(shell.energy_shift,2)])appendText(tr,t,'td');table.append(tr);}wrapper.append(table);$('structure-result').append(wrapper);
    appendText($('structure-result'),`R-factor ${fit.statistics.rfactor?.toPrecision(4)} · 约化 χ² ${fit.statistics.reduced_chi_square?.toPrecision(3) ?? '未确定'} · AICc ${fit.statistics.aicc?.toFixed(1)} · 独立点数 ${fit.statistics.n_independent.toFixed(1)} / 自由参数 ${fit.statistics.n_variables}`);
    appendText($('structure-result'),`拟合范围 k = ${fit.fit_range.kmin.toFixed(2)}–${fit.fit_range.kmax.toFixed(2)} Å⁻¹，R = ${fit.fit_range.rmin.toFixed(2)}–${fit.fit_range.rmax.toFixed(2)} Å。${quick?'残差仅评价第一配位峰区间。':''}± 为模型内标准误差；假设范围不是置信区间；“未确定”后的估计仅作诊断。`,'p').className='muted';
    const c=fit.curves,weight=fit.fit_range.kweight;
    $('fit-plots').append(chart(c.data_k,c.data_chi,'实验与拟合 k²χ(k) · 绿：实验，橙：模型','k / Å⁻¹',[{x:c.data_k,y:c.data_chi.map((v,i)=>v*c.data_k[i]**weight),color:'#246451'},{x:c.k,y:c.chi_fit.map((v,i)=>v*c.k[i]**weight),color:'#ce7441'}],quick?fit.fit_range.kmax:Infinity,quick?fit.fit_range.kmin:-Infinity));
    for(const [key,title] of [['mag','|χ(R)|'],['re','Re χ(R)'],['im','Im χ(R)']])if(c['data_'+key])$('fit-plots').append(chart(c.data_r,c['data_'+key],(quick?'第一壳层实验与拟合 ':'实验与拟合 ')+title,'R / Å',[{x:c.data_r,y:c['data_'+key],color:'#246451'},{x:c.r,y:c['fit_'+key],color:'#ce7441'}],quick?fit.fit_range.rmax:6,quick?fit.fit_range.rmin:-Infinity));
    for(const warning of fit.warnings)appendText($('structure-result'),warning).className='warning';
  }
  $('assumptions').replaceChildren();for(const assumption of fit?.assumptions || structure.assumptions || [])appendText($('assumptions'),assumption);
  if(fit?.source?.geometry)appendText($('assumptions'),'当前参考几何：'+fit.source.geometry);
  $('template-list').replaceChildren();for(const t of structure.template_assessment || state.templates)appendText($('template-list'),`${t.name} · ${t.paths} · ${t.status==='qualified_candidate'?'已计算匹配路径':'尚无可用路径'}`);
  const compact={...structure};delete compact.fit;delete compact.fits;
  compact.candidates=(structure.candidates || []).map(c=>({...c,paths:c.paths?.map(({text,...p})=>p),generation:c.generation?{input_sha256:c.generation.input_sha256,executable_sha256:c.generation.executable_sha256}:null}));
  $('model-record').textContent=JSON.stringify(compact,null,2);
  $('fit-diagnostics').textContent=fit?JSON.stringify({parameters:fit.parameters,initial_trials:fit.initial_trials,stability:fit.stability,statistics:fit.statistics,first_shell_window:structure.first_shell_window,constraint_strategy:fit.constraint_strategy,fixed_parameters:fit.fixed_parameters,unconstrained_attempts:fit.unconstrained_attempts,constrained_attempts:fit.constrained_attempts,assumption_sensitivity:fit.assumption_sensitivity,unresolved:fit.unresolved_parameters,report:fit.report},null,2):'尚无可报告的拟合诊断。';
}
function download(blob,name) {const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=name;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);}
$('export').addEventListener('click',async()=>{
  if(state.busy)return;state.busy=true;controls();
  try {
    const files=[];
    for(const file of state.files){const bytes=new Uint8Array(await file.arrayBuffer());let binary='';for(let i=0;i<bytes.length;i+=8192)binary+=String.fromCharCode(...bytes.subarray(i,i+8192));files.push({name:file.name,encoding:'base64',data:btoa(binary)});}
    const report={schema_version:3,kind:'webxas-session',created_at:new Date().toISOString(),sources:files,supports:state.supports.map(({file,...s})=>s),
      spectra:state.datasets.map(d=>({filename:d.filename,purpose:d.purpose,original_raw_text:d.raw_text,context:{...(d.previous_attempt?.context || d.result?.context || {}),...(d.context || {})},manual_context:d.context || {},selected_fit_id:d.selectedFit,result:d.result || null,last_attempt:d.previous_attempt || null,last_error:d.lastError || null}))};
    download(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}),'webxas-results.json');
  } catch(err){error('导出失败：'+err.message);}finally{state.busy=false;controls();}
});
api('/api/path-templates').then(data=>{state.templates=data.templates;}).catch(err=>error('路径模板清单暂不可用：'+err.message));
controls();
function chart(x, y, title, axis, curves = null, maxX = Infinity, minX = -Infinity) {
  const panel=document.createElement('div');appendText(panel,title).className='plot-label';
  const lines=(curves || [{x,y,color:'#246451'}]).map(c=>({...c,x:c.x.filter(v=>v>=minX&&v<=maxX),y:c.y.filter((v,i)=>c.x[i]>=minX&&c.x[i]<=maxX)}));
  if(lines.some(c=>!c.x.length || c.x.length!==c.y.length || ![...c.x,...c.y].every(Number.isFinite)))return panel;
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
  svg.setAttribute('viewBox','0 0 480 260');svg.setAttribute('class','chart');svg.setAttribute('role','img');svg.setAttribute('aria-label',title);
  const [loX,hiX]=lines.flatMap(c=>c.x).reduce(([lo,hi],v)=>[Math.min(lo,v),Math.max(hi,v)],[Infinity,-Infinity]);
  const [loY,hiY]=lines.flatMap(c=>c.y).reduce(([lo,hi],v)=>[Math.min(lo,v),Math.max(hi,v)],[Infinity,-Infinity]);
  const px=v=>62+(v-loX)/(hiX-loX || 1)*395,py=v=>210-(v-loY)/(hiY-loY || 1)*180;
  function node(tag,attrs,text) {const el=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);if(text!==undefined)el.textContent=text;svg.append(el);}
  for(let i=0;i<=4;i++) {
    const yy=loY+(hiY-loY)*i/4,xx=loX+(hiX-loX)*i/4;
    node('line',{x1:62,y1:py(yy),x2:457,y2:py(yy),stroke:'#e6eeea'});
    node('text',{x:56,y:py(yy)+4,'text-anchor':'end','font-size':12,fill:'#77867e'},Number(yy.toPrecision(3)));
    node('text',{x:px(xx),y:230,'text-anchor':'middle','font-size':12,fill:'#77867e'},Number(xx.toPrecision(4)));
  }
  for(const c of lines)node('path',{d:c.x.map((v,i)=>`${i?'L':'M'}${px(v).toFixed(2)},${py(c.y[i]).toFixed(2)}`).join(' '),fill:'none',stroke:c.color,'stroke-width':1.5});
  node('text',{x:457,y:254,'text-anchor':'end','font-size':13,fill:'#5d7167'},axis);panel.append(svg);return panel;
}
function heatmap(w) {
  const panel=document.createElement('div');panel.className='wavelet-panel';appendText(panel,'k–R 空间小波变换图（二阶加权 χ(k)，Cauchy 模值）').className='plot-label';
  const canvas=document.createElement('canvas');canvas.width=960;canvas.height=420;canvas.setAttribute('role','img');canvas.setAttribute('aria-label','χ(k) 小波变换图，横轴 k，纵轴未校正相移的 R，颜色表示小波模值');canvas.className='wavelet';panel.append(canvas);
  const ctx=canvas.getContext('2d'),left=80,top=22,width=750,height=330;
  let peak=0;for(const row of w.magnitude)for(const value of row)peak=Math.max(peak,value);
  const stops=[[.99,.99,.95],[.74,.89,.83],[.24,.63,.55],[.10,.35,.44],[.12,.15,.31]];
  function color(value) {const u=Math.max(0,Math.min(1,value))*4,i=Math.min(3,Math.floor(u)),t=u-i;return `rgb(${stops[i].map((c,j)=>Math.round(255*(c+(stops[i+1][j]-c)*t))).join(',')})`;}
  const tile=document.createElement('canvas');tile.width=w.k.length;tile.height=w.r.length;const pixels=tile.getContext('2d');
  for(let r=0;r<w.r.length;r++)for(let k=0;k<w.k.length;k++){pixels.fillStyle=color(w.magnitude[r][k]/(peak || 1));pixels.fillRect(k,w.r.length-r-1,1,1);}
  ctx.drawImage(tile,left,top,width,height);ctx.strokeStyle='#d9e5df';ctx.strokeRect(left,top,width,height);ctx.font='17px Segoe UI';ctx.fillStyle='#62766b';
  const k0=w.k[0],k1=w.k.at(-1),r0=w.r[0],r1=w.r.at(-1);
  for(let i=0;i<=5;i++) {ctx.textAlign='center';ctx.fillText((k0+(k1-k0)*i/5).toFixed(1),left+width*i/5,top+height+28);ctx.textAlign='right';ctx.fillText((r0+(r1-r0)*i/5).toFixed(1),left-10,top+height-height*i/5+5);}
  ctx.textAlign='center';ctx.fillText('k / Å⁻¹',left+width/2,405);ctx.save();ctx.translate(24,top+height/2);ctx.rotate(-Math.PI/2);ctx.fillText('R / Å · 未校正相移',0,0);ctx.restore();
  for(let y=0;y<height;y++){ctx.fillStyle=color(1-y/height);ctx.fillRect(858,top+y,20,1);}
  ctx.fillStyle='#62766b';ctx.textAlign='left';ctx.font='14px Segoe UI';ctx.fillText(peak.toPrecision(3),888,top+10);ctx.fillText('0',888,top+height);ctx.fillText('|WT|',855,top+height+28);
  const caption=appendText(panel,'颜色为线性模值；边缘受窗口影响。小波图用于比较局域散射特征，不独立鉴定元素。');caption.className='muted';
  const readout=appendText(panel,'移动指针查看 k、R 与模值。');readout.className='muted wavelet-readout';
  canvas.addEventListener('pointermove',event=>{const bounds=canvas.getBoundingClientRect(),x=(event.clientX-bounds.left)*canvas.width/bounds.width,y=(event.clientY-bounds.top)*canvas.height/bounds.height;
    if(x<left || x>left+width || y<top || y>top+height)return;
    const k=Math.min(w.k.length-1,Math.floor((x-left)/width*w.k.length)),r=Math.min(w.r.length-1,Math.floor((top+height-y)/height*w.r.length));
    readout.textContent=`k = ${w.k[k].toFixed(2)} Å⁻¹ · R = ${w.r[r].toFixed(2)} Å · |WT| = ${w.magnitude[r][k].toPrecision(4)}`;});
  return panel;
}
