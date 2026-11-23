'use strict';
const $=id=>document.getElementById(id);
const esc=value=>String(value??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const uid=prefix=>prefix+'_'+crypto.randomUUID().replaceAll('-','');
const state={project:null,datasetId:null,selected:new Set(),paths:new Set(),model:null,dirty:false,job:null,run:null,arrayCache:new Map()};
const parameterFields=[['e0','E₀ / eV',''],['energy_shift','仪器能量偏移 / eV',0],['pre1','前沿下界 / eV',-150],['pre2','前沿上界 / eV',-30],['norm1','归一化下界 / eV',150],['norm2','归一化上界 / eV',700],['nnorm','归一化阶数',2],['rbkg','背景 R / Å',1],['bkg_kweight','背景 k 权重',1],['bkg_kmax','背景最大 k',''],['kmin','FT 最小 k / Å⁻¹',2],['kmax','FT 最大 k / Å⁻¹',12],['kweight','FT k 权重',2],['dk','FT 窗口渐变 dk',1]];
const backgroundFields=[['ek0','E→k 原点 / eV',''],['bkg_nclamp','端点约束点数',2],['bkg_clamp_lo','低 k 约束权重',1],['bkg_clamp_hi','高 k 约束权重',1]];
for(const [id,fields] of [['processing-fields',parameterFields],['background-fields',backgroundFields]]){
 $(id).innerHTML=fields.map(([key,label,value])=>`<label>${label}<input id="pp-${key}" type="number" step="any" value="${value}"></label>`).join('');
}
const processingFields=[...parameterFields,...backgroundFields];
function status(message){$('status').textContent=message;}
function fail(error){$('error').textContent=error.message||String(error);$('error').hidden=false;status('操作未完成，请检查上方说明。');}
function clearError(){$('error').hidden=true;}
function messageFrom(data){if(data.message)return `${data.code?data.code+'：':''}${data.message}`;if(Array.isArray(data.detail))return data.detail.map(v=>`${(v.loc||[]).join('.')}：${v.msg}`).join('\n');return data.detail||JSON.stringify(data);}
async function request(path,options={}){const response=await fetch(path,options);if(!response.ok){let detail;try{detail=messageFrom(await response.json());}catch{detail=await response.text();}throw new Error(detail||`HTTP ${response.status}`);}return response.json();}
const json=(method,value)=>({method,headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
const url=tail=>`/api/pro/projects/${state.project.id}${tail}`;
const activeDataset=()=>state.project?.datasets.find(d=>d.id===state.datasetId);
const latestProcessing=id=>state.project?.processing.filter(p=>p.dataset_id===id).at(-1);
const option=(value,label,current)=>`<option value="${esc(value)}"${value===current?' selected':''}>${esc(label)}</option>`;
function guarded(fn){return async event=>{
 event?.preventDefault();
 const exclusive=event?.currentTarget?.id!=='cancel-job';
 if(exclusive&&state.uiBusy){status('请等待当前操作完成。');return;}
 if(exclusive)state.uiBusy=true;
 clearError();available();
 try{await fn(event);}catch(error){fail(error);}finally{if(exclusive)state.uiBusy=false;available();}
};}
function available(){
 document.body.classList.toggle('has-project',Boolean(state.project));document.body.classList.toggle('has-model',Boolean(state.model));
 const yes=!!state.project,busy=!!state.job||!!state.uiBusy;
 for(const id of ['export-project','refresh-project'])$(id).disabled=!yes||busy;
 for(const id of ['upload-spectra','import-path','import-structure'])$(id).disabled=!yes||busy;
 for(const id of ['create-project','open-project','open-file','open-demo'])$(id).disabled=busy;
 for(const id of ['save-metadata','process-data','save-calibration'])$(id).disabled=!activeDataset()||busy;
 $('generate-paths').disabled=!yes||!state.project.structures.length||busy;
 $('build-model').disabled=!yes||!state.project.paths.length||busy;
 for(const id of ['add-parameter','validate-model','save-model','apply-windows'])$(id).disabled=!state.model||busy;
 $('fit-model').disabled=!state.model||state.dirty||state.modelValidation?.fit_allowed===false||busy;
 $('sensitivity-model').disabled=!state.model||state.dirty||state.modelValidation?.fit_allowed===false||busy;
 $('export-report').disabled=!state.run||busy;
 $('compare-models').disabled=(state.project?.runs.filter(r=>r.result).length||0)<2||busy;
 for(const element of document.querySelectorAll('#metadata-form input,#metadata-form select,#processing-form input,#processing-form select,#calibration-form input,#calibration-form select,[data-param] input,[data-param] select,[data-binding] input,[data-dataset-binding] input,[data-dataset-binding] select,[data-dataset],[data-joint],[data-path]'))element.disabled=busy;
 for(const id of ['model-select','run-select','result-dataset','contribution-path'])$(id).disabled=busy;
}
function setProject(project){if(state.project?.id!==project.id){setProStage(location.hash.slice(1));state.fitRenderGeneration=(state.fitRenderGeneration||0)+1;state.run=null;state.model=null;state.dirty=false;state.arrayCache.clear();for(const id of ['parameter-table','binding-table','dataset-bindings-table','processing-plots','fit-plots','estimate-table','correlation-table','result-metrics','result-warnings'])$(id).innerHTML='';$('model-json').value='';$('draft-state').textContent='尚无模型草稿';$('result-state').textContent='尚无运行';$('result-context').textContent=project.metadata?.not_experimental_data?'当前是同引擎合成演示，仅验证软件集成，不代表实验精度。':'每次运行保留提交时的模型快照。';}state.project=project;$('auto-return').href='/auto?project='+encodeURIComponent(project.id);localStorage.setItem('webxas-pro-project',project.id);$('project-title').textContent=project.title;$('project-id').value=project.id;$('revision').textContent=`版本 ${project.revision}`;$('project-info').textContent=`${project.datasets.length} 条数据 · ${project.paths.length} 条路径`;
if(!project.datasets.some(d=>d.id===state.datasetId))state.datasetId=project.datasets[0]?.id||null;
state.selected=new Set([...state.selected].filter(id=>project.datasets.some(d=>d.id===id)));if(!state.selected.size&&state.datasetId)state.selected.add(state.datasetId);
state.paths=new Set([...state.paths].filter(id=>project.paths.some(p=>p.id===id)));if(!state.paths.size)for(const p of project.paths)state.paths.add(p.id);
renderDatasets();renderCalibrationControls();renderMetadata();renderPaths();renderHistory();renderSensitivity();$('structure-select').innerHTML='<option value="">选择结构</option>'+project.structures.map(s=>option(s.id,s.id,s.id)).join('');available();}
function renderDatasets(){$('dataset-list').innerHTML=state.project.datasets.map(d=>`<div class="dataset-row ${d.id===state.datasetId?'active':''}"><input type="checkbox" aria-label="联合选择 ${esc(d.filename)}" data-joint="${d.id}" ${state.selected.has(d.id)?'checked':''}><button data-dataset="${d.id}">${esc(d.filename)}<span class="dataset-state">${esc(d.element||'待声明')} ${esc(d.edge||'')} · ${latestProcessing(d.id)?'已有处理版本':'未处理'}</span></button></div>`).join('')||'<p class="empty">导入第一份光谱</p>';
$('dataset-list').querySelectorAll('[data-dataset]').forEach(button=>button.onclick=guarded(async()=>{state.datasetId=button.dataset.dataset;renderDatasets();renderMetadata();available();await renderProcessing();}));
$('dataset-list').querySelectorAll('[data-joint]').forEach(box=>box.onchange=()=>{box.checked?state.selected.add(box.dataset.joint):state.selected.delete(box.dataset.joint);if(state.model)markDirty('数据选择已变，请重建或检查模型。');});
const issues=state.project.migration_report||[];$('migration').innerHTML=issues.length?`<div class="notice"><strong>工程迁移说明</strong>${issues.map(x=>`<p>${esc(x.message||x)}</p>`).join('')}</div>`:'';}
function renderMetadata(){const d=activeDataset();$('dataset-name').textContent=d?.filename||'未选择数据';$('raw-preview').textContent=d?.raw_text?.slice(0,10000)||'无原始文本';if(!d)return;
$('data-kind').value=d.kind==='chi'?'chi':'mu';$('data-element').value=d.element||'';$('data-edge').value=d.edge||'K';$('sample-id').value=d.sample_id||'';for(const key of ['energy','signal','reference'])$('map-'+key).value=(d.mapping?.[key]??({energy:0,signal:1,reference:2}[key]))+1;$('map-mode').value=d.mapping?.mode||'mu';$('energy-unit').value=d.mapping?.energy_unit||'eV';$('input-kweight').value=d.metadata?.input_kweight??0;$('input-kweight-confirmed').checked=!!d.metadata?.input_kweight_confirmed;$('conditions').value=Object.keys(d.conditions||{}).length?JSON.stringify(d.conditions):'';$('path-element').value=d.element||'';$('path-edge').value=d.edge||'K';const p=latestProcessing(d.id);$('processing-calibration').value=p?.calibration_id||'';for(const [key,,value]of processingFields)$('pp-'+key).value=p?.parameters?.[key]??value;$('pp-bkg-uncertainty').checked=!!p?.parameters?.bkg_uncertainty;$('normalization-only').checked=!!p?.metadata?.normalization_only;$('processing-summary').textContent=p?`处理版本 ${p.id}`:'尚未处理';}
function renderPaths(){const paths=state.project.paths;$('path-list').innerHTML=paths.length?`<table><thead><tr><th>使用</th><th>路径</th><th>元素与边</th><th>类型</th><th>有效半程长 / Å</th><th>简并度</th></tr></thead><tbody>${paths.map(p=>`<tr><td><input type="checkbox" data-path="${p.id}" ${state.paths.has(p.id)?'checked':''} aria-label="选择路径 ${esc(p.id)}"></td><td>${esc(p.id)}<small>${esc(p.absorber_site)}</small></td><td>${esc(p.absorber_element)} ${esc(p.edge)}</td><td>${p.nleg===2?'单散射':`多重散射 (${p.nleg} legs)`}</td><td>${p.reff.toFixed(4)}</td><td>${p.degen}</td></tr>`).join('')}</tbody></table>`:'<p class="empty">工程尚无 FEFF 路径</p>';$('path-list').querySelectorAll('[data-path]').forEach(box=>box.onchange=()=>{box.checked?state.paths.add(box.dataset.path):state.paths.delete(box.dataset.path);});}
function renderHistory(){const previous=$('model-select').value;$('model-select').innerHTML='<option value="">选择模型</option>'+state.project.models.map(m=>option(m.id,m.title||m.id,previous)).join('');const selected=state.run?.id;$('run-select').innerHTML='<option value="">选择历史运行</option>'+state.project.runs.filter(r=>r.result).map(r=>option(r.id,`${r.id} · ${r.result.assessment}`,selected)).join('');}
function markDirty(reason='模型草稿已修改，保存后运行。'){state.dirty=true;state.modelValidation=null;$('draft-state').textContent='未保存的模型草稿';$('result-context').textContent=reason+' 已有图表仍属于原运行。';available();}
function loadModel(model,saved=true){setProStage('model-section');state.model=structuredClone(model);state.modelValidation=null;state.dirty=!saved;state.selected=new Set(model.bindings.map(b=>b.dataset_id));state.paths=new Set(model.bindings.flatMap(b=>b.paths.map(p=>p.path_id)));renderDatasets();renderPaths();$('model-title').value=model.title;const t=model.bindings[0]?.transform;if(t)for(const key of ['kmin','kmax','rmin','rmax','kweight'])$('fit-'+key).value=t[key];renderModel();$('draft-state').textContent=saved?'已保存模型':'未保存的模型草稿';available();}
function renderModel(){const m=state.model;if(!m)return;
$('dataset-bindings-table').innerHTML=`<table><thead><tr><th>数据集</th><th>k min</th><th>k max</th><th>R min</th><th>R max</th><th>k 权重</th><th>噪声方式</th><th>噪声 σ</th></tr></thead><tbody>${m.bindings.map((b,i)=>`<tr data-dataset-binding="${i}"><td>${esc(state.project.datasets.find(d=>d.id===b.dataset_id)?.filename||b.dataset_id)}</td>${['kmin','kmax','rmin','rmax','kweight'].map(key=>`<td><input type="number" step="any" data-transform="${key}" value="${b.transform[key]}"></td>`).join('')}<td><select data-noise="source">${['unknown','engine_estimate','explicit_scalar','explicit_array'].map(key=>option(key,{unknown:'未声明',engine_estimate:'引擎估计',explicit_scalar:'显式标量',explicit_array:'已保存数组'}[key],b.noise.source)).join('')}</select></td><td><input type="number" step="any" data-noise="value" value="${b.noise.value??''}"></td></tr>`).join('')}</tbody></table>`;
const owners=[...m.bindings.map(b=>[b.dataset_id,state.project.datasets.find(d=>d.id===b.dataset_id)?.filename||b.dataset_id]),...m.bindings.flatMap(b=>b.paths.map(p=>[p.id,p.id]))];
$('parameter-table').innerHTML=`<table><thead><tr><th>参数 ID</th><th>类型</th><th>值 / 初值</th><th>最小</th><th>最大</th><th>表达式</th><th>单位</th><th>作用域</th><th>所属对象</th><th></th></tr></thead><tbody>${m.parameters.nodes.map((p,i)=>`<tr data-param="${i}"><td><input data-field="id" class="id" value="${esc(p.id)}" aria-label="参数 ID ${i+1}"></td><td><select data-field="kind">${['free','fixed','expr'].map(k=>option(k,{free:'自由',fixed:'固定',expr:'关联'}[k],p.kind)).join('')}</select></td><td><input data-field="value" type="number" step="any" value="${p.value??''}"></td><td><input data-field="minimum" type="number" step="any" value="${p.minimum??''}"></td><td><input data-field="maximum" type="number" step="any" value="${p.maximum??''}"></td><td><input data-field="expression" class="expr" value="${esc(p.expression||'')}"></td><td><select data-field="unit">${['1','eV','angstrom','angstrom2','angstrom-1','K'].map(u=>option(u,u,p.unit)).join('')}</select></td><td><select data-field="scope">${['global','dataset','path'].map(s=>option(s,{global:'全局',dataset:'数据集',path:'路径'}[s],p.scope)).join('')}</select></td><td><select data-field="owner_id">${option('','全局无所属',p.owner_id||'')}${owners.map(([id,label])=>option(id,label,p.owner_id)).join('')}</select></td><td><button data-remove-param="${i}" class="secondary compact">删除</button></td></tr>`).join('')}</tbody></table>`;
$('binding-table').innerHTML=`<table><thead><tr><th>数据 / 路径实例</th><th>启用</th><th>振幅 s02</th><th>ΔE₀ / eV</th><th>ΔR / Å</th><th>σ² / Å²</th><th>几何依据</th></tr></thead><tbody>${m.bindings.flatMap((b,bi)=>b.paths.map((p,pi)=>`<tr data-binding="${bi}:${pi}"><td>${esc(state.project.datasets.find(d=>d.id===b.dataset_id)?.filename)}<small>${esc(p.id)} → ${esc(p.path_id)}</small></td><td><input type="checkbox" data-expr="enabled" ${p.enabled?'checked':''}></td>${['s02','e0','deltar','sigma2'].map(key=>`<td><input data-expr="${key}" class="expr" value="${esc(p.expressions[key]||'0')}"></td>`).join('')}<td><input data-expr="geometry_basis" value="${esc(p.geometry_basis||'')}"></td></tr>`)).join('')}</tbody></table>`;
for(const input of document.querySelectorAll('#parameter-table input,#parameter-table select,#binding-table input,#dataset-bindings-table input,#dataset-bindings-table select'))input.addEventListener('change',()=>markDirty());
document.querySelectorAll('[data-remove-param]').forEach(button=>button.onclick=guarded(async()=>{readModel();state.model.parameters.nodes.splice(Number(button.dataset.removeParam),1);renderModel();markDirty();}));$('model-json').value=JSON.stringify(m,null,2);}
function readModel(){const m=state.model;if(!m)throw new Error('先建立或选择模型。');m.title=$('model-title').value.trim()||'专业 EXAFS 模型';for(const row of $('parameter-table').querySelectorAll('[data-param]')){const p=m.parameters.nodes[Number(row.dataset.param)];for(const field of row.querySelectorAll('[data-field]')){const key=field.dataset.field;p[key]=['value','minimum','maximum'].includes(key)?(field.value===''?null:Number(field.value)):field.value||null;}if(p.kind==='expr'){p.value=null;p.minimum=null;p.maximum=null;}else p.expression=null;if(p.kind==='fixed'){p.minimum=null;p.maximum=null;}if(p.scope==='global')p.owner_id=null;}
for(const row of $('binding-table').querySelectorAll('[data-binding]')){const [bi,pi]=row.dataset.binding.split(':').map(Number);const p=m.bindings[bi].paths[pi];for(const field of row.querySelectorAll('[data-expr]')){const key=field.dataset.expr;if(key==='enabled')p.enabled=field.checked;else if(key==='geometry_basis')p.geometry_basis=field.value;else p.expressions[key]=field.value;}}
for(const row of $('dataset-bindings-table').querySelectorAll('[data-dataset-binding]')){const b=m.bindings[Number(row.dataset.datasetBinding)];for(const input of row.querySelectorAll('[data-transform]'))b.transform[input.dataset.transform]=Number(input.value);const source=row.querySelector('[data-noise="source"]').value;if(source==='explicit_scalar')b.noise={source,value:Number(row.querySelector('[data-noise="value"]').value),unit:'chi',processing_revision_id:b.processing_revision_id};else if(source!==b.noise.source)b.noise={source,unit:'chi',processing_revision_id:b.processing_revision_id};}$('model-json').value=JSON.stringify(m,null,2);return m;}
function buildModel(){const datasets=state.project.datasets.filter(d=>state.selected.has(d.id));if(!datasets.length)throw new Error('勾选至少一条独立测量数据。');const selectedPaths=state.project.paths.filter(p=>state.paths.has(p.id));if(!selectedPaths.length)throw new Error('勾选至少一条 FEFF 路径。');const m={id:uid('model'),revision:0,title:$('model-title').value,bindings:[],parameters:{nodes:[]},coordination_groups:[],notes:'固定几何起始模板；S02=1的条件结果，单散射共享关系须由分析人员确认。'};const add=(id,unit,scope,owner,value,kind='free',min=null,max=null)=>{if(!m.parameters.nodes.some(p=>p.id===id))m.parameters.nodes.push({id,label:id,unit,scope,owner_id:owner,kind,value,minimum:min,maximum:max,expression:null,physical_basis:kind==='fixed'?'起始模型固定条件':'专家可编辑的小幅修正'});};
datasets.forEach((d,index)=>{const processed=latestProcessing(d.id);if(!processed)throw new Error(`${d.filename} 尚未完成预处理。`);const paths=selectedPaths.filter(p=>p.absorber_element===d.element&&p.edge===d.edge);if(!paths.length)throw new Error(`${d.filename} 没有匹配元素与吸收边的路径。`);const prefix='d'+index,shared=$('share-structure').checked;add(prefix+'_s02','1','dataset',d.id,1,'fixed');add(prefix+'_e0','eV','dataset',d.id,0,'free',-20,20);const hasSS=paths.some(p=>p.nleg===2);if(hasSS){add(shared?'shared_dr':prefix+'_dr','angstrom',shared?'global':'dataset',shared?null:d.id,0,'free',-0.3,0.3);add(shared?'shared_sigma2':prefix+'_sigma2','angstrom2',shared?'global':'dataset',shared?null:d.id,0.003,'free',0,0.03);}if(paths.some(p=>p.nleg>2))add(prefix+'_ms_sigma2','angstrom2','dataset',d.id,0.003,'fixed');m.bindings.push({dataset_id:d.id,processing_revision_id:processed.id,paths:paths.map((p,pi)=>({id:`path_d${index}_${pi}`,path_id:p.id,enabled:true,expressions:{s02:prefix+'_s02',e0:prefix+'_e0',deltar:p.nleg===2?(shared?'shared_dr':prefix+'_dr'):'0',sigma2:p.nleg===2?(shared?'shared_sigma2':prefix+'_sigma2'):prefix+'_ms_sigma2'},amplitude_semantics:'scale',cn_group_id:null,geometry_basis:p.nleg>2?'固定FEFF几何；MS ΔR=0起点，按实际几何完善':'单散射小幅距离修正，需检查各壳层适用性'})),transform:{fitspace:'r',kmin:Number($('fit-kmin').value),kmax:Number($('fit-kmax').value),rmin:Number($('fit-rmin').value),rmax:Number($('fit-rmax').value),kweight:Number($('fit-kweight').value),dk:1,window:'hanning',kstep:0.05,nfft:2048},noise:processed.noise||{source:'engine_estimate',value:null,array:null,unit:'chi',processing_revision_id:null,estimation_range:null,assumptions:[]}});});loadModel(m,false);markDirty();}
async function runJob(job){state.job=job;$('cancel-job').disabled=false;$('job-panel').hidden=false;available();try{while(true){const record=await request(url(`/jobs/${job.job_id}`));$('job-state').textContent=({queued:'等待计算',running:'计算中',cancelled:'已取消',timed_out:'已超时',interrupted:'运行中断',failed:'运行失败',succeeded:'计算完成'}[record.state]||record.state);$('job-stage').textContent=record.heartbeat?.stage||'';if(record.heartbeat?.progress!=null)$('job-progress').value=record.heartbeat.progress;else $('job-progress').removeAttribute('value');if(['succeeded','failed','cancelled','timed_out','interrupted'].includes(record.state)){if(record.state!=='succeeded')throw new Error(record.error?.message||record.state);const project=await request(url(`/jobs/${job.job_id}/commit`),{method:'POST'});setProject(project);status('结果已保存到工程。');return project;}await new Promise(resolve=>setTimeout(resolve,350));}}finally{state.job=null;$('cancel-job').disabled=true;available();}}
async function loadArray(ref,projectId=state.project.id){if(!ref)return null;const key=projectId+':'+ref.sha256;if(!state.arrayCache.has(key)){const result=await request(`/api/pro/projects/${projectId}/arrays?path=`+encodeURIComponent(ref.path));state.arrayCache.set(key,result.values??result);}return state.arrayCache.get(key);}
function plot(title,x,series){if(!x?.length)return '';const valid=series.filter(s=>s.y?.length===x.length);if(!valid.length)return '';let xmin=Math.min(...x),xmax=Math.max(...x),ymin=Infinity,ymax=-Infinity;for(const {y}of valid)for(const v of y){if(Number.isFinite(v)){ymin=Math.min(ymin,v);ymax=Math.max(ymax,v);}}if(!Number.isFinite(ymin)||xmin===xmax)return '';if(ymin===ymax){ymin-=1;ymax+=1;}const W=600,H=240,L=52,T=15,PW=530,PH=185;const sx=v=>L+(v-xmin)/(xmax-xmin)*PW,sy=v=>T+(ymax-v)/(ymax-ymin)*PH;const lines=valid.map(s=>`<polyline points="${x.map((v,i)=>Number.isFinite(s.y[i])?`${sx(v).toFixed(2)},${sy(s.y[i]).toFixed(2)}`:'').filter(Boolean).join(' ')}" fill="none" stroke="${s.color}" stroke-width="1.6"/>`).join('');const ticks=[0,0.5,1].map(f=>`<text x="${L+f*PW}" y="224" font-size="11" fill="#748579" text-anchor="middle">${(xmin+f*(xmax-xmin)).toPrecision(3)}</text><text x="45" y="${T+f*PH+4}" font-size="11" fill="#748579" text-anchor="end">${(ymax-f*(ymax-ymin)).toPrecision(3)}</text>`).join('');return `<div class="plot"><h4>${esc(title)}</h4><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}"><path d="M${L} ${T}V${T+PH}H${L+PW}" fill="none" stroke="#c8d7cd"/>${ticks}${lines}</svg><div class="legend">${valid.map(s=>`<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join('')}</div></div>`;}
async function renderProcessing(){const p=latestProcessing(state.datasetId);if(!p){$('processing-plots').innerHTML='';return;}const arrays={};for(const key of ['energy','mu','norm','k','chi','r','chir_re','chir_im','chir_mag'])if(p.arrays[key])arrays[key]=await loadArray(p.arrays[key]);$('processing-plots').innerHTML=plot('χ(k) · k / Å⁻¹',arrays.k,[{name:'数据',y:arrays.chi,color:'#25745a'}])+plot('复数 FT · R / Å',arrays.r,[{name:'实部',y:arrays.chir_re,color:'#25745a'},{name:'虚部',y:arrays.chir_im,color:'#b8753f'}])+plot('归一化吸收谱 · E / eV',arrays.energy,[{name:'归一化',y:arrays.norm,color:'#25745a'}]);}
async function showRun(run){setProStage('result-section');state.run=run;const result=run.result;if(!result)return;$('result-state').textContent=`${run.task_state} · ${result.assessment}`;$('result-context').textContent=`运行 ${run.id} · 模型 ${run.model_id} · ${result.uncertainty_basis}；${state.dirty?'当前草稿与该快照不同。':'图表来自该运行快照。'}`;const s=result.statistics;$('result-metrics').innerHTML=[['R-factor',s.rfactor],['约化 χ²',s.reduced_chisquare],['Nind',s.n_independent],['自由变量',s.n_variables]].map(([label,value])=>`<div class="metric"><small>${label}</small><strong>${value==null?'不可用':Number(value).toPrecision(4)}</strong></div>`).join('');$('result-warnings').innerHTML=[!result.converged?'优化未收敛：'+result.termination_reason:'',...(result.warnings||[])].filter(Boolean).map(w=>`<p class="notice">${esc(w)}</p>`).join('');$('estimate-table').innerHTML=`<table><thead><tr><th>参数</th><th>值</th><th>统计误差</th><th>单位</th><th>状态</th></tr></thead><tbody>${result.parameters.map(p=>`<tr><td>${esc(p.id)}</td><td>${p.value.toPrecision(6)}</td><td>${p.stderr==null?'不可估计':p.stderr.toPrecision(3)}</td><td>${esc(p.unit)}</td><td>${esc(p.at_bound?'触界':p.uncertainty_reason||'')}</td></tr>`).join('')}</tbody></table>`;$('result-dataset').innerHTML=result.datasets.map(d=>option(d.dataset_id,state.project.datasets.find(x=>x.id===d.dataset_id)?.filename||d.dataset_id,result.datasets[0]?.dataset_id)).join('');await renderFit();renderHistory();available();}
async function renderFit(){
 const generation=state.fitRenderGeneration=(state.fitRenderGeneration||0)+1;
 const run=state.run,projectId=state.project.id;
 const d=run?.result?.datasets.find(d=>d.dataset_id===$('result-dataset').value);
 if(!d){$('fit-plots').innerHTML='';return;}
 $('fit-plots').innerHTML='<p class="hint">正在加载所选谱的运行快照…</p>';
 const a={};for(const [key,ref]of Object.entries(d.arrays))a[key]=await loadArray(ref,projectId);
 if(generation!==state.fitRenderGeneration)return;
 const transform=run.input_snapshot.model.bindings.find(b=>b.dataset_id===d.dataset_id).transform;
 const kIndices=a.k.map((v,i)=>v>=transform.kmin-0.5&&v<=transform.kmax+0.5?i:-1).filter(i=>i>=0);
 const rIndices=a.r.map((v,i)=>v<=transform.rmax+1?i:-1).filter(i=>i>=0);
 const cut=(values,indices)=>values?indices.map(i=>values[i]):null;
 const k=cut(a.k,kIndices),r=cut(a.r,rIndices),data=cut(a.data_chi,kIndices),model=cut(a.model_chi,kIndices);
 const residual=data&&model?data.map((v,i)=>v-model[i]):null;
 const contributions=Object.keys(d.path_contributions||{}),previous=$('contribution-path').value;
 $('contribution-path').innerHTML=contributions.map(id=>option(id,id,contributions.includes(previous)?previous:contributions[0])).join('')||'<option value="">无路径输出</option>';
 const selected=d.path_contributions?.[$('contribution-path').value];let contributionPlot='';
 if(selected){const x=await loadArray(selected.k,projectId),y=await loadArray(selected.chi,projectId);contributionPlot=plot('选定路径 χ(k) 贡献',x,[{name:$('contribution-path').value,y,color:'#8d709b'}]);}
 if(generation!==state.fitRenderGeneration)return;
 $('fit-plots').innerHTML=plot('χ(k) · k / Å⁻¹',k,[{name:'数据',y:data,color:'#25745a'},{name:'模型',y:model,color:'#ce7d3e'}])+plot('χ(k) 残差',k,[{name:'数据 − 模型',y:residual,color:'#6b749b'}])+plot('FT 实部 · R / Å',r,[{name:'数据',y:cut(a.data_re,rIndices),color:'#25745a'},{name:'模型',y:cut(a.model_re,rIndices),color:'#ce7d3e'}])+plot('FT 虚部 · R / Å',r,[{name:'数据',y:cut(a.data_im,rIndices),color:'#25745a'},{name:'模型',y:cut(a.model_im,rIndices),color:'#ce7d3e'}])+contributionPlot;
 const covariance=run.result.covariance;
 if(covariance?.matrix){const names=covariance.var_names,matrix=covariance.matrix;$('correlation-table').innerHTML=`<p class="hint">${esc(covariance.scaling_description)}</p><table><thead><tr><th>参数</th>${names.map(n=>`<th>${esc(n)}</th>`).join('')}</tr></thead><tbody>${matrix.map((row,i)=>`<tr><th>${esc(names[i])}</th>${row.map((v,j)=>`<td>${matrix[i][i]>0&&matrix[j][j]>0?(v/Math.sqrt(matrix[i][i]*matrix[j][j])).toFixed(3):'不可估计'}</td>`).join('')}</tr>`).join('')}</tbody></table>`;}else $('correlation-table').textContent=covariance?.missing_reason||'本次运行未获得可用协方差。';
}
async function download(response,name){if(!response.ok){let data;try{data=await response.json();}catch{throw new Error(`HTTP ${response.status}`);}throw new Error(messageFrom(data));}const object=URL.createObjectURL(await response.blob());const link=document.createElement('a');link.href=object;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(object),10000);}
$('create-project').onclick=guarded(async()=>{state.datasetId=null;state.selected.clear();state.model=null;state.run=null;setProject(await request('/api/pro/projects',json('POST',{title:$('new-title').value.trim()||'XAS 分析工程'})));status('工程已创建。');});
$('open-project').onclick=guarded(async()=>{state.model=null;state.run=null;setProject(await request('/api/pro/projects/'+$('project-id').value.trim()));await renderProcessing();status('工程已打开。');});
$('refresh-project').onclick=guarded(async()=>{setProject(await request(url('')));await renderProcessing();status('工程已刷新。');});
$('open-file').onclick=()=>$('project-file').click();$('project-file').onchange=guarded(async()=>{const file=$('project-file').files[0];if(!file)return;const form=new FormData();form.append('file',file);state.model=null;state.run=null;setProject(await request('/api/pro/projects/import',{method:'POST',body:form}));await renderProcessing();status('工程文件已导入；请核对迁移说明。');$('project-file').value='';});
$('export-project').onclick=guarded(async()=>{await download(await fetch(url('/export?expected_revision='+state.project.revision),{method:'POST'}),'webxas-project.zip');status('工程已导出。');});
$('upload-spectra').onclick=()=>$('spectra-file').click();$('spectra-file').onchange=guarded(async()=>{for(const file of $('spectra-file').files){const form=new FormData();form.append('file',file);setProject(await request(url('/datasets?expected_revision='+state.project.revision),{method:'POST',body:form}));}state.datasetId=state.project.datasets.at(-1)?.id;renderMetadata();renderDatasets();available();status('光谱已导入，请确认数据声明。');$('spectra-file').value='';});
$('metadata-form').onsubmit=guarded(async()=>{const d=structuredClone(activeDataset());d.kind=$('data-kind').value;d.element=$('data-element').value.trim()||null;d.edge=$('data-edge').value;d.sample_id=$('sample-id').value.trim()||null;d.conditions=$('conditions').value.trim()?JSON.parse($('conditions').value):{};for(const key of ['energy','signal','reference'])d.mapping[key]=Number($('map-'+key).value)-1;d.mapping.mode=$('map-mode').value;d.mapping.energy_unit=$('energy-unit').value;if(d.kind==='chi'&&!$('input-kweight-confirmed').checked)throw new Error('请确认输入 χ(k) 的 k 权重后保存声明。');d.metadata={...d.metadata,input_kweight:Number($('input-kweight').value),input_kweight_confirmed:d.kind==='chi'&&$('input-kweight-confirmed').checked};setProject(await request(url('/datasets/'+d.id),json('PATCH',{expected_revision:state.project.revision,dataset:d})));if(state.model)markDirty('数据声明已改变，请重新处理并更新模型。');status('数据声明已保存。');});
$('processing-form').onsubmit=guarded(async()=>{
 const parameters={};
 for(const [key] of processingFields){
  const input=$('pp-'+key).value;
  parameters[key]=input===''?null:Number(input);
 }
 parameters.bkg_uncertainty=$('pp-bkg-uncertainty').checked;
 const normalizationOnly=$('normalization-only').checked;
 const noise=$('noise-source').value==='explicit_scalar'
  ? {source:'explicit_scalar',value:Number($('noise-value').value),unit:'chi'}
  : {source:'engine_estimate',unit:'chi',estimation_range:[15,30],assumptions:['white_noise_high_R']};
 status('预处理任务已提交。');
 const job=await request(url('/processing'),json('POST',{
  expected_revision:state.project.revision,dataset_id:state.datasetId,parameters,noise,
  normalization_only:normalizationOnly,calibration_id:$('processing-calibration').value||null
 }));
 await runJob(job);
 await renderProcessing();
 if(state.model)markDirty('预处理已更新，请更新模型绑定后运行。');
});
$('import-path').onclick=()=>$('path-file').click();$('path-file').onchange=guarded(async()=>{const form=new FormData();for(const file of $('path-file').files)form.append('files',file);const query=new URLSearchParams({expected_revision:state.project.revision,element:$('path-element').value.trim(),edge:$('path-edge').value,absorber_site:$('path-site').value.trim()});setProject(await request(url('/paths?'+query),{method:'POST',body:form}));status('路径已导入。');$('path-file').value='';});
$('build-model').onclick=guarded(async()=>buildModel());
$('add-parameter').onclick=guarded(async()=>{readModel();state.model.parameters.nodes.push({id:'p_'+state.model.parameters.nodes.length,label:'',unit:'1',scope:'global',owner_id:null,kind:'fixed',value:1,minimum:null,maximum:null,expression:null,physical_basis:''});renderModel();markDirty();});
$('validate-model').onclick=guarded(async()=>{const result=await request(url('/models/validate'),json('POST',{expected_revision:state.project.revision,model:readModel()}));state.modelValidation=result;$('validation-output').hidden=false;$('validation-output').textContent=(result.fit_allowed===false?'模型结构有效，但拟合条件不满足。':'模型有效，可进入拟合。')+`\n${result.free_ids.length} 个自由变量\n`+(result.diagnostics||[]).map(d=>`[${d.severity}] ${d.message||d.code}`).join('\n');status(result.fit_allowed===false?'请先处理模型中的不可辨识或约束问题。':'模型检查通过。');available();});
$('save-model').onclick=guarded(async()=>{const model=structuredClone(readModel()),checked=state.modelValidation;model.id=uid('model');model.revision=(state.model.revision||0)+1;setProject(await request(url('/models'),json('POST',{expected_revision:state.project.revision,model})));loadModel(model,true);state.modelValidation=checked;$('model-select').value=model.id;available();status('新模型版本已保存。');});
$('fit-model').onclick=guarded(async()=>{if(state.dirty)throw new Error('先保存当前模型草稿。');const modelId=state.model.id;const job=await request(url('/fits'),json('POST',{expected_revision:state.project.revision,model_id:modelId,max_evaluations:Number($('max-evaluations').value),timeout_seconds:300.0}));await runJob(job);const run=state.project.runs.at(-1);if(run)await showRun(run);});
$('model-select').onchange=guarded(async()=>{const model=state.project.models.find(m=>m.id===$('model-select').value);if(model)loadModel(model,true);});
$('run-select').onchange=guarded(async()=>{const run=state.project.runs.find(r=>r.id===$('run-select').value);if(run)await showRun(run);});
$('result-dataset').onchange=guarded(renderFit);$('cancel-job').onclick=guarded(async()=>{if(state.job)await request(url(`/jobs/${state.job.job_id}/cancel`),{method:'POST'});});
$('apply-json').onclick=guarded(async()=>{loadModel(JSON.parse($('model-json').value),false);markDirty();});
$('model-title').addEventListener('change',()=>{if(state.model)markDirty();});
$('apply-windows').onclick=guarded(async()=>{readModel();for(const b of state.model.bindings)for(const key of ['kmin','kmax','rmin','rmax','kweight'])b.transform[key]=Number($('fit-'+key).value);renderModel();markDirty();});
$('open-demo').onclick=guarded(async()=>{state.model=null;state.run=null;setProject(await request('/api/pro/projects/demo',{method:'POST'}));status('合成演示已打开：用于软件集成验证，不代表真实实验精度。');});
$('import-structure').onclick=()=>$('structure-file').click();
$('structure-file').onchange=guarded(async()=>{const file=$('structure-file').files[0];if(!file)return;const form=new FormData();form.append('file',file);const query=new URLSearchParams({expected_revision:state.project.revision,element:$('path-element').value.trim(),edge:$('path-edge').value,absorber_site:$('path-site').value.trim(),source_format:$('structure-format').value});setProject(await request(url('/structures?'+query),{method:'POST',body:form}));status('结构已保存。');$('structure-file').value='';});
$('generate-paths').onclick=guarded(async()=>{if(!$('structure-select').value)throw new Error('先选择结构。');const job=await request(url('/paths/generate'),json('POST',{expected_revision:state.project.revision,structure_id:$('structure-select').value,profile:{nleg:Number($('feff-nleg').value),rmax:Number($('feff-rmax').value),timeout_seconds:75.0}}));await runJob(job);status('专业 FEFF 路径已生成并保存。');});
$('contribution-path').onchange=guarded(renderFit);
$('export-report').onclick=guarded(async()=>{await download(await fetch(url('/reports?run_id='+encodeURIComponent(state.run.id)),{method:'POST'}),'webxas-run-report.zip');status('运行报告已导出。');});
$('compare-models').onclick=guarded(async()=>{const runs=state.project.runs.filter(r=>r.result).slice(-2);const result=await request(url('/compare'),json('POST',{run_ids:runs.map(r=>r.id)}));renderComparison(result);});
const previous=localStorage.getItem('webxas-pro-project');if(previous)$('project-id').value=previous;
const linkedProject=new URLSearchParams(location.search).get('project');
if(linkedProject)guarded(async()=>{setProject(await request('/api/pro/projects/'+encodeURIComponent(linkedProject)));status('已从自动工作台打开同一工程。');})();
available();

function renderCalibrationControls(){
 const selected=$('processing-calibration').value;
 $('processing-calibration').innerHTML='<option value="">不额外应用校准</option>'+(state.project?.calibrations||[]).filter(c=>c.kind==='energy').map(c=>option(c.id,`${c.element} ${c.edge} · ${c.value} eV`,selected)).join('');
 $('calibration-reference').innerHTML=(state.project?.datasets||[]).map(d=>option(d.id,d.filename,state.datasetId)).join('')||'<option value="">先导入参考谱</option>';
 $('save-calibration').disabled=!state.project?.datasets.length||!!state.job;
 $('calibration-list').innerHTML=(state.project?.calibrations||[]).map(c=>`<p class="hint">${esc(c.id)} · ${esc(c.kind)} = ${c.value} ${esc(c.unit)} · ${esc(c.applicability_note)}</p>`).join('');
}
$('calibration-form').onsubmit=guarded(async()=>{
 const d=activeDataset();if(!d?.element||!d.edge)throw new Error('先确认当前数据的元素与吸收边。');
 const kind=$('calibration-kind').value;
 const calibration={id:uid('cal'),kind,value:Number($('calibration-value').value),unit:kind==='energy'?'eV':'1',reference_dataset_ids:[$('calibration-reference').value],element:d.element,edge:d.edge,stderr:$('calibration-stderr').value===''?null:Number($('calibration-stderr').value),uncertainty_basis:kind==='s02'?'conditional_fixed':'not_estimated',reference_model_id:state.project.models.some(m=>m.id===state.model?.id)?state.model.id:null,conditions:d.conditions,applicability_note:$('calibration-note').value};
 setProject(await request(url('/calibrations'),json('POST',{expected_revision:state.project.revision,calibration})));status('标定记录已保存；请检查其适用数据与模型条件。');
});

function renderSensitivity(){
 const records=state.project?.metadata?.sensitivity_scans||[];
 const scan=Array.isArray(records)?records.at(-1):records;
 const cases=scan?.cases||[];
 $('sensitivity-section').hidden=!cases.length;
 $('sensitivity-table').innerHTML=`<table><thead><tr><th>检查</th><th>状态</th><th>运行快照</th><th>说明</th></tr></thead><tbody>${cases.map(c=>`<tr data-sensitivity-case="${esc(c.case_id||c.name)}"><td>${esc(c.case_id||c.name)}</td><td>${esc(c.status)}</td><td>${c.run_id?`<button data-sensitivity-run="${esc(c.run_id)}" class="secondary compact">查看该运行</button>`:'—'}</td><td>${esc(c.reason||'')}</td></tr>`).join('')}</tbody></table>`;
 $('sensitivity-table').querySelectorAll('[data-sensitivity-run]').forEach(button=>button.onclick=guarded(async()=>{const run=state.project.runs.find(r=>r.id===button.dataset.sensitivityRun);if(run)await showRun(run);}));
}
$('sensitivity-model').onclick=guarded(async()=>{
 if(state.dirty)throw new Error('先保存当前模型草稿。');
 const job=await request(url('/sensitivity'),json('POST',{expected_revision:state.project.revision,model_id:state.model.id,max_evaluations:Number($('max-evaluations').value),timeout_seconds:900.0}));
 await runJob(job);renderSensitivity();$('sensitivity-section').open=true;
 status('敏感性检查已完成，所有条件与状态保存在工程中。');
});

function renderComparison(result){
 const number=value=>value==null?'不可用':Number(value).toPrecision(4);
 const assessment={conditional:'条件结果',needs_review:'需复核',not_identifiable:'不可辨识',validated:'已验收'};
 const rows=(result.runs||[]).map(r=>`<tr><td>${esc(r.run_id)}<small>${esc(r.model_id)}</small></td><td>${esc(assessment[r.assessment]||r.assessment||'无结果')}</td><td>${number(r.statistics?.rfactor)}</td><td>${number(r.statistics?.reduced_chisquare)}</td><td>${r.statistics?.n_variables??'不可用'}</td></tr>`).join('');
 $('comparison-output').hidden=false;
 $('comparison-output').innerHTML=`<h3>运行比较</h3><p class="notice">${result.comparability?'分析条件一致，可以审查各项指标。':'分析条件不可直接比较。'} ${esc(result.conclusion||'')} 本次比较不确定唯一结构。</p>${(result.reasons||[]).length?`<ul>${result.reasons.map(r=>`<li>${esc(r.message||r.reason||r)}</li>`).join('')}</ul>`:''}<div class="table-scroll"><table><thead><tr><th>运行与模型</th><th>评估</th><th>R-factor</th><th>约化 χ²</th><th>自由变量</th></tr></thead><tbody>${rows}</tbody></table></div><details><summary>原始记录</summary><pre>${esc(JSON.stringify(result,null,2))}</pre></details>`;
}

function setProStage(id) {
 const valid=['data-section','model-section','result-section'];
 const target=valid.includes(id)?id:valid[0];
 document.body.dataset.proStage=target;
 for(const link of document.querySelectorAll('.steps a')) {
   if(link.hash==='#'+target)link.setAttribute('aria-current','step');
   else link.removeAttribute('aria-current');
 }
}
window.addEventListener('hashchange',()=>setProStage(location.hash.slice(1)));
setProStage(location.hash.slice(1));
document.querySelectorAll('.steps a').forEach(a=>a.addEventListener('click',()=>setProStage(a.hash.slice(1))));
