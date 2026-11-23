'use strict';
// Product choices are separate from numerical processing settings.
const MODES = [
  ['process','光谱处理','自动处理 · 质量检查','01'],
  ['xanes','XANES 分析','近边特征 · 参考组分','02'],
  ['exafs','EXAFS 结构拟合','已知模型 · 局域结构','03'],
  ['series','多谱与联合分析','状态变化 · 联合拟合','04'],
  ['plot','科研绘图','已有结果 · 科研出图','05']
];
const FAMILY_NAMES = {F01:'输入与背景',F02:'归一化 XANES',F03:'一阶导数',F04:'k 空间',F05:'R 空间',F06:'小波变换',F07:'拟合与残差',F08:'路径与结构',F09:'参考组分',F10:'状态趋势与热图',F11:'拟合诊断',F13:'PCA / MCR'};
const MODE_FAMILIES = {
  process:['F01','F02','F03','F04','F05','F06'],
  xanes:['F01','F02','F03','F09','F13'],
  exafs:['F02','F04','F05','F06','F07','F08','F11'],
  series:['F02','F03','F04','F05','F06','F07','F08','F09','F10','F11','F13'],
  plot:Object.keys(FAMILY_NAMES)
};
function operationSelection() {
  if ($('task').value === 'exafs') return ['structural_fit'];
  return [...document.querySelectorAll('[data-operation]:checked')].map(el=>el.dataset.operation);
}
function selectedFamilies() {
  return [...document.querySelectorAll('[data-family]:checked')].map(el=>el.dataset.family);
}
function loadWorkflowControls(w) {
  const i=w.intent;
  $('decomp-count').value=i.decomposition_components||2;$('decomp-window').value=(i.decomposition_window_eV||[]).join(' ');
  for (const [key,id] of Object.entries({left_eV:'pe-left',peak_eV:'pe-peak',right_eV:'pe-right'}))$(id).value=(i.preedge_windows?.[key]||[]).join(' ');
  $('lcf-ordinate').value=i.lcf?.normalization_ordinate||'norm';
  S.plan=null;S.checkToken++;
  ['references-confirmed','joint-confirmed','independent-repeats','calibration-confirmed','merge-repeats'].forEach(id=>$(id).checked=false);
  ['reference-note','joint-note','repeats-note','calibration-reference','calibration-energy','calibration-note'].forEach(id=>$(id).value='');
  $('state-key').value='temperature_K';$('series-relation').value='independent';
  document.querySelectorAll('[data-cal-target]').forEach(el=>el.checked=false);
  $('task').value=i.task;
  S.selected=new Set(i.dataset_ids);S.references=new Set(i.reference_ids||[]);
  S.operations=i.requested_operations??[...(i.reference_ids?.length?['lcf']:[]),...(['exafs','series'].includes(i.task)&&(i.model_id||i.model_recipe_id)?['structural_fit']:[])];
  if(i.task==='exafs')S.operations=['structural_fit'];
  $('model').value=i.model_id||'';
  S.overrides={};
  for(const key of ['model_recipe_id','calibration_ids','fit','lcf','processing_overrides','use_existing_processing']) {
    if(i[key]!=null)S.overrides[key]=structuredClone(i[key]);
  }
  syncAnswers({conditions:i.conditions||{},series:i.series||{}});
  const cal=i.conditions?.reference_calibration;
  S.standards=new Set(cal?.reference_dataset_id?[cal.reference_dataset_id]:[]);
  renderData();
  if(cal?.apply_to_dataset_ids)document.querySelectorAll('[data-cal-target]').forEach(el=>el.checked=cal.apply_to_dataset_ids.includes(el.dataset.calTarget));
  S.familyChoices=S.delivery?.figure_families||null;
  renderTask();renderStateTable();
}
function renderModes() {
  $('mode-cards').innerHTML = MODES.map(([id,name,description,number])=>`<button class="mode-card" data-mode="${id}" role="radio" aria-checked="${$('task').value===id}"><span class="mode-number">${number}</span><strong>${name}</strong><small>${description}</small></button>`).join('');
  const radios=[...document.querySelectorAll('[data-mode]')];
  radios.forEach((el,index)=>{
    el.tabIndex=el.getAttribute('aria-checked')==='true'?0:-1;
    el.onkeydown=async event=>{
      let at=index;
      if(['ArrowRight','ArrowDown'].includes(event.key))at=(index+1)%radios.length;
      else if(['ArrowLeft','ArrowUp'].includes(event.key))at=(index+radios.length-1)%radios.length;
      else if(event.key==='Home')at=0;
      else if(event.key==='End')at=radios.length-1;
      else return;
      event.preventDefault();
      const id=radios[at].dataset.mode;
      await radios[at].onclick();
      document.querySelector('[data-mode="'+id+'"]')?.focus();
    };
  });
  radios.forEach(el=>el.onclick=action(async()=>{
    if ($('task').value===el.dataset.mode) return;
    $('task').value=el.dataset.mode;
    S.operations = el.dataset.mode==='exafs' ? ['structural_fit'] : [];
    S.familyChoices=null;
    invalidate();renderData();renderTask();await preflight();
  }));
}
function renderFeatures() {
  const task=$('task').value, selections=S.operations||[];
  $('decomp-settings').hidden=!selections.includes('decomposition');
  $('preedge-panel').hidden=!selections.includes('preedge');
  const caps=S.plan?.capabilities||[
    {id:'preedge',label:'预边积分',available:['xanes','series'].includes(task)},
    {id:'lcf',label:'参考组分拟合',available:['xanes','series'].includes(task)},
    {id:'decomposition',label:'PCA / MCR',available:['xanes','series'].includes(task)},
    {id:'structural_fit',label:'结构拟合',available:['exafs','series'].includes(task)}
  ];
  $('analysis-features').innerHTML=`<div class="automatic-step"><span>✓</span><div><strong>${task==='plot'?'使用已有结果':'自动预处理'}</strong></div></div>`+caps.filter(c=>c.available).map(c=>{
    const required=task==='exafs'&&c.id==='structural_fit', selected=required||selections.includes(c.id);
    const missing=c.id==='lcf'&&!S.references.size?'需上传并指定参考谱':c.id==='structural_fit'&&!$('model').value?'需选择已审查模型':'';
    return `<label class="feature ${!c.available?'unavailable':''}"><input type="checkbox" data-operation="${c.id}" ${selected?'checked':''} ${!c.available||required?'data-unavailable="true" disabled':''}><span><strong>${esc(c.label)}</strong><small>${esc(required?(missing||'必需'):missing)}</small></span></label>`;
  }).join('');
  document.querySelectorAll('[data-operation]').forEach(el=>el.onchange=action(async()=>{
    S.operations=operationSelection();invalidate();renderTask();await preflight();
  }));
  const families=MODE_FAMILIES[task].filter(id=>id!=='F13'||selections.includes('decomposition')||task==='plot');
  const checked=S.familyChoices||families;
  $('figure-features').innerHTML=families.map(id=>`<label class="check"><input type="checkbox" data-family="${id}" ${checked.includes(id)?'checked':''}>${FAMILY_NAMES[id]}</label>`).join('');
  document.querySelectorAll('[data-family]').forEach(el=>el.onchange=()=>{
    if(S.busy)return;S.familyChoices=selectedFamilies();$('output-change').hidden=!S.delivery;
  });
}
function refreshChoices() {
 renderModes();renderFeatures();
 const bindings=Object.entries(S.overrides.calibration_ids||{});
 $('legacy-calibrations').innerHTML=bindings.length?`<p class="hint">沿用历史能量校准 · ${bindings.length} 条谱</p><button id="clear-calibrations" class="text-button">清除历史校准绑定</button>`:'';
 if($('clear-calibrations'))$('clear-calibrations').onclick=action(async()=>{S.overrides.calibration_ids={};invalidate();renderTask();await preflight();});
 busy(S.busy);
}
