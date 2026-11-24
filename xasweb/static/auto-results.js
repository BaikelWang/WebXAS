'use strict';
// Every figure, interpretation and download belongs to the same delivery snapshot.
function fileURL(name) {return auto(`/deliveries/${encodeURIComponent(S.delivery.id)}/files?filename=${encodeURIComponent(name)}`);}
function metricValue(m) {
  let value=typeof m.value==='number'?Number(m.value.toPrecision(7)).toString():typeof m.value==='object'?JSON.stringify(m.value):String(m.value??'');
  if(m.uncertainty!=null)value+=' ± '+m.uncertainty;
  if(m.unit==='not_declared')return value+'（单位未记录）';
  return value+(m.unit&&m.unit!=='1'?' '+m.unit:'');
}
function moduleTitle(f) {
  const method=f.insight?.method?.figure_method||{};
  if(method.quantity==='normalized_difference')return '归一化近边差谱';
  if(method.quantity==='reference_coefficients')return '参考组分系数';
  if(method.quantity==='objective_residual')return '拟合目标残差';
  if(method.quantity==='preedge_integral')return method.panel_title;
  if(f.family==='F13')return f.title;
  if(f.family==='F07')return `${String(method.space).toLowerCase()==='r'?'R':'k'} 空间${method.source_panel_index>0?'残差':'拟合'}`;
  if(f.family==='F08'&&method.projection)return '结构示意 · '+method.projection;
  return FAMILY_NAMES[f.family]||f.title;
}
function figureCard(f,index) {
  const i=f.insight||{observations:['此图来自已保存的历史交付。完整方法与数值见原始结果包。'],metrics:[],limitations:['历史记录未保存逐图解读，未根据当前项目重新推断。']};
  const sample=(f.display_sample_names||f.sample_names||[]).join('、')||'运行结果';
  const formats=f.formats||f.files||[];
  const imageFiles=Array.isArray(formats)?formats:Object.values(formats);
  const links=imageFiles.filter(n=>typeof n==='string').map(n=>`<a href="${fileURL(n)}" download>${esc(n.split('.').at(-1).toUpperCase())}</a>`).join('');
  const data=(f.data_files||[]).map(n=>`<a href="${fileURL(n)}" download>${esc(n.split('/').at(-1))}</a>`).join('');
  const display=i.display||{summary:'历史结果未保存逐图解读。',metrics:[],notices:[]};
  const metrics=(display.metrics||[]).filter(m=>m.value!=null);
  const methodSection=({F01:'absorption',F02:'normalization',F03:'edge',F04:'fourier',F05:'fourier',F06:'wavelet',F07:'structure',F08:'structure',F09:'lcf',F10:'series',F11:'structure',F13:'decomposition'})[f.family]||'normalization';
  const fullMetrics=(i.metrics||[]).filter(m=>m.value!=null);
  const basis=i.method_basis;
  const refs=(basis?.references||[]).filter(ref=>ref.url?.startsWith('https://')).map(ref=>`<a href="${esc(ref.url)}" target="_blank" rel="noopener noreferrer">${esc(ref.label)}</a>`).join(' · ');
  return `<article class="figure-card" id="result-${esc(f.id)}" data-figure="${esc(f.id)}">
    <div class="figure-heading"><h3 title="${esc(f.title)}">${esc(moduleTitle(f))}</h3><span class="sample-badge" title="${esc(sample)}">${esc(sample)}</span></div>
    <button class="figure-open" data-zoom="${esc(f.id)}" aria-label="放大查看 ${esc(f.title)}"><img loading="lazy" decoding="async" src="${previewURL(f)}" alt="${esc(f.title)}"><span>放大查看 ↗</span></button>
    <div class="figure-analysis">
    ${f.family==='F09'?'<p class="quantitation-note">组分定量仍有一份标样超出验证范围。<a href="/methods#lcf">查看验证</a></p>':''}${display.summary?`<p class="result-text">${esc(display.summary)}</p>`:''}
    ${metrics.length?`<div class="metrics">${metrics.slice(0,2).map(m=>`<div><span>${esc(m.label)}</span><strong>${esc(metricValue(m))}</strong></div>`).join('')}</div>`:''}
    ${(display.notices||[]).length?`<div class="figure-notices">${display.notices.map(t=>`<p>${esc(t)}</p>`).join('')}</div>`:''}</div>
    <div class="figure-footer"><div class="download-actions"><details class="download-menu"><summary>下载图片</summary><div>${links||'<span>请重新生成图件</span>'}</div></details><details class="download-menu"><summary>下载数据</summary><div>${data||'<span>此图没有独立数据文件</span>'}</div></details></div>
    <details class="figure-source"><summary>本图参数</summary><a href="/methods#${methodSection}" target="_blank" rel="noopener">算法与验证 ↗</a>${basis?`<p>${esc(basis.summary)}</p>`:''}${refs?`<p>${refs}</p>`:''}${fullMetrics.length?`<dl class="metric-definitions">${fullMetrics.map(m=>`<dt>${esc(m.label)} · ${esc(metricValue(m))}</dt><dd>${esc(m.basis||'')}</dd>`).join('')}</dl>`:''}${(i.observations||[]).map(t=>`<p>${esc(t)}</p>`).join('')}${(i.limitations||[]).map(t=>`<p>${esc(t)}</p>`).join('')}<details><summary>来源信息</summary><pre>${esc(JSON.stringify({workflow_id:S.delivery.workflow_id,figure_id:f.id,method:i.method,sources:i.sources||f.sources},null,2))}</pre></details></details></div>
  </article>`;
}
function renderFigureCards() {
  const all=(S.delivery?.figures||[]).filter(f=>f.filename?.endsWith('.png'));
  S.figures=all;
  const nameIds=new Map();for(const f of all)(f.sample_names||[]).forEach((name,i)=>{if(!nameIds.has(name))nameIds.set(name,new Set());nameIds.get(name).add(f.dataset_ids?.[i]);});
  for(const f of all)f.display_sample_names=(f.sample_names||[]).map((name,i)=>nameIds.get(name).size>1?`${name} · ${f.dataset_ids[i].slice(-6)}`:name);

  const filter=$('sample-filter').value;
  const ids=[...new Set(all.flatMap(f=>f.dataset_ids||[]))];
  $('sample-filter').innerHTML='<option value="">全部样品</option>'+ids.map(id=>{
    const f=all.find(f=>(f.dataset_ids||[]).includes(id));
    const pos=f.dataset_ids.indexOf(id);
    return `<option value="${esc(id)}">${esc(f.display_sample_names?.[pos]||id)}</option>`;
  }).join('');
  if(ids.includes(filter))$('sample-filter').value=filter;
  const selected=all.filter(f=>!$('sample-filter').value||(f.dataset_ids||[]).includes($('sample-filter').value));
  const order=f=>f.family==='F10'?-1:Number(f.family.slice(1));
  selected.sort((a,b)=>(a.family==='F10'?-1:0)-(b.family==='F10'?-1:0)||(a.dataset_ids||[]).join().localeCompare((b.dataset_ids||[]).join())||order(a)-order(b));
  updateFigureIndex(selected,ids.length);
  $('figure-count').textContent=`${selected.length} 张图`;
  $('figure-list').innerHTML=selected.map((f,index)=>{const key=(f.dataset_ids||[]).join();const previous=selected[index-1];const heading=ids.length>1&&(index===0||key!==(previous?.dataset_ids||[]).join())?`<h3 class="sample-group">${esc((f.dataset_ids||[]).length>1?'跨样品分析':(f.display_sample_names||[]).join('、')||'运行结果')}</h3>`:'';return heading+figureCard(f,index);}).join('')||'<div class="empty-results"><h3>暂无图件</h3><p>选择图件后点击生成。</p></div>';
  document.querySelectorAll('[data-zoom]').forEach(el=>el.onclick=()=>{
    const f=S.figures.find(f=>f.id===el.dataset.zoom);
    $('large-image').src=previewURL(f);$('large-title').textContent=f.title;openDialog('image-dialog');
  });
  // Native details menus are keyboard accessible; close siblings on opening.
  document.querySelectorAll('.download-menu').forEach(d=>d.addEventListener('toggle',()=>{
    if(d.open)document.querySelectorAll('.download-menu[open]').forEach(other=>{if(other!==d)other.open=false;});
  }));
}
