/* Pure grouping and display-comparison helpers; no scientific processing. */
(function(root) {
  'use strict';
  const median = values => { const a = [...values].sort((a,b)=>a-b), n=a.length; return n % 2 ? a[(n-1)/2] : (a[n/2-1]+a[n/2])/2; };
  function suggest(d, index=0) {
    const header = d.simple_input.header || [], fields = {};
    for (const line of header) {
      const match = line.match(/^\s*#?\s*(Element\.symbol|Scan\.element|Element\.edge|Scan\.edge|Sample\.name|Sample|Scan\.type)\s*:\s*(.+?)\s*$/i);
      if (match) { const key = match[1].toLowerCase().replace('scan.element','element.symbol').replace('scan.edge','element.edge'); (fields[key] ||= []).push(match[2]); }
    }
    const unique = key => [...new Set((fields[key] || []).map(v=>v.trim()))];
    const conflict = unique('element.symbol').length > 1 || unique('element.edge').length > 1;
    const sample = unique('sample.name')[0] || unique('sample')[0] || '';
    const name = sample || d.label || d.filename;
    const basename = name.split('/').at(-1).trim().replace(/\.(xdi|dat|txt|csv|xmu|chi|prj|gz)$/i,'');
    const reference = /(?:^|[\s_.-])(ref(?:erence)?|foil|standard)(?:[\s_.-]|\d|$)|参照|参考|标样/i.test(name + ' ' + d.filename + ' ' + (fields['scan.type'] || []).join(' '));
    const stem = basename.replace(/(?:[\s_.-]+(?:scan|repeat|rep|run)?[\s_.-]*\d+|(?:scan|repeat|rep|run)[\s_.-]*\d+)+$/i,'').replace(/(?:[\s_.-])(?:ref(?:erence)?|foil|standard)$/i,'').trim() || basename;
    let element = unique('element.symbol')[0] || '';
    let source = element ? '文件头元素声明' : '';
    if (!element) {
      if (/(?:^|[\s_.-])(?:Fe(?=[A-Z\d\s_.-]|$)|iron\b)|铁/.test(name)) element='Fe';
      else if (/(?:^|[\s_.-])(?:Zn(?=[A-Z\d\s_.-]|$)|zinc\b)|氧化锌|锌/.test(name)) element='Zn';
      else { const token = name.match(/(?:^|[\s_.-])(Cu|Ni|Co|Mn|Cr|Ti|V|Mo|Pd|Pt|Au|Ag|Ce|U|Zr|Ru|Rh|Sn|W)(?=[A-Z\d\s_.-]|$)/); if(token) element=token[1]; }
      if(element) source='名称线索（待核对）';
    }
    const edge = (unique('element.edge')[0] || '').toUpperCase();
    if (conflict) element='';
    const domain = d.selection?.kind || d.kind;
    const role = reference ? '参照组' : '样品组';
    const group = conflict ? `元数据冲突 · ${index+1}` : `${element || '元素待确认'}${edge ? ' '+edge : ''} · ${stem} · ${role}`;
    return {group, element, edge, reference, domain, source:conflict ? '元素或吸收边声明冲突，需人工核对' : source || '名称前缀分组（元素未知）', conflict};
  }
  function interpolate(x,y,grid) {
    let j=0;
    return grid.map(v=>{ while(j<x.length-2 && x[j+1]<v) j++; const f=(v-x[j])/(x[j+1]-x[j]); return y[j]*(1-f)+y[j+1]*f; });
  }
  function compare(curves, referenceId=null) {
    if (curves.length<2) throw new Error('至少勾选两条兼容光谱后才能比较差异。');
    for (const c of curves) if(c.x.length<2 || c.x.length!==c.y.length || c.x.some((v,i)=>!Number.isFinite(v)||(i>0&&v<=c.x[i-1])) || c.y.some(v=>!Number.isFinite(v))) throw new Error('光谱含无效值或横坐标未严格递增，不能比较。');
    const lo=Math.max(...curves.map(c=>c.x[0])), hi=Math.min(...curves.map(c=>c.x.at(-1)));
    if(!(hi>lo)) throw new Error('所选光谱没有共同坐标范围，不能计算差异。');
    if(curves.some(c=>c.x.filter(v=>v>=lo&&v<=hi).length<20)) throw new Error('共同范围内至少需要每条光谱各 20 个实测点。');
    const grid=Array.from({length:301},(_,i)=>lo+(hi-lo)*i/300);
    const sampled=curves.map(c=>({...c,x:grid,y:interpolate(c.x,c.y,grid)}));
    const ref=sampled.find(c=>c.id===referenceId);
    const baseline=ref ? ref.y : grid.map((_,i)=>median(sampled.map(c=>c.y[i])));
    const scale=Math.max(...baseline)-Math.min(...baseline);
    if(scale<1e-10) throw new Error('对比基线接近平坦，无法可靠计算相对差异。');
    const pair=!ref && curves.length===2;
    const rows=sampled.map((c,i)=>({...c, score:Math.sqrt(c.y.reduce((sum,v,j)=>sum+(v-(pair?sampled[1-i].y[j]:baseline[j]))**2,0)/grid.length)/scale, isReference:c.id===ref?.id}));
    return {rows, baseline, grid, lo, hi, pair, baselineLabel:ref ? `参照：${ref.label}` : pair ? '两谱相互比较（不判定哪条异常）' : '所选谱的逐点中位曲线', coverage:curves.map(c=>({id:c.id,fraction:(hi-lo)/(c.x.at(-1)-c.x[0])}))};
  }
  const api={suggest,compare};
  if(typeof module!=='undefined' && module.exports) module.exports=api;
  else root.SeriesCore=api;
})(typeof globalThis==='undefined'?this:globalThis);
