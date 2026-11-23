'use strict';

function updateWorkbenchView() {
  const hasData = Boolean(S.project?.datasets.length);
  const canRead = Boolean(S.delivery || S.workflow?.state === 'partial');
  const stage = S.job ? 'running' : !hasData ? 'empty'
    : canRead && (S.view === 'read' || (!S.view && !S.dirty)) ? 'read' : 'config';
  document.body.dataset.stage = stage;
  const intake = $('intake');
  const host = stage === 'empty' ? $('entry-slot') : $('configuration');
  if (intake.parentElement !== host) host.prepend(intake);
  $('configure').hidden = stage !== 'read';
  $('figure-index').hidden = stage !== 'read';
}

function changeWorkbenchView(view) {
  S.view = view;
  updateWorkbenchView();
  const target = view === 'read' ? $('result-title') : $('upload');
  target.tabIndex = target.tagName === 'H2' ? -1 : 0;
  target.focus({preventScroll:true});
  window.scrollTo({top:0,behavior:'instant'});
}

function updateFigureIndex(figures,sampleCount) {
  const jump = $('figure-jump');
  const rail = $('figure-links');
  rail.replaceChildren();
  jump.replaceChildren(new Option('选择图件',''));
  for (const f of figures) {
    const id = 'result-'+f.id;
    const sample = sampleCount > 1 ? (f.display_sample_names || f.sample_names || []).join('、') : '';
    const label = [moduleTitle(f),sample].filter(Boolean).join(' · ');
    const link = document.createElement('a');
    link.href = '#'+encodeURIComponent(id);
    link.textContent = label;
    rail.append(link);
    jump.add(new Option(label,id));
  }
  $('sample-filter-label').hidden = sampleCount <= 1;
  $('jump-label').hidden = figures.length < 2;
  jump.onchange = () => {
    const card = document.getElementById(jump.value);
    if (card) {card.tabIndex=-1;card.focus();card.scrollIntoView({block:'start'});}
    jump.value='';
  };
}

function renderWorkflowIssues(workflow) {
  const missing = S.delivery?.missing || [];
  const unfinished = ['partial','needs_input'].includes(workflow.state);
  const host = $('issues');
  host.replaceChildren();
  if (!unfinished && !missing.length) return;
  const notice = document.createElement('details');
  notice.className = 'delivery-notice';
  const heading = document.createElement('summary');
  heading.textContent = missing.length ? missing.length+' 项图件未生成' : '部分分析未完成';
  notice.append(heading);
  const items = missing.map(m=>[FAMILY_NAMES[m.family] || m.family,m.reason].filter(Boolean).join(' · '));
  if (unfinished) items.push(...(workflow.issues || []).map(issue=>issue.message).filter(Boolean));
  const list = document.createElement('ul');
  for (const message of new Set(items)) {const row=document.createElement('li');row.textContent=message;list.append(row);}
  if (list.childElementCount) notice.append(list);
  const button = document.createElement('button');
  button.type='button';button.className='text-button';button.textContent='检查分析条件';
  button.onclick=()=>changeWorkbenchView('config');
  notice.append(button);host.append(notice);
}

document.addEventListener('DOMContentLoaded', updateWorkbenchView);
