(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = {q:'',topic:'',kind:'',view:'',page:1,entry:'',original:'',originalPart:null,metadata:null,request:null,readerRequest:null,originalRequest:null,current:null,timer:null,scrollTimer:null,restoring:false};
  const browsePositions = new Map();
  history.scrollRestoration = 'manual';
  const STORAGE_KEY = 'webxas.knowledge.reading.v1';
  const defaultReading = () => ({fontSize:18,records:{}});
  let reading = defaultReading();
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
    if (stored && typeof stored.records === 'object' && stored.records !== null && !Array.isArray(stored.records)) {
      reading = {fontSize:Math.max(16, Math.min(24, Number(stored.fontSize) || 18)),records:{}};
      for (const [id, record] of Object.entries(stored.records)) {
        if (record && typeof record === 'object' && typeof record.title === 'string' && !['__proto__','constructor','prototype'].includes(id)) reading.records[id] = record;
      }
    }
  } catch (_) { /* Browsing remains available when local storage is unavailable. */ }
  const text = (tag, value, className) => {
    const node = document.createElement(tag);
    if (value !== null && value !== undefined) node.textContent = String(value);
    if (className) node.className = className;
    return node;
  };
  const safeUrl = value => {
    if (typeof value !== 'string' || !value.trim()) return null;
    try { const url = new URL(value); return ['https:','http:'].includes(url.protocol) ? url.href : null; } catch (_) { return null; }
  };
  const safeImage = value => {
    if (typeof value !== 'string' || !value.trim()) return null;
    try {
      const url = new URL(value, location.origin);
      return url.origin === location.origin && /^\/api\/knowledge\/assets\/[a-zA-Z0-9._-]+$/.test(url.pathname) ? url.href : null;
    } catch (_) { return null; }
  };
  function externalLink(title, url, className) {
    const safe = safeUrl(url), node = text(safe ? 'a' : 'span', title, className);
    if (safe) { node.href = safe; node.target = '_blank'; node.rel = 'noopener noreferrer'; }
    return node;
  }
  function persist() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(reading)); return true; }
    catch (_) { $('announcement').textContent = '当前浏览器无法保存阅读记录；正文仍可正常阅读。'; return false; }
  }
  async function api(path, signal) {
    const response = await fetch(path, {signal,cache:'no-store'});
    if (!response.ok) throw new Error(response.status === 404 ? '没有找到这份资料，请返回目录重新选择。' : '请稍后重试，或确认本地软件服务正在运行。');
    return response.json();
  }
  const topicTitle = id => state.metadata?.topics?.find(t => t.id === id)?.title || '';
  const kindTitle = id => state.metadata?.kinds?.find(k => k.id === id)?.title || '';
  const authorText = item => Array.isArray(item.authors) ? item.authors.join('、') : item.authors;
  const originalKinds = [{id:'pdf',title:'教材 PDF'}, {id:'article',title:'论文'}, {id:'website',title:'网页'}, {id:'video',title:'视频'}];
  const videoNoteId = /^video-note-(BV[A-Za-z0-9]{10})-P(0[1-9]|[1-9][0-9])$/;
  const collectionTitles = {originals:'原始资料',library:'资料索引',atlas:'图谱与案例',bookmarks:'我的书签'};
  const volumeMarks = {basics:'μ(E)',measurement:'I₀ / Iₜ',processing:'χ(k) → χ(R)',xanes:'E₀',exafs:'k²χ(k)',wavelet:'W(k, R)',reporting:'Δχ(R)',operando:'μ(E, t)'};
  const readingLabel = item => ({sample:'中文样章',chapter:'专题章节',guide:'主题导读',planned:'编写中'}[item.reading_status] || item.reading_status || '');
  const isHome = () => !state.view && !state.q && !state.topic && !state.kind && state.page === 1;
  function locationParams(includeEntry = true) {
    const p = new URLSearchParams();
    for (const key of ['view','q','topic','kind']) if (state[key]) p.set(key, state[key]);
    if (state.page > 1) p.set('page', String(state.page));
    if (includeEntry && state.entry) p.set('entry', state.entry);
    if (includeEntry && state.original) {
      p.set('original', state.original);
      if (state.originalPart) p.set('part', String(state.originalPart));
    }
    return p;
  }
  function locationUrl(includeEntry = true) { const p = locationParams(includeEntry); return '/knowledge' + (p.size ? '?' + p.toString() : ''); }
  function writeLocation(push = false) {
    const url = locationUrl();
    if (url !== location.pathname + location.search) history[push ? 'pushState' : 'replaceState']({}, '', url);
  }
  function readLocation() {
    const p = new URLSearchParams(location.search);
    state.q = (p.get('q') || '').slice(0,200); state.topic = p.get('topic') || ''; state.kind = p.get('kind') || '';
    state.view = ['originals','library','atlas','bookmarks'].includes(p.get('view')) ? p.get('view') : '';
    if (!availableKinds().some(kind => kind.id === state.kind) && (state.metadata || state.view === 'originals')) state.kind = '';
    state.page = Math.max(1,Number.parseInt(p.get('page'),10) || 1); state.entry = p.get('entry') || '';
    state.original = p.get('original') || '';
    const part = Number(p.get('part'));
    state.originalPart = state.original && Number.isSafeInteger(part) && part > 0 ? part : null;
    $('search').value = state.q;
  }
  function entryLink(item, className, label) {
    const a = text('a', label ?? item.title, className), p = locationParams(false); p.set('entry',item.id);
    a.href = '/knowledge?' + p.toString();
    a.dataset.entryId = item.id;
    a.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button !== 0) return;
      event.preventDefault(); openEntry(item.id, true, a);
    });
    return a;
  }
  function videoNoteLink(part, className, label) {
    if (!videoNoteId.test(part?.note_entry_id || '')) return null;
    return entryLink({id:part.note_entry_id}, className, label);
  }
  function originalLink(item, label = item.title, entryId = state.entry) {
    const a = text('a', label, 'original-link');
    a.dataset.originalId = item.id;
    const p = locationParams();
    const note = videoNoteId.exec(entryId);
    const part = note && item.id === 'bilibili-' + note[1].toLowerCase() ? Number(note[2]) : null;
    p.set('original', item.id);
    if (part) p.set('part', String(part)); else p.delete('part');
    a.href = '/knowledge?' + p.toString();
    a.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button !== 0) return;
      event.preventDefault();
      if (!state.entry) browsePositions.set(locationUrl(false), {top:window.scrollY,originalId:item.id});
      openOriginal(item.id, true, part);
    });
    return a;
  }
  function filterButton(label, attr, value, onClick, className) {
    const b = text('button',label,className); b.type = 'button'; b.dataset[attr] = value; b.addEventListener('click',onClick); return b;
  }
  function renderNavigation() {
    const options = state.metadata.topics.map(topic => new Option(topic.title, topic.id));
    $('topic-select').replaceChildren(new Option('全部主题',''), ...options);
    const indexTopics = [{title:'全部主题',id:''}, ...state.metadata.topics];
    $('index-topic-buttons').replaceChildren(...indexTopics.map(topic =>
      filterButton(topic.title,'topic',topic.id,() => { state.topic=topic.id; filtersChanged(); },'index-topic')));
    updateBookmarkCount();
  }
  function availableKinds() {
    if (state.view === 'originals') return originalKinds;
    let kinds = state.metadata?.kinds || [];
    if (state.view === 'library') kinds = kinds.filter(k => ['literature','blog','manual','video','video_note'].includes(k.id));
    if (state.view === 'atlas') kinds = kinds.filter(k => ['case','figure'].includes(k.id));
    return kinds;
  }
  function renderKinds() {
    if ($('kinds').dataset.view === state.view) return;
    const kinds = availableKinds();
    const buttons = [filterButton('全部','kind','',() => { state.kind=''; filtersChanged(); },'kind-button')];
    for (const kind of kinds) buttons.push(filterButton(kind.title,'kind',kind.id,() => { state.kind=kind.id; filtersChanged(); },'kind-button'));
    $('kinds').replaceChildren(...buttons);
    $('kinds').dataset.view = state.view;
  }
  function updateBookmarkCount() {
    const count = Object.values(reading.records).filter(record => record.bookmarked).length;
    $('bookmark-count').textContent = count ? '· ' + count : '';
    $('bookmark-count').hidden = !count;
  }
  function updateFilterUI() {
    renderKinds();
    for (const node of document.querySelectorAll('#primary-navigation [data-view]')) {
      if (node.dataset.view === state.view) node.setAttribute('aria-current','page'); else node.removeAttribute('aria-current');
    }
    for (const b of $('kinds').querySelectorAll('button')) b.setAttribute('aria-pressed',String(b.dataset.kind === state.kind));
    for (const b of $('index-topic-buttons').querySelectorAll('button')) b.setAttribute('aria-pressed',String(b.dataset.topic === state.topic));
    $('topic-select').value = state.topic; $('clear-search').hidden = !$('search').value;
    const originalShelf = state.view === 'originals';
    const library = state.view === 'library';
    $('browse-page').dataset.view = state.view;
    $('filter-row').hidden = false;
    $('index-topics').hidden = !library;
    $('topic-select').hidden = originalShelf || library;
    $('reset-filters').hidden = !(state.q || state.topic || state.kind);
    const placeholders = {originals:'搜索原文标题、作者或 DOI',library:'搜索教材、论文、手册或视频笔记',atlas:'搜索图谱或案例',bookmarks:'搜索我的书签'};
    $('search').placeholder = placeholders[state.view] || '搜索章节、术语或资料';
    $('results-title').textContent = collectionTitles[state.view] || (state.q ? '全馆搜索' : topicTitle(state.topic) || kindTitle(state.kind) || '全部知识');
    const home = isHome();
    $('search-area').classList.toggle('home-search', home);
    $('home-title').hidden = !home; $('home-view').hidden = !home; $('results').hidden = home;
  }
  function cardStatus(item) {
    if (state.view === 'bookmarks') return Math.round(Number(reading.records[item.id]?.percent) || 0) + '% 已读';
    return item.read_status_label || readingLabel(item);
  }
  function card(item) {
    const article = text('article',null,'entry-card');
    const top = text('div',null,'card-top');
    top.append(text('span',item.subtype || item.kind_label || kindTitle(item.kind)));
    const topic = topicTitle(item.topics?.[0]);
    if (topic) top.append(text('span',topic));
    article.append(top);
    const h = text('h2',null,'card-title'); h.append(entryLink(item,'card-link')); article.append(h);
    article.append(text('p',state.q ? item.snippet || item.summary || '' : item.summary || '','card-summary'));
    const meta = text('div',null,'card-meta');
    const publication = [authorText(item), item.year].filter(Boolean).join(' · ');
    if (publication) meta.append(text('span',publication));
    const status = cardStatus(item);
    if (status) meta.append(text('span',status));
    article.append(meta); return article;
  }
  function renderHome() {
    const volumes = state.metadata.volumes || [];
    const recent = Object.entries(reading.records).sort((left,right) => (right[1].updatedAt || 0) - (left[1].updatedAt || 0))[0];
    const selectedChapter = volumes.flatMap(volume => volume.chapters || []).find(chapter => chapter.id === 'guide-G22' && chapter.available);
    const featured = recent ? {...recent[1],id:recent[0]} : selectedChapter || state.metadata.featured?.[0];
    $('featured').replaceChildren(...(featured ? [featureCard(featured, Boolean(recent))] : []));
    const entries = volumes.map(volume => {
      const box = text('article',null,'volume');
      box.dataset.volume = volume.id;
      const title = text('h3',volume.title,'volume-title');
      title.id = 'volume-title-' + volume.id;
      box.setAttribute('aria-labelledby', title.id);
      const heading = text('div',null,'volume-header');
      heading.append(title);
      if (volumeMarks[volume.id]) {
        const mark = text('span',volumeMarks[volume.id],'volume-symbol');
        mark.setAttribute('aria-hidden','true');
        heading.append(mark);
      }
      box.append(heading);
      const chapters = text('ul',null,'volume-chapters');
      for (const chapter of volume.chapters || []) {
        const row = text('li');
        row.append(chapter.available ? entryLink(chapter,'chapter-link') : text('span',chapter.title + ' · 编写中','chapter-unavailable'));
        chapters.append(row);
      }
      box.append(chapters);
      return box;
    });
    $('volumes').replaceChildren(...entries);
    $('volumes').setAttribute('aria-busy','false');
  }
  function featureCard(item, resume) {
    const featured = text('article',null,'featured-card');
    const cover = text('div',null,'book-cover');
    cover.setAttribute('aria-hidden','true');
    cover.append(text('span','XAS','book-cover-title'));
    const imprint = text('span',null,'book-imprint');
    imprint.append(text('em','χ'),text('span','WebXAS'));
    cover.append(imprint);
    const copy = text('div');
    const label = resume ? '继续阅读' : '精选章节';
    const heading = text('h2');
    heading.append(entryLink(item,'feature-link'));
    copy.append(text('p',label,'feature-label'),heading);
    featured.append(cover,copy);
    return featured;
  }
  function localBookmarks() {
    const q = state.q.toLocaleLowerCase();
    const all = Object.entries(reading.records).filter(([,r]) => r.bookmarked).map(([id,r]) => ({...r,id})).sort((a,b) => (b.updatedAt||0)-(a.updatedAt||0));
    const items = all.filter(r => (!state.topic || (r.topics || []).includes(state.topic)) && (!state.kind || r.kind === state.kind) && (!q || [r.title,r.summary,r.authors].join(' ').toLocaleLowerCase().includes(q)));
    const pageSize = 12, last = Math.max(1,Math.ceil(items.length/pageSize)); state.page = Math.min(state.page,last);
    return {items:items.slice((state.page-1)*pageSize,state.page*pageSize),total:items.length,page:state.page,page_size:pageSize};
  }
  function indexCard(item) {
    const article = text('article',null,'index-card');
    const heading = text('h3');
    heading.append(entryLink(item,'index-title-link'));
    const kind = item.subtype || item.kind_label || kindTitle(item.kind);
    const publication = [authorText(item), item.year].filter(Boolean).join(' · ');
    article.append(text('p',kind,'index-kind'),heading);
    if (publication) article.append(text('p',publication,'index-credit'));
    if (item.summary) article.append(text('p',item.summary,'index-summary'));
    const related = item.original_refs || [];
    if (related.length === 1) {
      article.append(originalLink(related[0],'查看原始资料 →',item.id));
    } else if (related.length) {
      const files = text('details',null,'index-originals');
      files.append(text('summary','原始资料（' + related.length + '）'));
      files.append(...related.map(original => originalLink(original,original.title,item.id)));
      article.append(files);
    }
    return article;
  }
  async function loadEntries() {
    state.request?.abort(); const controller = new AbortController(); state.request = controller;
    updateFilterUI(); $('load-error').hidden = true; $('empty').hidden = true; $('pagination').hidden = true;
    if (isHome()) { renderHome(); $('entry-grid').replaceChildren(); $('entry-grid').setAttribute('aria-busy','false'); return; }
    $('result-count').textContent = '正在查找…'; $('entry-grid').setAttribute('aria-busy','true'); $('entry-grid').replaceChildren();
    const p = new URLSearchParams({q:state.q,topic:state.topic,kind:state.kind,page:state.page,page_size:12});
    if (['library','atlas'].includes(state.view)) p.set('view',state.view);
    try {
      if (state.view === 'originals') {
        const originals = await api('/api/knowledge/originals', controller.signal);
        if (state.request !== controller) return;
        const query = state.q.normalize('NFKC').toLowerCase();
        const shelf = window.WebXASKnowledgeShelf;
        const matching = originals.filter(item => (!state.kind || shelf.mediaType(item) === state.kind) && [item.title,item.credit,item.version,item.url]
          .join(' ').normalize('NFKC').toLowerCase().includes(query));
        $('entry-grid').replaceChildren(...shelf.render(matching, {text, originalLink, externalLink}));
        $('entry-grid').setAttribute('aria-busy','false');
        $('result-count').textContent = matching.length + ' 项原始资料';
        $('empty').hidden = matching.length !== 0;
        $('empty-title').textContent = '没有找到相关原文';
        $('empty-description').textContent = '试试原标题、作者姓氏或 DOI。';
        return;
      }
      const data = state.view === 'bookmarks' ? localBookmarks() : await api('/api/knowledge/entries?' + p.toString(),controller.signal);
      if (state.request !== controller) return;
      const lastPage = Math.max(1,Math.ceil(data.total/data.page_size));
      if (state.page > lastPage) { state.page=lastPage; writeLocation(); return loadEntries(); }
      const cards = state.view === 'atlas'
        ? window.WebXASKnowledgeAtlas.render(data.items, {text, entryLink, safeImage,
          showFeatured:!state.q && !state.topic && state.page === 1, featuredKind:state.kind})
        : data.items.map(state.view === 'library' ? indexCard : card);
      writeLocation(); $('entry-grid').replaceChildren(...cards); $('entry-grid').setAttribute('aria-busy','false');
      $('result-count').textContent = data.total + (state.view === 'bookmarks' ? ' 篇书签' : ' 条内容');
      $('empty').hidden = data.total !== 0;
      $('empty-title').textContent = state.view === 'bookmarks' && !state.q && !state.topic && !state.kind ? '还没有收藏的篇章' : '没有找到相关资料';
      $('empty-description').textContent = state.view === 'bookmarks' ? '在正文右上方点“加入书签”，下次从这里接着读。' : '试试更短的关键词，或放宽主题和类型。';
      $('pagination').hidden = lastPage <= 1; $('page-label').textContent = data.page + ' / ' + lastPage;
      $('previous-page').disabled = data.page <= 1; $('next-page').disabled = data.page >= lastPage;
    } catch (error) {
      if (error.name === 'AbortError') return;
      $('entry-grid').replaceChildren(); $('entry-grid').setAttribute('aria-busy','false'); $('load-error').hidden = false; $('error-message').textContent = error.message; $('result-count').textContent = '';
    }
  }
  function filtersChanged(push = true) { clearTimeout(state.timer); state.page=1; state.entry=''; writeLocation(push); loadEntries(); }
  function resetFilters() { state.q=''; state.kind=''; state.topic=''; $('search').value=''; filtersChanged(); }
  function navigate(view = '') {
    saveProgress(); state.readerRequest?.abort(); state.current=null; state.entry=''; state.original=''; state.originalPart=null; state.view=view; state.q=''; state.topic=''; state.kind=''; state.page=1; $('search').value='';
    state.restoring=false;
    showBrowse(); writeLocation(true); window.scrollTo({top:0,behavior:'instant'}); return loadEntries();
  }
  async function showVolumes() {
    if (!isHome() || state.entry || state.original) await navigate();
    const directory = $('volume-directory');
    $('volume-heading').focus({preventScroll:true});
    directory.scrollIntoView({block:'start',behavior:'instant'});
  }
  async function restoreBrowse() {
    const url = locationUrl(false);
    const position = browsePositions.get(url);
    showBrowse();
    await loadEntries();
    if (state.entry || state.original || locationUrl(false) !== url) return;
    if (location.hash === '#volume-directory' && isHome()) {
      await showVolumes();
      return;
    }
    if (position) {
      const container = position.volume ? document.querySelector(`[data-volume="${CSS.escape(position.volume)}"]`) : $('browse-page');
      const selector = position.originalId
        ? `[data-original-id="${CSS.escape(position.originalId)}"]`
        : `[data-entry-id="${CSS.escape(position.entryId)}"]`;
      const trigger = container?.querySelector(selector);
      const details = trigger?.closest('details');
      if (details) details.open = true;
      trigger?.focus({preventScroll:true});
    }
    window.scrollTo({top:position?.top || 0,behavior:'instant'});
  }
  function showBrowse() {
    clearOriginal();
    $('reader').hidden = true; $('browse-page').hidden = false; document.title = 'WebXAS · XAS 知识图书馆';
    if ($('figure-viewer').open) $('figure-viewer').close(); updateBookmarkCount();
  }
  function sourceAnchor(sourceId) { return 'source-' + String(sourceId).replace(/[^a-zA-Z0-9_-]/g,'-'); }
  function sourceReference(citation, index) {
    const label = '[' + (index+1) + '] ' + (citation.title || '原文依据') + (citation.locator ? ' · ' + citation.locator : '');
    const original = state.current?.originals?.find(item => item.id === citation.original_id);
    if (original) return originalLink(original, label + ' · 对照原文');
    if (safeUrl(citation.url)) return externalLink(label,citation.url);
    if (citation.source_id) {
      const a = text('a',label); a.href = '#' + sourceAnchor(citation.source_id);
      a.addEventListener('click',() => { const sources=$('reader-sources'); if (sources) sources.open=true; }); return a;
    }
    return text('span',label);
  }
  function renderSources(sources) {
    const section = text('details',null,'sources-section'); section.id='reader-sources';
    section.append(text('summary','来源与阅读范围 · ' + sources.length));
    for (const [index,source] of sources.entries()) {
      const box=text('div',null,'source-card'); box.id=sourceAnchor(source.id || index);
      box.append(externalLink(source.title || '查看原始来源',source.url,'source-title'));
      if (source.locator) box.append(text('p',source.locator,'source-location'));
      const details=text('details'); details.append(text('summary','阅读与获取说明'));
      for (const field of ['read_scope','access']) if (source[field]) details.append(text('p',source[field]));
      if (source.checked_at) details.append(text('p','核验日期：' + source.checked_at));
      box.append(details); section.append(box);
    }
    return section;
  }
  function openFigure(block, url) {
    $('figure-full').src=url; $('figure-full').alt=block.alt || block.caption || 'XAS 科学图件';
    $('figure-caption').textContent=[block.caption,block.author ? '作者：' + block.author : '',block.license ? '许可：' + block.license : ''].filter(Boolean).join(' · ');
    $('figure-viewer').showModal(); $('close-figure').focus();
  }
  function renderBlock(block) {
    if (block.type === 'paragraph') return text('p',block.text || '');
    if (block.type === 'heading') return text('h3',block.text || '');
    if (block.type === 'list') { const ul=text('ul'); for (const item of block.items || []) ul.append(text('li',item)); return ul; }
    if (block.type === 'entry_links') {
      const box=text('div',null,'reader-entry-links');
      if (block.label) box.append(text('p',block.label));
      const list=text('ul');
      for (const item of block.links) { const row=text('li'); row.append(entryLink(item)); list.append(row); }
      box.append(list); return box;
    }
    if (block.type === 'note') return text('aside',block.text || '','reader-note');
    if (block.type === 'formula') {
      const box=text('div',null,'reader-formula'); box.append(text('p',block.text || '','formula-text'));
      if (block.label) box.append(text('p',block.label,'formula-label')); return box;
    }
    if (block.type === 'table') {
      const wrap=text('div',null,'reader-table-wrap'); wrap.tabIndex=0; wrap.setAttribute('role','region'); wrap.setAttribute('aria-label',block.label || '数据表格，可横向滚动');
      const table=text('table',null,'reader-table'), head=text('thead'), row=text('tr');
      for (const value of block.headers || []) { const th=text('th',value); th.scope='col'; row.append(th); }
      head.append(row); table.append(head); const body=text('tbody');
      for (const [index,values] of (block.rows || []).entries()) {
        const tr=text('tr');
        const url=Array.isArray(block.row_urls) && safeUrl(block.row_urls[index]);
        for (const [column,value] of values.entries()) {
          const timecode=column === 0 && url;
          const td=text('td',timecode ? null : value);
          if (timecode) td.append(externalLink(value,url,'reader-timecode'));
          tr.append(td);
        }
        body.append(tr);
      }
      table.append(body); wrap.append(table); return wrap;
    }
    if (block.type === 'figure') {
      const figure=text('figure',null,'reader-figure');
      const url=safeImage(block.url || (block.asset_id ? '/api/knowledge/assets/' + encodeURIComponent(block.asset_id) : ''));
      if (url) {
        const button=text('button',null,'figure-button'), img=text('img'); button.type='button'; button.setAttribute('aria-label','放大图件：' + (block.alt || block.caption || '科学图件'));
        img.src=url; img.alt=block.alt || block.caption || 'XAS 科学图件'; img.loading='lazy'; img.decoding='async';
        img.addEventListener('error',() => { button.replaceChildren(text('span','图件暂时无法显示，请查看原始来源。','loading-line')); button.disabled=true; });
        button.append(img); button.addEventListener('click',() => openFigure(block,url)); figure.append(button);
      } else figure.append(text('p','图件暂时不可用，请查看原始来源。','reader-note'));
      const caption=text('figcaption');
      if (block.caption) caption.append(text('span',block.caption));
      if (block.author || block.license) caption.append(text('span',[block.author ? '作者：' + block.author : '',block.license ? '许可：' + block.license : ''].filter(Boolean).join(' · '),'figure-hint'));
      if (safeUrl(block.source_url)) caption.append(externalLink(' 查看图件来源 ↗',block.source_url));
      figure.append(caption); return figure;
    }
    return null;
  }
  function renderEntry(entry) {
    const content=$('reader-content'); content.replaceChildren(); const toc=[];
    const volume = state.metadata.volumes?.find(item => item.chapters?.some(chapter => chapter.id === entry.id));
    const chapterLabel = volume ? volume.title + ' · ' + entry.id.replace(/^guide-/, '') : entry.subtype || entry.kind_label || kindTitle(entry.kind);
    content.append(text('p',chapterLabel,'reader-kind'));
    const h=text('h1',entry.title,'reader-title'); h.id='reader-title'; h.tabIndex=-1; content.append(h);
    const meta=text('div',null,'reader-meta');
    if (entry.kind !== 'guide') {
      for (const value of [authorText(entry),entry.year,entry.read_status_label]) if (value) meta.append(text('span',value));
      if (meta.childElementCount) content.append(meta);
    }
    if (entry.summary) content.append(text('p',entry.summary,'reader-lead'));
    if (entry.kind === 'video_note' && safeUrl(entry.url)) {
      content.append(externalLink('在 B 站观看本集 ↗',entry.url,'source-action'));
    }
    if (entry.originals?.length) {
      const shelf = text('details',null,'entry-originals');
      shelf.append(text('summary','对照原始资料 · ' + entry.originals.length));
      const links = text('div',null,'entry-original-links');
      for (const original of entry.originals) links.append(originalLink(original));
      shelf.append(links);
      content.append(shelf);
    } else if (entry.kind !== 'video_note' && safeUrl(entry.url)) {
      content.append(externalLink(entry.kind === 'video' ? '在来源网站观看 ↗' : '查看原始资料 ↗',entry.url,'source-action'));
    }
    if (entry.kind === 'video') {
      const parts = (entry.originals || []).flatMap(original => original.video?.parts || []);
      const links = parts.map(part => videoNoteLink(part, null, 'P' + part.p + ' · ' + (part.note_title || part.label))).filter(Boolean);
      if (links.length) {
        const notes = text('details',null,'reader-video-notes');
        const list = text('div');
        list.append(...links);
        notes.append(text('summary','分集笔记 · ' + links.length),list);
        content.append(notes);
      }
    }
    for (const [index,block] of (entry.sections || []).entries()) {
      if (block.title === '阅读问题' && block.paragraphs?.length === 1 && block.paragraphs[0] === entry.summary) continue;
      const section=text('section',null,'reader-section'); section.id=block.id ? String(block.id).replace(/[^a-zA-Z0-9_-]/g,'-') : 'section-' + index;
      if (block.title) section.append(text('h2',block.title));
      const tocLink=text('a',block.title || '第 ' + (index+1) + ' 节'); tocLink.href='#' + section.id; tocLink.dataset.section=section.id;
      tocLink.addEventListener('click', event => {
        event.preventDefault();
        $('toc-details').open = false;
        section.tabIndex = -1;
        section.focus({preventScroll:true});
        section.scrollIntoView({block:'start'});
      });
      toc.push(tocLink);
      if (Array.isArray(block.blocks) && block.blocks.length) {
        for (const item of block.blocks) { const node=renderBlock(item); if (node) section.append(node); }
      } else {
        for (const p of block.paragraphs || []) section.append(text('p',p));
        if (block.bullets?.length) { const ul=text('ul'); for (const b of block.bullets) ul.append(text('li',b)); section.append(ul); }
      }
      if (block.citations?.length) { const refs=text('details',null,'section-references'); refs.append(text('summary','本节出处 · ' + block.citations.length)); const citations=text('div',null,'citations'); for (const [i,citation] of block.citations.entries()) citations.append(sourceReference(citation,i)); refs.append(citations); section.append(refs); }
      content.append(section);
    }
    if (entry.sources?.length) {
      content.append(renderSources(entry.sources));
      const sourceLink = text('a', '来源与阅读范围');
      sourceLink.href = '#reader-sources';
      sourceLink.addEventListener('click', event => {
        event.preventDefault();
        const details = $('reader-sources');
        details.open = true;
        $('toc-details').open = false;
        details.querySelector('summary').focus({preventScroll:true});
        details.scrollIntoView({ block: 'start' });
      });
      toc.push(sourceLink);
    }
    if (entry.previous || entry.next) {
      const nav=text('nav',null,'chapter-navigation'); nav.setAttribute('aria-label','章节导航');
      for (const [key,label] of [['previous','← 上一章'],['next','下一章 →']]) {
        if (entry[key]) { const a=entryLink(entry[key]); a.replaceChildren(text('small',label),text('span',entry[key].title)); nav.append(a); } else nav.append(text('span'));
      }
      content.append(nav);
    }
    if (entry.related?.length) {
      const related=text('section',null,'related-section'); related.append(text('h2','延伸阅读'));
      for (const item of entry.related) { const a=entryLink(item,'related-item',''); a.append(text('span',item.title),text('span','→')); related.append(a); }
      content.append(related);
    }
    $('reader-toc').replaceChildren(...toc);
    $('toc-details').open = false;
    $('toc-details').hidden = !toc.length;
    updateBookmarkButton(); document.title=entry.title + ' · WebXAS 知识图书馆';
  }
  function updateBookmarkButton() {
    const marked=Boolean(reading.records[state.entry]?.bookmarked);
    $('bookmark-entry').setAttribute('aria-pressed',String(marked)); $('bookmark-entry').textContent=marked ? '已加入书签' : '加入书签';
  }
  function recordCurrent() {
    const e=state.current; if (!e) return null;
    if (['__proto__','constructor','prototype'].includes(e.id)) return null;
    const old=reading.records[e.id] || {};
    reading.records[e.id]={...old,id:e.id,title:e.title,kind:e.kind,kind_label:e.kind_label,subtype:e.subtype,summary:e.summary,authors:e.authors,topics:e.topics,reading_status:e.reading_status,read_status_label:e.read_status_label,updatedAt:Date.now()};
    return reading.records[e.id];
  }
  function saveProgress() {
    if (!state.current || state.restoring || $('reader').hidden) return;
    const record=recordCurrent(); if (!record) return;
    const article=$('reader-article'), top=article.getBoundingClientRect().top+window.scrollY;
    const distance=Math.max(1,article.offsetHeight-window.innerHeight+100);
    const offset=Math.max(0,window.scrollY-top+90);
    record.percent=Math.max(0,Math.min(100,offset/distance*100)); record.offset=offset;
    let current=null;
    for (const section of $('reader-content').querySelectorAll('.reader-section')) { if (section.getBoundingClientRect().top <= 140) current=section.id; }
    record.section=current;
    for (const a of $('reader-toc').querySelectorAll('[data-section]')) {
      if (a.dataset.section === current) a.setAttribute('aria-current','location'); else a.removeAttribute('aria-current');
    }
    $('reading-progress-bar').style.width=record.percent + '%'; persist();
  }
  async function openEntry(id, push = false, trigger = null) {
    if (trigger && !state.entry && !state.original) {
      browsePositions.set(locationUrl(false), {
        top: window.scrollY,
        entryId: trigger.dataset.entryId,
        volume: trigger.closest('.volume')?.dataset.volume,
      });
    }
    clearOriginal();
    state.original=''; state.originalPart=null;
    saveProgress(); state.readerRequest?.abort(); state.request?.abort(); clearTimeout(state.timer); clearTimeout(state.scrollTimer);
    const controller=new AbortController(); state.readerRequest=controller; state.current=null; state.restoring=false; state.entry=id; writeLocation(push);
    $('browse-page').hidden=true; $('reader').hidden=false; $('reader-content').hidden=true; $('reader-loading').hidden=false;
    $('reader-loading').textContent='正在打开正文…'; $('reader-toc').replaceChildren(); $('toc-details').hidden=true; $('bookmark-entry').disabled=true;
    document.querySelector('.font-controls').open=false;
    $('close-reader').href=locationUrl(false);
    const collection = collectionTitles[state.view];
    $('close-reader').textContent='← 返回' + (state.q ? '搜索结果' : collection || (isHome() ? '书库' : '目录'));
    $('reading-progress-bar').style.width='0%'; window.scrollTo({top:0,behavior:'instant'});
    try {
      const entry=await api('/api/knowledge/entries/' + encodeURIComponent(id),controller.signal);
      if (state.readerRequest !== controller) return;
      state.current=entry; const saved=reading.records[id] ? {...reading.records[id]} : null;
      renderEntry(entry); $('reader-loading').hidden=true; $('reader-content').hidden=false; $('bookmark-entry').disabled=false;
      state.restoring=true;
      requestAnimationFrame(() => requestAnimationFrame(() => {
        if (state.readerRequest !== controller) return;
        if (saved && saved.percent > 0 && saved.percent < 99) {
          const section=saved.section ? document.getElementById(saved.section) : null;
          if (section) section.scrollIntoView({block:'start',behavior:'instant'});
          else window.scrollTo({top:Math.max(0,$('reader-article').getBoundingClientRect().top+window.scrollY+(Number(saved.offset)||0)-90),behavior:'instant'});
        } else $('reader-title').focus({preventScroll:true});
        state.restoring=false; saveProgress();
      }));
    } catch (error) {
      if (error.name === 'AbortError') return;
      state.restoring=false; $('reader-loading').hidden=true;
      const box=text('div',null,'reader-error'); box.append(text('p',error.message));
      const button=text('button','返回目录'); button.type='button'; button.addEventListener('click',closeEntry); box.append(button);
      $('reader-content').replaceChildren(box); $('reader-content').hidden=false;
    }
  }
  function closeEntry() {
    saveProgress(); state.readerRequest?.abort(); state.restoring=false; state.current=null; state.entry=''; writeLocation(true); return restoreBrowse();
  }
  function clearOriginal() {
    state.originalRequest?.abort();
    state.originalRequest = null;
    $('original-reader').hidden = true;
    $('original-reader').replaceChildren();
  }
  async function openOriginal(id, push = false, part = null) {
    saveProgress();
    state.readerRequest?.abort();
    state.request?.abort();
    clearTimeout(state.timer);
    clearTimeout(state.scrollTimer);
    clearOriginal();
    const controller = new AbortController();
    state.originalRequest = controller;
    state.original = id;
    state.originalPart = part;
    writeLocation(push);
    $('browse-page').hidden = true;
    $('reader').hidden = true;
    const panel = $('original-reader');
    panel.hidden = false;
    panel.replaceChildren(text('p','正在打开原始资料…','loading-line'));
    window.scrollTo({top:0,behavior:'instant'});
    try {
      const original = await api('/api/knowledge/originals/' + encodeURIComponent(id), controller.signal);
      if (state.originalRequest !== controller) return;
      const exitLabel = state.entry ? '← 返回解读' : '← 返回' + (collectionTitles[state.view] || '原始资料');
      panel.replaceChildren(window.WebXASOriginals.render(original, {
        onExit:closeOriginal, exitLabel, selectedPart:state.originalPart, videoNoteLink,
        onPartChange:part => { state.originalPart = part; writeLocation(); },
      }));
      panel.querySelector('.original-reader__title')?.focus({preventScroll:true});
      document.title = original.title + ' · WebXAS 原始资料';
    } catch (error) {
      if (state.originalRequest !== controller) return;
      const box = text('div',null,'error-state');
      const back = text('button','返回目录');
      back.type = 'button';
      back.addEventListener('click',closeOriginal);
      box.append(text('p',error.message),back);
      panel.replaceChildren(box);
    }
  }
  function closeOriginal() {
    clearOriginal();
    state.original = ''; state.originalPart = null;
    if (isHome() && !state.entry) state.view = 'originals';
    writeLocation(true);
    return state.entry ? openEntry(state.entry) : restoreBrowse();
  }
  function applyFont() { document.documentElement.style.setProperty('--reader-font',reading.fontSize + 'px'); $('font-smaller').disabled=reading.fontSize<=16; $('font-larger').disabled=reading.fontSize>=24; }
  function resizeFont(delta) { reading.fontSize=Math.max(16,Math.min(24,reading.fontSize+delta)); applyFont(); persist(); }
  async function initialize() {
    readLocation(); applyFont();
    $('volumes').replaceChildren(text('p','正在加载目录…','loading-line'));
    try {
      state.metadata=await api('/api/knowledge'); renderNavigation();
      if (state.topic && !state.metadata.topics.some(t => t.id === state.topic)) state.topic='';
      if (state.kind && !availableKinds().some(k => k.id === state.kind)) state.kind='';
      if (state.view === 'originals') state.topic = '';
      if (state.original) await openOriginal(state.original, false, state.originalPart);
      else if (state.entry) await openEntry(state.entry);
      else await restoreBrowse();
    } catch (error) { $('load-error').hidden=false; $('error-message').textContent=error.message; $('volumes').replaceChildren(); $('volumes').setAttribute('aria-busy','false'); }
  }
  for (const a of document.querySelectorAll('[data-view]')) a.addEventListener('click',event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button !== 0) return;
    event.preventDefault(); navigate(a.dataset.view);
  });
  $('volume-shortcut').addEventListener('click', event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button !== 0) return;
    if (!state.metadata) return;
    event.preventDefault();
    showVolumes();
  });
  let composing=false;
  function submitSearch(push = true) {
    state.q=$('search').value.trim();
    filtersChanged(push);
  }
  function searchChanged() {
    $('clear-search').hidden=!$('search').value; if (composing) return;
    clearTimeout(state.timer); state.timer=setTimeout(() => submitSearch(false),320);
  }
  $('search').addEventListener('compositionstart',() => { composing=true; clearTimeout(state.timer); });
  $('search').addEventListener('compositionend',() => { composing=false; searchChanged(); }); $('search').addEventListener('input',searchChanged);
  $('search-form').addEventListener('submit',event => { event.preventDefault(); submitSearch(); });
  $('clear-search').addEventListener('click',() => { $('search').value=''; state.q=''; filtersChanged(); $('search').focus(); });
  $('topic-select').addEventListener('change',event => { state.topic=event.target.value; filtersChanged(); });
  $('reset-filters').addEventListener('click',resetFilters);
  $('empty-reset').addEventListener('click',() => state.q || state.topic || state.kind ? resetFilters() : navigate());
  $('previous-page').addEventListener('click',() => { state.page--; writeLocation(true); loadEntries(); $('results').scrollIntoView({block:'start'}); });
  $('next-page').addEventListener('click',() => { state.page++; writeLocation(true); loadEntries(); $('results').scrollIntoView({block:'start'}); });
  $('retry').addEventListener('click',() => state.metadata ? loadEntries() : initialize());
  $('close-reader').addEventListener('click',event => { if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return; event.preventDefault(); closeEntry(); });
  $('font-smaller').addEventListener('click',() => resizeFont(-1)); $('font-larger').addEventListener('click',() => resizeFont(1));
  $('bookmark-entry').addEventListener('click',() => {
    const record=recordCurrent(); if (!record) return; record.bookmarked=!record.bookmarked; persist(); updateBookmarkButton(); updateBookmarkCount();
    $('announcement').textContent=record.bookmarked ? '已加入本机书签。' : '已移除书签。';
  });
  $('close-figure').addEventListener('click',() => $('figure-viewer').close());
  $('figure-viewer').addEventListener('click',event => {
    if (event.target !== $('figure-viewer')) return;
    const rect=$('figure-viewer').getBoundingClientRect();
    if (event.clientX<rect.left || event.clientX>rect.right || event.clientY<rect.top || event.clientY>rect.bottom) $('figure-viewer').close();
  });
  addEventListener('scroll',() => { if (state.current && !state.restoring) { clearTimeout(state.scrollTimer); state.scrollTimer=setTimeout(saveProgress,150); } },{passive:true});
  addEventListener('pagehide',saveProgress);
  addEventListener('popstate',() => {
    saveProgress(); state.readerRequest?.abort(); state.current=null; state.restoring=false; readLocation();
    if (state.original) openOriginal(state.original, false, state.originalPart);
    else if (state.entry) openEntry(state.entry);
    else restoreBrowse();
  });
  addEventListener('keydown',async event => {
    if (event.key === '/' && !['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName) && !$('figure-viewer').open) {
      event.preventDefault();
      if (!$('original-reader').hidden) await closeOriginal();
      if (!$('reader').hidden) await closeEntry();
      $('search').focus();
    }
  });
  initialize();
})();
