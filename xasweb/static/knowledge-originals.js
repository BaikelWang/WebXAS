(() => {
  'use strict';

  const bilibiliPublic = 'public_video_page_playback_unverified';
  const bilibiliPaid = 'paid_course_confirm_before_use';

  function element(tag, className, label) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (label) node.textContent = label;
    return node;
  }

  function httpsUrl(value) {
    if (typeof value !== 'string' || !/^https:\/\//i.test(value)) return null;
    try {
      const url = new URL(value);
      if (!url.hostname || url.username || url.password) return null;
      return url;
    } catch {
      return null;
    }
  }

  function pdfUrl(value, id) {
    if (typeof value !== 'string' || typeof id !== 'string' || !id) return null;
    try {
      const path = `/api/knowledge/originals/${encodeURIComponent(id)}/pdf`;
      if (!value.startsWith(path)) return null;
      const fragment = value.slice(path.length);
      if (fragment && !/^#page=[1-9]\d*$/.test(fragment)) return null;
      const url = new URL(value, window.location.origin);
      return url.pathname === path ? url : null;
    } catch {
      return null;
    }
  }

  function externalLink(label, url, className) {
    const link = element('a', className, label);
    link.href = url.href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    const icon = element('span', 'original-reader__new-tab', '↗');
    icon.setAttribute('aria-hidden', 'true');
    link.append(icon);
    link.setAttribute('aria-label', `${label}（新标签页）`);
    return link;
  }

  function provenance(original) {
    const licenseUrl = httpsUrl(original.license_url);
    const list = element('dl', 'original-reader__provenance');
    const fields = [
      ['来源', original.credit],
      [original.video ? '收录' : '版本', original.version, httpsUrl(original.source_pdf_url)],
      [original.video ? '访问' : '授权', original.license, licenseUrl],
    ];
    if (original.video) {
      const video = original.video;
      const transcript = new Map([['pending', '待转写'], ['received', '已有转写稿'], ['checked', '转写已校对']]);
      const noteCount = video.parts?.filter(part => part.note_entry_id).length || 0;
      const status = [
        video.playback_verified ? '播放已核验' : '播放未核验',
        noteCount ? `${noteCount} 份图文笔记` : transcript.get(video.transcript_status),
        !noteCount && video.content_review_status === 'unreviewed' ? '内容未审核' : null,
      ].filter(Boolean);
      fields.push(['内容状态', status.join(' · ')]);
    }
    for (const [label, value, url] of fields) {
      if (!value) continue;
      const item = element('div');
      const description = element('dd');
      if (url) description.append(externalLink(value, url));
      else description.textContent = value;
      item.append(element('dt', null, label), description);
      list.append(item);
    }
    return list;
  }

  function renderNotes(notes, jumpToPage, title = '解读笔记') {
    const panel = element('aside', 'original-reader__notes');
    panel.setAttribute('aria-label', title);
    const narrow = matchMedia('(max-width: 760px)').matches;
    const disclosure = element('details','annotation-fold');
    disclosure.open = !narrow;
    disclosure.append(element('summary',null,title));
    panel.append(disclosure);
    if (!Array.isArray(notes) || !notes.length) {
      disclosure.append(element('p', 'original-reader__message', '暂无解读笔记。'));
      return panel;
    }

    const list = element('ol', 'original-reader__note-list');
    for (const note of notes) {
      const item = element('li', 'original-reader__note');
      const page = note.page;
      const hasPage = Number.isSafeInteger(page) && page > 0;
      const label = note.locator || (hasPage ? `第 ${page} 页` : '');
      if (jumpToPage && hasPage) {
        const locator = element('button', 'original-reader__locator', label);
        locator.type = 'button';
        locator.setAttribute('aria-label', `${label}，跳至 PDF 第 ${page} 页`);
        locator.addEventListener('click', () => jumpToPage(page));
        item.append(locator);
      } else if (label) {
        item.append(element('p', 'original-reader__locator', label));
      }
      item.append(element('p', 'original-reader__note-text', note.text));
      list.append(item);
    }
    disclosure.append(list);
    return panel;
  }

  function renderExternalPreview(preview, original, publisherUrl) {
    const embedUrl = httpsUrl(original.embed_url);
    const isVideo = original.type === 'video';
    const canEmbed = (original.type === 'website' || isVideo) && embedUrl && embedUrl.origin !== window.location.origin;
    const placeholder = element('div', 'original-reader__placeholder');
    preview.append(placeholder);
    if (!canEmbed) {
      const message = publisherUrl
        ? (isVideo ? '请打开发布页观看视频。' : '请打开发布页阅读原文。')
        : '暂时没有可打开的发布页。';
      placeholder.append(element('p', 'original-reader__message', message));
      return;
    }

    const button = element('button', 'original-reader__load', isVideo ? '在本页加载视频' : '在本页加载网页');
    button.type = 'button';
    placeholder.append(button, element('p', 'original-reader__message', '加载时将连接发布方网站。'));
    button.addEventListener('click', () => {
      const frame = element('iframe', 'original-reader__frame');
      frame.title = isVideo ? '原文视频播放器' : '原文网页预览';
      const youtubeEmbed = isVideo && ['www.youtube.com', 'www.youtube-nocookie.com'].includes(embedUrl.host)
        && embedUrl.pathname.startsWith('/embed/');
      // YouTube requires an origin referrer from clients embedding its player.
      frame.referrerPolicy = youtubeEmbed ? 'strict-origin-when-cross-origin' : 'no-referrer';
      frame.setAttribute('sandbox', isVideo ? 'allow-scripts allow-same-origin allow-presentation' : 'allow-scripts allow-same-origin');
      if (isVideo) frame.allowFullscreen = true;
      frame.src = embedUrl.href;
      preview.replaceChildren(frame);
      if (publisherUrl) {
        preview.append(element('p', 'original-reader__preview-hint', '预览无法显示时，可通过上方的发布页链接访问。'));
      }
      preview.focus({preventScroll: true});
    }, {once: true});
  }

  function bilibiliPage(original) {
    const video = original.video;
    if (video.access === bilibiliPublic && /^BV[A-Za-z0-9]{10}$/.test(video.bvid || '')) {
      const canonical = 'https://www.bilibili.com/video/' + video.bvid + '/';
      if (original.url === canonical) return new URL(canonical);
    }
    if (video.access === bilibiliPaid) {
      const course = /^bilibili-course-(ss[0-9]+)$/.exec(original.id);
      const canonical = course && 'https://www.bilibili.com/cheese/play/' + course[1];
      if (canonical && original.url === canonical) return new URL(canonical);
    }
    return null;
  }

  function renderBilibiliPreview(preview, video, publisherUrl, publisherLink, {selectedPart, onPartChange, videoNoteLink}) {
    const player = element('div');
    if (!publisherUrl || video.access === bilibiliPaid) {
      player.className = 'original-reader__placeholder';
      player.append(element('p', 'original-reader__message', publisherUrl
        ? '请在 B 站课程页确认价格和观看权限。'
        : '暂时没有可打开的 B 站发布页。'));
      preview.append(player);
      return;
    }

    const parts = (video.parts || []).filter(part => Number.isSafeInteger(part.p) && part.p > 0);
    const selection = element('label', 'original-reader__part-control', '分 P');
    const select = element('select');
    for (const part of parts) {
      const duration = part.duration ? ' · ' + part.duration : '';
      const option = element('option', null, 'P' + part.p + ' · ' + part.label + duration);
      option.value = part.p;
      select.append(option);
    }
    const initialPart = parts.find(part => part.p === selectedPart) || parts.find(part => part.first_batch_reason);
    if (initialPart) select.value = initialPart.p;
    selection.append(select);
    const priority = element('p', 'original-reader__message');
    priority.setAttribute('role', 'status');
    const notes = element('div', 'original-reader__video-notes');
    const controls = element('div', 'original-reader__video-controls');
    if (parts.length > 1) controls.append(selection);
    controls.append(notes, priority);
    preview.append(controls, player);

    function showSelection() {
      const part = parts.find(item => String(item.p) === select.value);
      const direct = new URL(publisherUrl);
      if (part) direct.searchParams.set('p', part.p);
      publisherLink.href = direct.href;
      publisherLink.setAttribute('aria-label', part
        ? '在 B 站观看 P' + part.p + '（新标签页）'
        : '在 B 站观看（新标签页）');
      const noteLink = videoNoteLink?.(part, 'original-reader__video-note', '阅读本集笔记 →');
      if (noteLink && part.note_title) noteLink.setAttribute('aria-label', '阅读本集笔记：' + part.note_title);
      notes.replaceChildren(...(noteLink ? [noteLink] : []));
      const reason = noteLink ? '' : part?.first_batch_reason || (parts.length <= 1 ? video.first_batch_reason : '');
      priority.textContent = reason ? '首批转写计划 · ' + reason : '';
      controls.hidden = !(parts.length > 1 || noteLink || reason);

      const placeholder = element('div', 'original-reader__placeholder');
      const button = element('button', 'original-reader__load', '在本页加载视频');
      button.type = 'button';
      placeholder.append(button, element('p', 'original-reader__message',
        '加载时将连接 B 站。若播放器无法打开，可前往 B 站观看。'));
      player.replaceChildren(placeholder);

      button.addEventListener('click', () => {
        const src = new URL('https://player.bilibili.com/player.html');
        src.searchParams.set('bvid', video.bvid);
        src.searchParams.set('autoplay', '0');
        if (part) src.searchParams.set('p', part.p);
        const frame = element('iframe', 'original-reader__frame');
        frame.title = part ? '哔哩哔哩播放器 · P' + part.p : '哔哩哔哩播放器';
        frame.referrerPolicy = 'strict-origin-when-cross-origin';
        frame.setAttribute('sandbox', 'allow-scripts allow-same-origin allow-presentation');
        frame.allowFullscreen = true;
        frame.src = src.href;
        player.replaceChildren(frame, element('p', 'original-reader__preview-hint',
          '如果播放器无法显示，可使用上方的 B 站观看链接。'));
        preview.focus({preventScroll: true});
      }, {once: true});
      return part;
    }

    select.addEventListener('change', () => {
      const part = showSelection();
      onPartChange?.(part?.p || null);
    });
    showSelection();
  }

  function pdfFrame(url) {
    const frame = element('iframe', 'original-reader__frame');
    frame.title = 'PDF 原文';
    frame.src = url.href;
    return frame;
  }

  function render(original, {onExit, exitLabel = '← 返回原始资料', selectedPart, onPartChange, videoNoteLink} = {}) {
    const reader = element('section', 'original-reader');
    reader.setAttribute('aria-label', '原文阅读器');
    const header = element('header', 'original-reader__header');
    const toolbar = element('div', 'original-reader__toolbar');
    const back = element('button', 'original-reader__back', exitLabel);
    back.type = 'button';
    back.addEventListener('click', () => onExit?.());
    toolbar.append(back);

    const links = element('div', 'original-reader__links');
    const publisherUrl = original.video ? bilibiliPage(original) : httpsUrl(original.url);
    const internalPdf = original.type === 'pdf' ? pdfUrl(original.embed_url, original.id) : null;
    const pdfLink = internalPdf ? externalLink('打开 PDF', internalPdf) : null;
    if (pdfLink) links.append(pdfLink);
    const publisherLabel = original.video
      ? (original.video.access === bilibiliPaid ? '打开 B 站课程页' : '在 B 站观看')
      : '打开发布页';
    const publisherLink = publisherUrl ? externalLink(publisherLabel, publisherUrl) : null;
    if (publisherLink) links.append(publisherLink);
    if (links.childElementCount) toolbar.append(links);

    const title = element('h1', 'original-reader__title', original.title);
    title.tabIndex = -1;
    header.append(toolbar, title, provenance(original));

    const layout = element('div', 'original-reader__layout');
    const preview = element('section', `original-reader__preview${original.type === 'video' ? ' original-reader__preview--video' : ''}`);
    preview.setAttribute('aria-label', '原文预览');
    preview.tabIndex = -1;
    let jumpToPage;
    if (internalPdf) {
      let frame = pdfFrame(internalPdf);
      const status = element('span', 'sr-only');
      status.setAttribute('role', 'status');
      preview.append(frame, status);
      jumpToPage = page => {
        internalPdf.hash = `page=${page}`;
        // Chromium's native PDF viewer applies page fragments on initial load.
        const nextFrame = pdfFrame(internalPdf);
        frame.replaceWith(nextFrame);
        frame = nextFrame;
        pdfLink.href = internalPdf.href;
        status.textContent = `已请求跳至 PDF 第 ${page} 页`;
        const previewTop = preview.getBoundingClientRect().top;
        if (previewTop < 0 || previewTop >= window.innerHeight) {
          preview.scrollIntoView({block: 'start'});
        }
      };
    } else if (original.video) {
      renderBilibiliPreview(preview, original.video, publisherUrl, publisherLink, {selectedPart, onPartChange, videoNoteLink});
    } else {
      renderExternalPreview(preview, original, publisherUrl);
    }

    layout.append(preview, renderNotes(original.notes, jumpToPage, original.video ? '收录说明' : undefined));
    reader.append(header, layout);
    return reader;
  }

  window.WebXASOriginals = {render};
})();
