(() => {
  'use strict';

  const sections = [
    {type: 'pdf', title: '电子教材与讲义'},
    {type: 'video', title: '视频'},
    {type: 'article', title: '论文'},
    {type: 'website', title: '原网页'},
  ];

  function mediaType(item) {
    return item.type === 'website' && item.url.startsWith('https://doi.org/10.') ? 'article' : item.type;
  }

  function publicationCard(item, {text, originalLink, externalLink}) {
    const card = text('li', null, 'knowledge-shelf__item knowledge-shelf__publication');
    const heading = text('h3', null, 'knowledge-shelf__title');
    const publication = externalLink(item.title, item.url, 'knowledge-shelf__doi-link');
    const arrow = text('span', ' ↗');
    arrow.setAttribute('aria-hidden', 'true');
    publication.append(arrow);
    heading.append(publication);
    const meta = text('p', item.credit, 'knowledge-shelf__credit');
    const notes = originalLink(item, '阅读笔记');
    notes.classList.add('knowledge-shelf__publication-notes');
    card.append(heading, meta, notes);
    return card;
  }

  function pdfDetails(item, title, credit, text) {
    const book = text('div', null, 'knowledge-shelf__book');
    const copy = text('div', null, 'knowledge-shelf__book-copy');
    copy.append(credit);
    const fallback = text('div', null, 'knowledge-shelf__cover');
    fallback.append(text('span', 'PDF', 'knowledge-shelf__format'));
    const coverPath = '/api/knowledge/originals/' + encodeURIComponent(item.id) + '/cover';
    if (item.cover_url !== coverPath) {
      fallback.append(title);
      book.append(fallback);
      return [book,copy];
    }

    const original = text('div', null, 'knowledge-shelf__cover knowledge-shelf__cover--original');
    const image = text('img', null, 'knowledge-shelf__cover-image');
    image.alt = '';
    image.loading = 'lazy';
    image.decoding = 'async';
    const label = text('span', item.cover_kind === 'cover' ? '原封面' : '原文首页', 'knowledge-shelf__cover-label');
    image.addEventListener('error', () => {
      fallback.append(title);
      original.replaceWith(fallback);
      label.remove();
    }, {once: true});
    image.src = coverPath;
    original.append(image);
    book.append(original);
    copy.prepend(label,title);
    return [book,copy];
  }

  function mediaCard(item, {text, originalLink}) {
    const card = text('li', null, 'knowledge-shelf__item');
    const link = originalLink(item, '');
    link.classList.add('knowledge-shelf__link');

    const title = text('h3', item.title, 'knowledge-shelf__title');
    title.id = `knowledge-shelf-title-${item.id}`;
    link.setAttribute('aria-labelledby', title.id);

    const credit = text('p', item.credit, 'knowledge-shelf__credit');
    credit.id = `knowledge-shelf-credit-${item.id}`;
    link.setAttribute('aria-describedby', credit.id);

    if (item.type === 'pdf') {
      link.append(...pdfDetails(item, title, credit, text));
    } else {
      const isVideo = item.type === 'video';
      const preview = text('span', null, 'knowledge-shelf__preview');
      preview.setAttribute('aria-hidden', 'true');
      if (isVideo) preview.append(text('span', null, 'knowledge-shelf__play'));

      const details = text('div', null, 'knowledge-shelf__details');
      const videoStatus = item.video?.access === 'paid_course_confirm_before_use' ? '付费课程 · 原站入口' : '查看视频';
      const status = isVideo ? videoStatus : item.embed_url ? '可在馆内加载' : '仅原站访问';
      const access = text('p', status, 'knowledge-shelf__access');
      access.id = `knowledge-shelf-access-${item.id}`;
      link.setAttribute('aria-describedby', `${credit.id} ${access.id}`);
      const arrow = text('span', '→', 'knowledge-shelf__arrow');
      arrow.setAttribute('aria-hidden', 'true');
      access.append(arrow);
      details.append(title, credit, access);
      link.append(preview, details);
    }

    card.append(link);
    return card;
  }

  function render(items, helpers) {
    const {text} = helpers;
    const shelf = [];
    for (const {type, title} of sections) {
      const matching = items.filter(item => mediaType(item) === type);
      if (!matching.length) continue;

      const section = text('section', null, 'knowledge-shelf');
      section.dataset.media = type;
      const header = text('header', null, 'knowledge-shelf__header');
      const heading = text('h2', title, 'knowledge-shelf__heading');
      heading.id = `knowledge-shelf-heading-${type}`;
      section.setAttribute('aria-labelledby', heading.id);
      header.append(heading, text('span', `${matching.length} 项`, 'knowledge-shelf__count'));

      const cards = text('ul', null, 'knowledge-shelf__items');
      cards.setAttribute('role', 'list');
      const renderCard = type === 'article' ? publicationCard : mediaCard;
      cards.append(...matching.map(item => renderCard(item, helpers)));
      section.append(header, cards);
      shelf.push(section);
    }
    return shelf;
  }

  window.WebXASKnowledgeShelf = {mediaType, render};
})();
