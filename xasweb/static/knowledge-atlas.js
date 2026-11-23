(() => {
  'use strict';

  const newvilleCredit = 'Matthew Newville · CC BY-SA 3.0';
  const znseCredit = 'WebXAS 教学再绘 · XrayLarch 公开 ZnSe 实谱';
  const znseCase = {
    id: 'guide-G23',
    asset: 'znse-r-fit',
    title: 'ZnSe 的 R 空间拟合',
    scope: 'Zn K 边 EXAFS',
    alt: 'ZnSe 实测与 Zn–Se 单路径模型的 R 空间模，浅蓝区域标出名义拟合范围',
    caption: '对照实测谱与 Zn–Se 单路径模型。正文给出拟合范围、复残差图和复算资料。',
    credit: znseCredit,
  };
  const teachingFigures = [
    {
      id: 'guide-G01',
      asset: 'foundation-fe-models-xanes',
      title: 'Fe 金属与化合物的 K 边 XANES',
      alt: 'Fe 金属和多种 Fe 化合物的 K 边 XANES，曲线纵向错开',
      caption: '比较不同材料的近边谱形。曲线纵向错开显示。',
      credit: newvilleCredit,
    },
    {
      id: 'guide-G22',
      asset: 'znse-absorption',
      title: '从探测器通道计算 ZnSe 吸收谱',
      alt: '由透射通道与入射通道读数的比值计算的 ZnSe 吸收谱，以及从边后数据拟合的背景',
      caption: '并列查看通道比算出的 μ(E) 与边后背景。',
      credit: znseCredit,
    },
    {
      id: 'guide-G05',
      asset: 'foundation-feo-1shell-chirmag',
      title: 'FeO 第一壳拟合的 R 空间模',
      alt: 'FeO 第一壳拟合中蓝色实验与红色模型在第一峰附近接近，实验还包含高壳峰',
      caption: '用第一壳拟合辨认第一峰与拟合范围以外的高壳峰。',
      credit: newvilleCredit,
    },
  ];

  function section(id, title, cards, layout, text) {
    const element = text('section', null, 'atlas-section');
    element.setAttribute('aria-labelledby', id);
    const heading = text('h2', title, 'atlas-section-title');
    heading.id = id;
    const grid = text('div', null, `atlas-cards ${layout}`);
    grid.append(...cards);
    element.append(heading, grid);
    return element;
  }

  function previewCard(preview, options, prominent = false) {
    const { text, entryLink, safeImage } = options;
    const url = safeImage(`/api/knowledge/assets/${preview.asset}`);
    if (!url) return null;

    const article = text('article', null, 'atlas-card atlas-preview-card');
    const link = entryLink(preview, 'atlas-card-link', '');
    const figure = text('figure', null, 'atlas-preview-figure');
    const plot = text('div', null, 'atlas-preview-plot');
    const image = text('img');
    image.alt = preview.alt;
    image.loading = prominent ? 'eager' : 'lazy';
    image.decoding = 'async';
    image.addEventListener('error', () => {
      plot.replaceChildren(text('p', '图件加载失败，可打开正文查看来源。', 'atlas-plot-error'));
    }, { once: true });
    image.src = url;
    plot.append(image);
    figure.append(plot, text('figcaption', preview.credit, 'atlas-card-credit'));

    const copy = text('div', null, 'atlas-preview-copy');
    const title = text('h3', preview.title, 'atlas-card-title');
    title.id = `atlas-preview-${preview.asset}`;
    if (preview.scope) copy.append(text('p', preview.scope, 'atlas-card-scope'));
    copy.append(title, text('p', preview.caption, 'atlas-card-summary'));
    link.setAttribute('aria-labelledby', title.id);
    link.append(figure, copy);
    article.append(link);
    return article;
  }

  function sourceCard(item, { text, entryLink }) {
    const isFigure = item.kind === 'figure';
    const article = text('article', null, 'atlas-card atlas-source-card');
    const link = entryLink(item, 'atlas-card-link', '');
    if (item.subtype) {
      const scope = isFigure ? `原文 ${item.subtype}` : item.subtype;
      link.append(text('p', scope, 'atlas-card-scope'));
    }
    const title = text('h3', item.title, 'atlas-card-title');
    title.id = `atlas-title-${item.id}`;
    link.append(title);
    const summary = isFigure ? item.snippet : item.summary;
    if (summary) link.append(text('p', summary, 'atlas-card-summary'));
    const credit = [item.authors, item.year].filter(Boolean).join(' · ');
    if (credit) link.append(text('p', credit, 'atlas-card-credit'));
    if (item.read_status_label) link.append(text('p', item.read_status_label, 'atlas-card-status'));
    link.setAttribute('aria-labelledby', title.id);
    article.append(link);
    return article;
  }

  function render(items, options) {
    const { text, showFeatured, featuredKind } = options;
    const sections = [];
    if (showFeatured && items.length) {
      if (!featuredKind || featuredKind === 'case') {
        const card = previewCard(znseCase, options, true);
        if (card) sections.push(section('atlas-reference-case', '实谱案例', [card], 'atlas-reference-card', text));
      }
      if (!featuredKind || featuredKind === 'figure') {
        const cards = teachingFigures.map(figure => previewCard(figure, options)).filter(Boolean);
        if (cards.length) sections.push(section('atlas-teaching-figures', '教学图谱', cards, 'atlas-figure-cards', text));
      }
    }

    const figures = items.filter(item => item.kind === 'figure');
    const cases = items.filter(item => item.kind === 'case');
    if (figures.length) {
      const cards = figures.map(item => sourceCard(item, options));
      sections.push(section('atlas-paper-figures', '论文图件索引', cards, 'atlas-figure-cards', text));
    }
    if (cases.length) {
      const cards = cases.map(item => sourceCard(item, options));
      sections.push(section('atlas-research-cases', '研究案例', cards, 'atlas-case-cards', text));
    }
    return sections;
  }

  window.WebXASKnowledgeAtlas = { render };
})();
