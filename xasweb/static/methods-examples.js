'use strict';
(async () => {
  const response = await fetch('/validation/manifest.json');
  if (!response.ok) return;
  const {figures} = await response.json();
  const sections = {G01:'normalization', G02:'edge', G03:'lcf', G04:'fourier', G05:'structure', G06:'wavelet', G07:'series', G08:'decomposition'};
  const link = (parent, label, href) => {
    const a = document.createElement('a');
    a.href = href;
    a.textContent = label;
    parent.append(a);
  };
  for (const record of figures) {
    const parent = document.getElementById(sections[record.id]);
    if (!parent) continue;
    const details = document.createElement('details');
    details.className = 'example';
    const summary = document.createElement('summary');
    summary.textContent = record.files.png ? '查看实验图、绘图数据与出处' : '查看实验用例与公开数据出处';
    details.append(summary);
    if (record.files.png) {
      const img = document.createElement('img');
      img.loading = 'lazy';
      img.src = '/validation/' + record.files.png;
      img.alt = record.title;
      details.append(img);
      link(details, 'CSV', '/validation/' + record.files.csv);
    }
    const p = document.createElement('p');
    p.textContent = record.experiment + ' ' + record.physical_evidence;
    details.append(p);
    link(details, '实验数据', record.source_url);
    if (record.attribution) {
      const label = document.createElement('p');
      label.textContent = '原始数据作者：' + record.attribution + '。WebXAS 重新处理并绘图。';
      details.append(label);
      link(details, '研究论文', record.article);
      link(details, 'CC BY 4.0', record.license);
    }
    parent.append(details);
  }
})().catch(error => console.warn('Gallery unavailable', error));
