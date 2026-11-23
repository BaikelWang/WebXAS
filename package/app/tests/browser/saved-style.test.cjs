const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const path = require('node:path');
const {test} = require('node:test');
const vm = require('node:vm');

// Exercise the shipped comparison and its two click handlers without starting
// the unrelated page boot sequence, worker or downloader.
const source = readFileSync(path.join(__dirname, '../../../../xasweb/static/auto.js'), 'utf8');
const first = source.indexOf('function matchesDeliveryOptions(');
const after = source.indexOf("$('regenerate').onclick", first);
assert.ok(first >= 0 && after > first);
const script = new vm.Script(source.slice(first, after), {filename:'auto.js'});

function workbench() {
  const controls = {redraw:{}, download:{}, 'export-status':{}};
  const state = {workflow:{id:'analysis-1'}, delivery:null};
  const input = {style:{profile:'classic',dpi:300}, families:['F01','F04'], pair:null};
  const downloads = [];
  let renders = 0;
  function snapshot() {
    const saved = {...input.style};
    if (saved.profile === 'classic') delete saved.profile;
    return {id:'bundle-'+renders, workflow_id:state.workflow.id, presentation:'individual',
      style:saved, figure_families:[...input.families], difference_pair:input.pair};
  }
  state.delivery = snapshot();
  const context = {
    S:state, $:id=>controls[id], action:handler=>handler,
    selectedFamilies:()=>input.families, differencePair:()=>input.pair, style:()=>input.style,
    createDelivery:async()=>{ renders++; state.delivery=snapshot(); },
    auto:route=>route, fetch:async route=>route, downloadResponse:async route=>downloads.push(route)
  };
  vm.createContext(context);
  script.runInContext(context);
  return {input, state, downloads, renders:()=>renders, redraw:()=>controls.redraw.onclick(),
    download:()=>controls.download.onclick(), matches:()=>context.matchesDeliveryOptions(state.delivery)};
}

test('unchanged classic or nature redraw and download reuse the same bundle', async () => {
  const page=workbench();
  assert.ok(page.matches()); // Classic is omitted from historical delivery manifests.
  await page.redraw(); await page.download();
  assert.equal(page.renders(),0);
  page.input.style.profile='nature';
  await page.redraw(); await page.download(); await page.redraw();
  assert.equal(page.renders(),1);
  page.input.style.profile='classic';
  await page.redraw(); await page.download();
  assert.equal(page.renders(),2);
  assert.deepEqual(page.downloads,['/deliveries/bundle-0/download','/deliveries/bundle-1/download','/deliveries/bundle-2/download']);
});

test('family order is irrelevant and different families require rendering', async () => {
  const page=workbench();
  page.input.families.reverse();
  assert.ok(page.matches());
  await page.download();
  assert.equal(page.renders(),0);
  page.input.families=['F02','F04'];
  assert.equal(page.matches(),false);
  await page.download();
  assert.equal(page.renders(),1);
});

test('ordered difference operands, workflow, presentation and style remain significant', async () => {
  const page=workbench();
  page.input.pair=['sample','reference'];
  await page.redraw();
  page.input.pair=['reference','sample'];
  assert.equal(page.matches(),false);
  await page.redraw();
  assert.equal(page.renders(),2);
  page.input.style.dpi=600;
  assert.equal(page.matches(),false);
  page.input.style.dpi=300;
  page.state.delivery.presentation='composite';
  assert.equal(page.matches(),false);
  page.state.delivery.presentation='individual';
  page.state.workflow.id='analysis-2';
  assert.equal(page.matches(),false);
});
