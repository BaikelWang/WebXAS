const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const path = require('node:path');
const {test} = require('node:test');
const vm = require('node:vm');

test('the fit card distinguishes missing, dimensionless and declared units', () => {
  const source = readFileSync(path.join(__dirname, '../../../../xasweb/static/auto-results.js'), 'utf8');
  const context = vm.createContext({});
  new vm.Script(source, {filename:'auto-results.js'}).runInContext(context);
  const metric = Object.freeze({value:0.25, uncertainty:0.05, unit:'not_declared'});
  assert.equal(context.metricValue(metric), '0.25 ± 0.05（单位未记录）');
  assert.equal(metric.unit, 'not_declared');
  assert.equal(context.metricValue({value:1.2, unit:'eV'}), '1.2 eV');
  assert.equal(context.metricValue({value:1.2, unit:'1'}), '1.2');
});
