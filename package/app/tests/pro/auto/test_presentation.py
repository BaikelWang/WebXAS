from copy import deepcopy

from xasweb.pro.figures.presentation import brief_insight


def row(label, value, unit='eV', basis=''):
    return {'label': label, 'value': value, 'unit': unit, 'basis': basis}


def test_brief_keeps_values_and_calibration_warning_without_rewriting_evidence():
    source = {'dataset_ids': ['a'], 'metrics': [
        row('横轴采样点数', 620, '1'), row('工作流已存边位', 8977.58, basis='未记录标准样绝对能量校准'),
        row('工作流已存白线高度', .991664, '1')], 'observations': ['full recorded observation'],
        'limitations': ['full scientific limitation']}
    original = deepcopy(source)
    display = brief_insight(source, 'F02')
    assert [m['label'] for m in display['metrics']] == ['吸收边位置', '白线强度']
    assert display['metrics'][0]['value'] == 8977.58
    assert display['notices'] == ['未做标准样能量校准。']
    assert source == original


def test_multispectrum_does_not_report_first_samples_feature_as_global_result():
    source = {'dataset_ids': ['a', 'b'], 'metrics': [row('工作流已存边位', 8977.58)]}
    display = brief_insight(source, 'F02')
    assert display['metrics'] == []
    assert display['summary'] == '2 个样品的近边谱对比。'


def test_phase_and_reference_limits_are_kept_brief_and_explicit():
    assert '不等于键长' in brief_insight({}, 'F05')['notices'][0]
    assert '不能单独确定散射元素' in brief_insight({}, 'F06')['notices'][0]
    assert '不直接等同含量或价态' in brief_insight({}, 'F09')['notices'][0]


def test_joint_fit_scope_and_nonconvergence_remain_visible():
    source = {'metrics': [row('本谱 residual_chi 的未加权 RMS', .02, '1')],
              'method': {'fit_statistics_scope': [{'scope': 'joint_run'}]}}
    result = brief_insight(source, 'F07', {'converged': False})
    assert '未收敛' in result['summary']
    assert result['metrics'][0]['value'] == .02
    assert result['notices'] == ['残差为本谱分量；整体统计量来自联合拟合。']


def test_legacy_no_values_does_not_manufacture_results():
    result = brief_insight({}, 'F02')
    assert result['metrics'] == []
    assert '历史' in result['summary']


def test_single_and_multiple_input_ranges_stay_correctly_scoped():
    source = {'dataset_ids': ['a'], 'metrics': [row('横轴下界', 8780.), row('横轴上界', 11300.)]}
    assert '8780–11300 eV' in brief_insight(source, 'F01')['summary']
    source['dataset_ids'].append('b')
    result = brief_insight(source, 'F01')
    assert '8780' not in result['summary']
    assert '2 个样品' in result['summary']


def test_r_space_and_legacy_fits_keep_recorded_nonconvergence_visible():
    legacy = {'observations': ['绑定联合运行未收敛，覆盖 3 条谱。']}
    assert '未收敛' in brief_insight(legacy, 'F07', {'space': 'r'})['summary']
    structured = {'method': {'fit_statistics_scope': [{'scope': 'joint_run', 'converged': False}]}}
    assert '未收敛' in brief_insight(structured, 'F07', {'space': 'r'})['summary']
    assert '未记录' in brief_insight({}, 'F07')['summary']


def test_r_residual_components_are_not_misrepresented_as_one_complex_norm():
    source = {'metrics': [row('本谱 residual_chir_re 的未加权 RMS', .1),
                          row('本谱 residual_chir_im 的未加权 RMS', .2)]}
    display = brief_insight(source, 'F07')
    assert [m['label'] for m in display['metrics']] == ['未加权残差 RMS（Re）', '未加权残差 RMS（Im）']
    assert [m['value'] for m in display['metrics']] == [.1, .2]
