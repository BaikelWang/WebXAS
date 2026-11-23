"""Bounded, explicitly conditional estimates for one observed coordination peak.

No file names, literature targets, or sample identities enter these decisions.
Fixed nuisance parameters are actual non-varying Larch parameters. Their
sensitivity envelope is kept separate from the optimizer's statistical error.
"""
import math
import time
import numpy as np
from .ingest import InputError

def finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def first_shell_window(k, chi, kmin, kmax, rbkg=1.):
    """Choose one observed low-R peak, independently of the candidate models."""
    from larch import Group
    from larch.xafs import xftf
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    k, chi = np.asarray(k, float), np.asarray(chi, float)
    if kmax-kmin < 3 or np.sqrt(np.mean(chi**2)) < 1e-10:
        raise InputError('第一壳信号或可用 Δk 不足；保留图谱，不生成结构数值。')
    group = Group()
    xftf(k, chi, group=group, kmin=kmin, kmax=kmax, kweight=2,
         dk=1., window='hanning', rmax_out=10)
    r, magnitude = np.asarray(group.r), np.asarray(group.chir_mag)
    smooth = gaussian_filter1d(magnitude, 1.)
    floor = max(1.05, min(float(rbkg), 1.3)+.15)
    region = (r >= floor) & (r <= 3.5)
    noise_region = (r >= 7) & (r <= 9.5)
    noise = max(float(np.median(magnitude[noise_region])), 1e-14)
    maximum = float(np.max(smooth[region]))
    peaks, properties = find_peaks(smooth, prominence=max(2.5*noise, .07*maximum))
    eligible = [int(i) for i in peaks if floor <= r[i] <= 3.5
                and smooth[i] >= max(4*noise, .18*maximum)]
    if not eligible:
        raise InputError('未检出高于高 R 噪声背景的独立第一协调峰；不强行拟合。')
    peak = eligible[0]
    center = float(r[peak])
    lower, upper = max(.7, center-.55), min(4.15, center+.65)
    # Nearby lobes can belong to one phase-dependent shell. A separated peak
    # must not silently expand the QFS fit into the next coordination shell.
    separated = [i for i in eligible[1:] if r[i]-center > .8]
    if separated:
        upper = min(upper, .5*(center+float(r[separated[0]])))
    nind = 2*(kmax-kmin)*(upper-lower)/np.pi
    if nind <= 4:
        raise InputError('第一协调峰窗口的信息量不足以估计 N、R 并检查稳定性。')
    snr = float(smooth[peak]/noise)
    decision = dict(peak_R=center, rmin=lower, rmax=upper, kmin=kmin, kmax=kmax,
                    peak_magnitude=float(smooth[peak]), noise_magnitude=noise,
                    signal_to_noise=snr, confidence='moderate' if snr < 8 else 'high',
                    n_independent_estimate=float(nind),
                    eligible_peak_R=[float(r[i]) for i in eligible],
                    method='first prominent low-R peak above background exclusion and high-R noise; same window for every candidate',
                    reason='峰位未经相位校正；仅用于选择第一协调峰窗口，不作为键长。',
                    limitation='背景残余、峰重叠或弱第一壳可能影响峰选择；本窗口不是壳层归属证明。')
    window = dict(kmin=kmin, kmax=kmax, rmin=lower, rmax=upper,
                  kweight=2, dk=1., window='hanning', fitspace='r')
    return window, decision


def fit_once(groups, k, chi, window, s02, start=0, fixed=None, delta_r_limit=.25, max_nfev=120):
    from larch import Group
    from larch.fitting import param, param_group
    from larch.xafs import feffpath, feffit, feffit_dataset, feffit_transform, feffit_report
    fixed = dict(fixed or {})
    independent = 2*(window['kmax']-window['kmin'])*(window['rmax']-window['rmin'])/np.pi
    nvars = 1+3*len(groups)-len(fixed)
    if independent <= nvars+2:
        raise InputError(f'独立信息量 {independent:.1f} 不足以支持 {nvars} 个自由参数和检查余量。')
    pars = param_group()
    names = []
    def add(name, value, low, high):
        names.append(name)
        assumed = name in fixed
        setattr(pars, name, param(fixed[name] if assumed else value,
                                 vary=not assumed, min=low, max=high))
    add('de0', 2.*start, -20, 20)
    fps = []
    for i, paths in enumerate(groups):
        add(f'amp{i}', 1.+.2*start, .02, 4.)
        add(f'dr{i}', .08*start, -delta_r_limit, delta_r_limit)
        add(f'sig2{i}', .005+.002*start, 0, .04)
        fps.extend(feffpath(str(file), s02=f'{s02:.12g}*amp{i}', e0='de0',
                            deltar=f'dr{i}', sigma2=f'sig2{i}') for file, _ in paths)
    dataset = feffit_dataset(data=Group(k=np.asarray(k), chi=np.asarray(chi)), paths=fps,
                            transform=feffit_transform(**window))
    result = feffit(pars, dataset, max_nfev=max_nfev)
    if result.n_independent <= result.nvarys+2:
        raise InputError('拟合独立点数不足以支撑参数与检查余量。')
    params = {}
    for name in names:
        p = result.params[name]
        params[name] = dict(value=finite(p.value), stderr=finite(p.stderr) if p.vary else None,
                            vary=bool(p.vary), assumed=not p.vary,
                            correlations={n:finite(v) for n,v in (p.correl or {}).items()} if p.vary else {},
                            at_bound=bool(p.vary and min(abs(p.value-p.min), abs(p.max-p.value)) < 1e-4))
    n, v = float(result.n_independent), int(result.nvarys)
    residual = max(float(result.rfactor), 1e-15)
    aicc = n*np.log(residual)+2*v+2*v*(v+1)/(n-v-1)
    return dict(parameters=params, fixed_parameters=fixed,
                statistics=dict(rfactor=finite(result.rfactor), aicc=finite(aicc), n_independent=n,
                    n_variables=v, chi_square=finite(result.chi_square),
                    reduced_chi_square=finite(result.chi2_reduced),
                    converged=bool(getattr(result, 'success', False))),
                curves=dict(k=dataset.model.k.tolist(), chi_fit=dataset.model.chi.tolist(),
                    data_k=np.asarray(k).tolist(), data_chi=np.asarray(chi).tolist(),
                    r=dataset.model.r.tolist(), fit_mag=np.abs(dataset.model.chir).tolist(),
                    fit_re=dataset.model.chir.real.tolist(), fit_im=dataset.model.chir.imag.tolist(),
                    data_r=dataset.data.r.tolist(), data_mag=np.abs(dataset.data.chir).tolist(),
                    data_re=dataset.data.chir.real.tolist(), data_im=dataset.data.chir.imag.tolist()),
                report=feffit_report(result))


def parameter_failures(fit):
    bad = set()
    for name, p in fit['parameters'].items():
        if not p.get('vary', True):
            continue
        if p['value'] is None or p['stderr'] is None or p['at_bound']:
            bad.add(name)
        elif name.startswith('amp') and p['stderr'] > abs(p['value'])*.75:
            bad.add(name)
        elif name.startswith('dr') and p['stderr'] > .15:
            bad.add(name)
        elif name.startswith('sig2') and p['stderr'] > max(.006, abs(p['value'])):
            bad.add(name)
        elif name == 'de0' and p['stderr'] > 8:
            bad.add(name)
        for other, corr in p['correlations'].items():
            if corr is not None and abs(corr) > .98:
                bad.update((name, other))
    stats = fit['statistics']
    if not stats['converged'] or stats['rfactor'] is None:
        bad.update(name for name,p in fit['parameters'].items() if p.get('vary', True))
    elif stats['rfactor'] >= 1.:
        # A zero EXAFS model has R-factor 1 in the same residual space. This
        # model explains no signal, unlike an imperfect but informative fit.
        bad.update(name for name in fit['parameters'] if name.startswith(('amp','dr')))
    return bad


def checked_fit(fit, groups, k, chi, window, s02, assumptions, delta_r_limit=.25, deadline=None):
    from xraydb import atomic_symbol
    fit.update(fit_range=window, assumptions=list(assumptions), warnings=[], stability=[])
    bad = parameter_failures(fit)
    fixed = fit.get('fixed_parameters', {})
    alternate = -1 if fit.get('selected_start') != -1 else 1
    previous = [trial for trial in fit.get('initial_trials',[]) if trial.get('statistics',{}).get('converged')
                and trial.get('start') != fit.get('selected_start') and 'parameters' in trial]
    # Several starts may fail to find the good minimum. They remain exported,
    # but a much worse or unconverged minimum is not evidence that two agreeing
    # converged starts estimate different parameters.
    previous = sorted(previous,key=lambda trial:trial['statistics']['aicc'])
    for label, win, start in [('不同初值', window, alternate),
          ('缩小 k 区间', {**window, 'kmin':window['kmin']+.25, 'kmax':window['kmax']-.25}, 0)]:
        try:
            if not (label == '不同初值' and previous) and deadline is not None and time.monotonic() >= deadline:
                fit['budget_incomplete'] = True
                raise InputError('本候选的计算份额已用完，稳定性检查未完成；这不是化学模型被排除的证据。')
            repeat = previous[0] if label == '不同初值' and previous else \
                fit_once(groups, k, chi, win, s02, start, fixed, delta_r_limit)
            changes = {name:abs(p['value']-repeat['parameters'][name]['value'])
                       for name,p in fit['parameters'].items() if p.get('vary', True)}
            tolerance = lambda name: .2 if name.startswith('amp') else .025 if name.startswith('dr') else .003 if name.startswith('sig') else 3.
            unstable = [name for name,change in changes.items()
                        if change > max(3*(fit['parameters'][name]['stderr'] or 0), tolerance(name))]
            unstable.extend(parameter_failures(repeat))
            if label == '不同初值':
                for trial in previous[1:]:
                    if trial['statistics']['aicc'] <= fit['statistics']['aicc']+6:
                        unstable.extend(name for name,p in fit['parameters'].items() if p.get('vary',True)
                            and abs(p['value']-trial['parameters'][name]['value']) >
                                max(3*(p['stderr'] or 0),tolerance(name)))
            bad.update(unstable)
            fit['stability'].append(dict(test=label, status='unstable' if unstable else 'passed',
                changes=changes, parameters=repeat['parameters'],
                method='compare independent converged starts and all competitive minima' if label == '不同初值' else 'shrink k interval'))
        except Exception as exc:
            bad.update(name for name,p in fit['parameters'].items() if p.get('vary', True))
            fit['stability'].append(dict(test=label, status='unresolved', reason=str(exc)))
    stats = fit['statistics']
    stats['fraction_signal_explained'] = None if stats['rfactor'] is None else 1.-stats['rfactor']
    if stats['rfactor'] is None or stats['rfactor'] > .05:
        fit['warnings'].append('第一壳残差较大；只提供通过参数检查的粗略条件估计，不能作为精确结构或物相结论。')
    if not stats['converged'] or stats['rfactor'] is None:
        fit['warnings'].append('未确认收敛或残差不可计算；相关结构数值保持未确定。')
    elif stats['rfactor'] >= 1.:
        fit['warnings'].append('该模型的残差不优于零 EXAFS 模型，未解释当前窗口中的信号；保留诊断曲线，不推荐 N、R。')
    if stats['reduced_chi_square'] is None or stats['reduced_chi_square'] > 10:
        fit['warnings'].append('约化 χ² 较大或不可估计；统计误差不含模型与噪声系统误差。')
    pars = fit['parameters']
    if len(groups) == 2 and groups[0][0][1].geom[1][1] == groups[1][0][1].geom[1][1]:
        separation = groups[1][0][1].reff+pars['dr1']['value']-groups[0][0][1].reff-pars['dr0']['value']
        if separation <= np.pi/(2*(window['kmax']-window['kmin'])):
            bad.update(name for name,p in pars.items() if p.get('vary', True))
            fit['warnings'].append('分裂第一壳间距未超过 Δk 对应的径向分辨率。')
    def value(name, v, error):
        assumed = name in fixed
        status = 'assumed' if assumed else 'unresolved' if name in bad else 'quick_estimate'
        return dict(value=None if name in bad else finite(v), stderr=None if assumed or name in bad else finite(error),
                    estimate=None if assumed else finite(v), estimate_stderr=None if assumed else finite(error),
                    status=status, estimated=not assumed, assumption_range=None)
    fit['shells'] = []
    for i, paths in enumerate(groups):
        amp, dr, sigma = f'amp{i}', f'dr{i}', f'sig2{i}'
        n = sum(p.degen for _,p in paths)
        reff = sum(p.degen*p.reff for _,p in paths)/n
        p = paths[0][1]
        fit['shells'].append(dict(pair=f'{atomic_symbol(p.geom[0][1])}–{atomic_symbol(p.geom[1][1])}',
            N=value(amp, n*pars[amp]['value'], n*pars[amp]['stderr'] if pars[amp]['stderr'] is not None else None),
            R=value(dr, reff+pars[dr]['value'], pars[dr]['stderr']),
            sigma_squared=value(sigma, pars[sigma]['value'], pars[sigma]['stderr']),
            energy_shift=value('de0', pars['de0']['value'], pars['de0']['stderr'])))
    for shell in fit['shells']:
        shell['N']['conditional_on'] = ['S0_squared', 'reference_environment', 'local_model']+list(fixed)
    fit['unresolved_parameters'] = sorted(bad)
    fit['assumed_parameters'] = sorted(fixed)
    fit['nr_available'] = all(s['N']['value'] is not None and s['R']['value'] is not None for s in fit['shells'])
    if bad:
        fit['warnings'].append('到界、误差、相关性或稳定性未通过的参数保持未确定；estimate 字段仅作诊断。')
    return fit


def quick_model_fit(groups, k, chi, window, s02, assumptions, delta_r_limit=.25, deadline=None, initial_attempt=None):
    """Free four-parameter attempt, then a bounded explicit-assumption fallback."""
    trials, solutions = [], []
    # Reserve time for a reduced-parameter attempt rather than spending the
    # whole candidate share on free-parameter starts. The global budget is
    # deliberately soft; each candidate receives a minimum detailed share.
    free_deadline = None if deadline is None else time.monotonic()+.45*max(0.,deadline-time.monotonic())
    for start in (0, -1, 1):
        if start != 0 and free_deadline is not None and time.monotonic() >= free_deadline:
            trials.append(dict(start=start,status='budget_incomplete',reason='自由拟合份额已用完，保留时间检查显式约束。'))
            continue
        try:
            fit = initial_attempt if start == 0 and initial_attempt is not None and initial_attempt['statistics']['converged'] else \
                fit_once(groups, k, chi, window, s02, start, delta_r_limit=delta_r_limit)
            trials.append(dict(start=start, statistics=fit['statistics'], parameters=fit['parameters']))
            if fit['statistics']['rfactor'] is not None:
                solutions.append((fit, start))
        except Exception as exc:
            trials.append(dict(start=start, reason=str(exc)))
    free = None
    if solutions:
        free, start = min(solutions, key=lambda pair:(not pair[0]['statistics']['converged'], pair[0]['statistics']['rfactor']))
        free.update(initial_trials=trials, selected_start=start)
        free = checked_fit(free, groups, k, chi, window, s02, assumptions, delta_r_limit, free_deadline)
        free['constraint_strategy'] = 'all_parameters_estimated'
        if not free['unresolved_parameters']:
            free.update(unconstrained_attempts=trials, constrained_attempts=[], assumption_sensitivity=[])
            return free
    if deadline is not None and time.monotonic() >= deadline:
        if free:
            free['warnings'].append('本轮计算预算已用完，未执行约束敏感性检查。')
            free.update(budget_incomplete=True,unconstrained_attempts=trials,constrained_attempts=[],assumption_sensitivity=[])
            return free
        raise InputError('计算预算已用完，未完成第一壳条件估计。')
    sigma_fixed = lambda sigma:{f'sig2{i}':sigma for i in range(len(groups))}
    tiers = [
        ('fixed_disorder', [sigma_fixed(x) for x in (.006, .003, .009)]),
        ('fixed_energy_shift', [{'de0':x} for x in (0., -3., 3.)]),
        ('fixed_nuisance_first_shell', [dict(de0=e, **sigma_fixed(s))
            for s,e in [(.006,0.),(.003,0.),(.009,0.),(.006,-3.),(.006,3.)]])]
    # Fix one nuisance first where information allows. Each trial still uses
    # the same R window, and every fixed parameter is removed from nvarys.
    if free and 'de0' in free['unresolved_parameters'] and not any(
            name.startswith('sig2') for name in free['unresolved_parameters']):
        tiers[0], tiers[1] = tiers[1], tiers[0]
    attempts, constrained, chosen = [], [], None
    selected_presets = []
    for strategy, presets in tiers:
        tier_solutions = []
        for preset_index, fixed in enumerate(presets):
            if deadline is not None and time.monotonic() >= deadline:
                attempts.append(dict(strategy=strategy, fixed_parameters=fixed, reason='计算预算已用完。'))
                break
            try:
                candidate = fit_once(groups, k, chi, window, s02, 0, fixed, delta_r_limit)
                bad = parameter_failures(candidate)
                attempts.append(dict(strategy=strategy, fixed_parameters=fixed, statistics=candidate['statistics'],
                    parameters=candidate['parameters'], unresolved_parameters=sorted(bad)))
                if not bad:
                    tier_solutions.append(candidate)
                # A failed central one-nuisance fit cannot be rescued by
                # choosing the most favourable preset; proceed to next tier.
                if preset_index == 0 and bad:
                    break
                if preset_index == 0:
                    candidate.update(selected_start=0)
                    candidate = checked_fit(candidate, groups, k, chi, window, s02, assumptions, delta_r_limit, deadline)
                    if candidate['unresolved_parameters']:
                        attempts[-1]['stability'] = candidate['stability']
                        attempts[-1]['unresolved_parameters'] = candidate['unresolved_parameters']
                        tier_solutions = []
                        break
            except Exception as exc:
                attempts.append(dict(strategy=strategy, fixed_parameters=fixed, reason=str(exc)))
                if preset_index == 0:
                    break
        if tier_solutions:
            constrained = tier_solutions
            chosen = tier_solutions[0]
            selected_presets = presets
            chosen['constraint_strategy'] = strategy
            break
    if chosen is None:
        if free:
            free.update(unconstrained_attempts=trials, constrained_attempts=attempts, assumption_sensitivity=[])
            return free
        error = InputError('自由与显式固定假设下的第一壳拟合均未通过数值或信息量检查。')
        error.fit_attempts = dict(unconstrained_attempts=trials, constrained_attempts=attempts)
        raise error
    chosen.update(initial_trials=trials, selected_start=0)
    chosen['assumptions'] += [
        '快速第一壳：fixed_parameters 中的 σ² 和/或 ΔE₀ 使用 Larch vary=False；不是实验测得值。N 与 R 是这些假设下的估计。',
        '对固定量采用 σ²=0.003/0.006/0.009 Å² 和/或 ΔE₀=-3/0/3 eV 的单因素敏感性检查；范围不是置信区间，也不涵盖所有系统误差。']
    chosen.update(unconstrained_attempts=trials,
                  constrained_attempts=attempts, assumption_sensitivity=attempts)
    if free:
        chosen['unconstrained_diagnostic'] = {key:free[key] for key in ('parameters','statistics','unresolved_parameters','stability')}
        if free.get('budget_incomplete'):
            chosen['warnings'].append('自由参数的全部检查未在计算份额内完成；采用通过检查的显式约束结果，不据此断言自由参数在科学上不可估计。')
    for i, shell in enumerate(chosen['shells']):
        n = sum(p.degen for _,p in groups[i])
        reff = sum(p.degen*p.reff for _,p in groups[i])/n
        for quantity, name, scale, offset in [('N', f'amp{i}', n, 0), ('R', f'dr{i}', 1, reff)]:
            numbers = [f['parameters'][name]['value']*scale+offset for f in constrained]
            if len(numbers) >= 2:
                shell[quantity]['assumption_range'] = [float(min(numbers)), float(max(numbers))]
    chosen['warnings'].append('N、R 为依赖固定 σ²/ΔE₀、S₀² 和参考势环境的初筛估计；假设范围与统计标准误差分开报告。')
    if len(constrained) < len(selected_presets):
        chosen['warnings'].append('部分假设组合未通过数值检查；敏感性范围不完整。')
    return chosen


def restrict_assumption_ranges(fit, groups, definitions):
    """Exclude sensitivity trials outside the same explicit distance priors."""
    if not definitions or not fit.get('assumption_sensitivity'):
        return fit
    from xraydb import atomic_symbol
    strategy = fit.get('constraint_strategy')
    excluded = False
    for i,(shell,paths) in enumerate(zip(fit['shells'],groups)):
        scatterer = atomic_symbol(paths[0][1].geom[1][1])
        bounds = [d for d in definitions if d['scatterer']==scatterer]
        if not bounds:
            continue
        degeneracy = sum(p.degen for _,p in paths)
        reference = sum(p.degen*p.reff for _,p in paths)/degeneracy
        numbers = []
        for trial in fit['assumption_sensitivity']:
            if trial.get('strategy') != strategy or trial.get('unresolved_parameters') or 'parameters' not in trial:
                continue
            pars = trial['parameters']
            distance = reference+pars[f'dr{i}']['value']
            accepted = any(d['minimum_A'] <= distance <= d['maximum_A'] for d in bounds)
            trial.setdefault('distance_prior_checks',[]).append(dict(shell=i,distance_A=distance,accepted=accepted))
            if accepted:
                numbers.append((degeneracy*pars[f'amp{i}']['value'],distance))
            else:
                excluded = True
        for j,quantity in enumerate(('N','R')):
            shell[quantity]['assumption_range'] = [min(v[j] for v in numbers),max(v[j] for v in numbers)] if len(numbers)>=2 else None
    if excluded:
        fit['warnings'].append('部分固定假设备选的距离超出记录的物理参考范围，已从假设敏感性范围排除；检查结果保留在导出中。')
    return fit
