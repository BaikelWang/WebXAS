from xasweb.pro.figures.presentation import brief_insight


def test_pca_and_mcr_only_explain_their_own_method():
    details = dict(components=2, window_eV=[7115,7200], relative_error=.01, stability=.2)
    pca = brief_insight({}, 'F13', {**details, 'quantity':'Explained variance','retained_variance':.95})
    assert pca['metrics'][0]['value'] == 95 and pca['metrics'][0]['unit']=='%'
    assert not pca['notices']
    mcr = brief_insight({}, 'F13', {**details,'quantity':'MCR coefficients'})
    assert mcr['metrics'][0]['value']==.01 and mcr['notices']
