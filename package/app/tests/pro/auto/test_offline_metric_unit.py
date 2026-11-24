from types import SimpleNamespace
from xasweb.pro.delivery import _readme


def test_fit_card_labels_missing_units_and_preserves_the_raw_record():
    metrics = [
        {'label':'Re 未加权 RMS','value':0.25,'unit':'not_declared','basis':'stored'},
        {'label':'Im 未加权 RMS','value':0.5,'unit':'eV','basis':'stored'},
    ]
    record = {'formats':[], 'data_files':[], 'sources':[], 'insight':{'metrics':metrics}, 'family':'F07',
              'id':'F07-1','filename':'fit.png','spec_file':'fit.json','title':'R fit'}
    page = _readme(SimpleNamespace(title='ZnSe'), {'state':'complete'}, [record], [])
    if isinstance(page, bytes):
        page = page.decode('utf-8')
    table = page.split('<table>', 1)[1].split('</table>', 1)[0]
    assert '0.25 （单位未记录）' in table and '0.5 eV' in table
    assert 'not_declared' not in table and 'not_declared' in page
    assert metrics[0]['unit'] == 'not_declared'
