import random
from ..settings import PROJ_CODE

_EXTENDED_EXT_2_HEX = {
    'G': '0',
    'H': '1',
    'J': '2',
    'K': '3',
    'M': '4',
    'N': '5',
    'P': '6',
    'Q': '7',
    'R': '8',
    'S': '9',
    'T': 'A',
    'U': 'B',
    'V': 'C',
    'W': 'F',
    'X': 'E',
    'Y': 'F',
}
_EXTENDED_HEX_2_EXT = {v: k for k, v in _EXTENDED_EXT_2_HEX.items()}
_EXTENDED_EXT_LIST = ''.join(_EXTENDED_EXT_2_HEX.keys())


def int2hex(i: int, length=2, extended=False):
    assert isinstance(i, int) and 0 <= i
    if i >= 16 ** length:
        if extended and (i < 2 * (16 ** length)):
            ei = i - 16 ** length
            eh = hex(ei)[2:].rjust(length, '0')[:length].upper()
            eh = _EXTENDED_HEX_2_EXT[eh[0]] + eh[1:]
            return eh
        else:
            raise ValueError('Number overflow!')
    else:
        return hex(i)[2:].rjust(length, '0')[:length].upper()


def int2str(i: int, length=2):
    assert isinstance(i, int) and 0 <= i
    return str(i).rjust(length, '0')[:length].upper()


def hex2int(h: str, extended=False):
    if extended and h[0] in _EXTENDED_EXT_LIST:
        eh = _EXTENDED_EXT_2_HEX[h[0]] + h[1:]
        ei = int(eh, base=16)
        offset = 16 ** len(h)
        return ei + offset
    else:
        return int(h, base=16)


SOURCE_TYPES = {
    # Project AG
    'CNF': 'Cellulose Nanofiber',
    'MMT': 'Montmorillonite',
    'GEL': 'Gelatin',
    'GLT': 'Gelatin',
    'GLY': 'Glycerol',

    # Project AH
    'MXN': 'MXene',
    'CAL': 'Calcium chloride',

    # Project AI
    'TMS': 'TMS',
    'CNC': 'Cellulose nanocrystal',
    'ALC': 'Ethanol',
    'LAP': 'Laponite',

    # Project AK
    'PLA': 'Polylactic acid',

    # Project SC
    'CNT': 'SWCNT',
    'SPP': 'SuperP',
    'BND': 'Binder',

    # Shared
    'WTR': 'DI Water',
    'AIR': 'Air gap for OT2',
    'UNK': 'Unknown',
}


SID_TYPE_LUT = {
    'B': 'Batch',
    'S': 'Sample',
    'O': 'OT2Protocol',
    # 'P': 'Photo',
    # 'T': 'TensileMeasurement',
    # 'L': 'TransparencyMeasurement',
    # 'D': 'FireRetardantMeasurement',
    # 'F': 'FilmMeasurement',
    # 'A': 'AeroGelMeasurement',
}
SID_TYPE_RVS_LUT = {v: k for k, v in SID_TYPE_LUT.items()}

BATCH_USAGE_LUT = {
    'A': 'arbitrary_input',
    'B': 'reserved_2',
    'C': 'reserved_3',
    'D': 'design_boundary_opt',
    'E': 'reserved_5',
    'F': 'design_boundary',
    'L': 'link_only',
    'M': 'miscellaneous',
    'X': 'link_only_deprecated',
    'T': 'test_data',
    'R': 'revision_exp',
    'S': 'transfer_learning_filtration'
}
BATCH_USAGE_LUT.update({str(i): 'optimization' for i in range(0, 10)})
BATCH_USAGE_RVS_LUT = {v: k for k, v in BATCH_USAGE_LUT.items()}
del (BATCH_USAGE_RVS_LUT['optimization'])


def get_sid_info(sid: str):
    assert len(sid) == 8
    p1, p2, t1, sep, c1, c2, c3, c4 = sid
    assert sep == '-'

    info = {}
    if p1.isupper():
        info['item_type'] = 'Source'
        info['src_type'] = p1 + p2 + t1
        info['src_sub_type'] = c3
        info['year'] = 2020 + hex2int(c1)
        info['month'] = '0123456789ABC'.index(c2)
        info['idx_str'] = '0' + c4
        info['idx_int'] = hex2int(info['idx_str'])
    else:
        info['item_type'] = SID_TYPE_LUT[t1]
        info['project'] = (p1 + p2).upper()

    if info['item_type'] == 'Batch':
        assert c3 == c4 == '0'
        info['usage'] = BATCH_USAGE_LUT[c1]
        info['idx_str'] = c1 + c2
        if c1 in '0123456789':
            info['idx_int'] = hex2int(info['idx_str'])
        else:
            info['idx_int'] = hex2int(c2)

    if info['item_type'] == 'Sample':
        info['usage'] = BATCH_USAGE_LUT[c1]
        info['batch_sid'] = p1 + p2 + SID_TYPE_RVS_LUT['Batch'] + sep + c1 + c2 + '00'
        info['idx_str'] = c3 + c4
        info['idx_int'] = hex2int(info['idx_str'])
    return info


def get_sid(item_type, batch_sid: str = None, i: int = None, usage: str = None, proj_code: str = PROJ_CODE,
            src_type: str = None, year: int = None, month: int = None):
    try:
        type_code = SID_TYPE_RVS_LUT[item_type]
    except KeyError:
        type_code = 'X'
        if item_type != 'Source':
            raise ValueError(f'Unknown item_type {item_type}')

    if isinstance(proj_code, str):
        proj_code = proj_code[:1].lower() + proj_code[1:].upper()

    if item_type == 'Source':
        assert year >= 2020, 'must be made on or later than year 2020'
        assert 1 <= month <= 12, 'month be between 1 and 12'
        year_code = int2hex(year - 2020, length=1)
        month_code = '0123456789ABC'[month]
        return f'{src_type}-{year_code}{month_code}{int2hex(i)}'
    elif item_type in ['Sample', 'OT2Protocol']:
        proj_code = batch_sid[:2]
        batch_id = batch_sid[4:6]
        return f'{proj_code}{type_code}-{batch_id}{int2hex(i)}'
    elif item_type == 'Batch':
        if usage == 'optimization':
            return f'{proj_code}{type_code}-{int2hex(i)}00'
        else:
            batch_type_code = BATCH_USAGE_RVS_LUT[usage]
            return f'{proj_code}{type_code}-{batch_type_code}{int2hex(i, length=1)}00'
    else:
        proj_code = proj_code or batch_sid[:2]
        if i is None:
            i = random.randint(1, 65535)
        return f'{proj_code}{type_code}-{int2hex(i, length=4)}'

