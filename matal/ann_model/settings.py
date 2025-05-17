import numpy as np

X_COLS = ['LAP', 'MMT', 'CMC', 'CNF', 'SLK',
          'AGR', 'ALG', 'CAR', 'CHS', 'PEC', 'PUL', 'STA', 'GEL', 'GLU', 'ZIN',
          'GLY', 'FFA', 'LAC', 'LEV', 'PHA', 'SRB', 'SUA', 'XYL',
          ]

TARGET_Y_COLS = {
    'grade': ['Detachability', 'FlatnessUni', 'Feasibility'],
    'optical': ['TransVis', 'TransIR', 'TransUV'],
    'tensile': ['TensileStrength', 'TensileStrain', 'TensileModulusLog10', 'TensileSED'],
    'fire': ['FireRR'],
}

TARGET_Y_SCALES = {
    'grade': np.array([1.0, 1.0, 1.0]),
    'optical': np.array([100.0, 100.0, 100.0]),
    'tensile': np.array([150, 150, 3, 20]),
    'fire': np.array([1.0]),
}

META_COLS = ['SampleID', 'DataSource', 'BatchNum']
