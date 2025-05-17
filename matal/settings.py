import os
from pathlib import Path

PROJ_CODE = 'AT'

BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
PROJ_DIR = BASE_DIR

CACHE_DIR = PROJ_DIR / 'cache'
DATA_DIR = PROJ_DIR / 'data'
MODEL_DIR = PROJ_DIR / 'model'

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
