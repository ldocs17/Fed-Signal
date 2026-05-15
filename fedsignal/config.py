import os
import torch

FRED_KEY = 'FRED_API'
MODEL_NAME = 'distilbert-base-uncased'
HORIZON = 60
MIN_TRAIN = 20
FORCE_REFRESH = False

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
CACHE_DIR = os.path.join(PROJECT_ROOT, 'data')
FRED_CACHE = os.path.join(CACHE_DIR, 'fred_data.pkl')
STATEMENTS_CACHE = os.path.join(CACHE_DIR, 'fomc_statements.json')
SENTENCES_PATH = os.path.join(_HERE, 'data', 'sentences_us.json')

device = torch.device('cpu')

SKIP_PATTERNS_US = [
    'Voting for', 'Voting against', 'Vice Chair', 'Chair;',
    'media inquiries', 'email', 'Implementation Note',
    'For release', 'press@', 'call 202',
    'Brainard', 'Bullard', 'Bowman', 'Harker', 'Mester',
    'Kashkari', 'George', 'Kaplan', 'Clarida', 'Waller',
    'Jefferson', 'Cook', 'Hammack', 'Logan', 'Paulson',
    'Miran', 'Barr', 'Williams', 'Powell',
]
