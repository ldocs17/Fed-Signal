import json
import os
import torch
import pytest
from unittest.mock import MagicMock, patch

from fedsignal.models.nlp import SentDataset, score_statement
from fedsignal.config import SENTENCES_PATH, SKIP_PATTERNS_US


def make_tokenizer(n=2):
    tok = MagicMock()
    tok.return_value = {
        'input_ids':      torch.zeros(n, 10, dtype=torch.long),
        'attention_mask': torch.ones(n,  10, dtype=torch.long),
    }
    # Also support single-sentence calls used inside score_statement
    tok.side_effect = lambda text, **kw: {
        'input_ids':      torch.zeros(1, 10, dtype=torch.long),
        'attention_mask': torch.ones(1,  10, dtype=torch.long),
    }
    return tok


def make_model(logit=0.6):
    model = MagicMock()
    model.return_value.logits = torch.tensor([[logit]])
    return model


# ── SentDataset ──────────────────────────────────────────────────────────────

def test_sentdataset_length():
    tok = MagicMock()
    tok.return_value = {
        'input_ids':      torch.zeros(3, 10, dtype=torch.long),
        'attention_mask': torch.ones(3,  10, dtype=torch.long),
    }
    ds = SentDataset(['a', 'b', 'c'], [0.5, 0.3, 0.8], tok)
    assert len(ds) == 3


def test_sentdataset_labels():
    tok = MagicMock()
    tok.return_value = {
        'input_ids':      torch.zeros(2, 10, dtype=torch.long),
        'attention_mask': torch.ones(2,  10, dtype=torch.long),
    }
    ds = SentDataset(['hello', 'world'], [0.5, 0.9], tok)
    assert abs(ds[0]['labels'].item() - 0.5) < 1e-6
    assert abs(ds[1]['labels'].item() - 0.9) < 1e-6


def test_sentdataset_item_has_input_ids():
    tok = MagicMock()
    tok.return_value = {
        'input_ids':      torch.zeros(1, 10, dtype=torch.long),
        'attention_mask': torch.ones(1,  10, dtype=torch.long),
    }
    ds = SentDataset(['test sentence'], [0.5], tok)
    assert 'input_ids' in ds[0]


# ── score_statement ───────────────────────────────────────────────────────────

def test_score_statement_short_sentence_skipped():
    model = make_model(0.7)
    tok   = make_tokenizer()
    # All sentences under 20 chars → empty → returns default
    mean, std = score_statement('Hi. Ok. Sure.', model, tok, [])
    assert mean == 0.5
    assert std  == 0.0
    model.assert_not_called()


def test_score_statement_returns_floats():
    model = make_model(0.7)
    tok   = make_tokenizer()
    mean, std = score_statement(
        'The Committee decided to raise rates due to elevated inflation.',
        model, tok, []
    )
    assert isinstance(mean, float)
    assert isinstance(std,  float)


def test_score_statement_skip_pattern_filters():
    model = make_model(0.7)
    tok   = make_tokenizer()
    # 'Powell' is in SKIP_PATTERNS_US; second sentence is long enough and clean
    text = 'Powell said rates are fine. Inflation remains elevated above target.'
    mean, std = score_statement(text, model, tok, SKIP_PATTERNS_US)
    # Only the second sentence should be scored
    assert model.call_count == 1


# ── sentences_us.json ─────────────────────────────────────────────────────────

def test_sentences_json_exists():
    assert os.path.exists(SENTENCES_PATH), f'Not found: {SENTENCES_PATH}'


def test_sentences_json_has_88_entries():
    with open(SENTENCES_PATH, encoding='utf-8') as f:
        data = json.load(f)
    assert len(data) == 88


def test_sentences_json_schema():
    with open(SENTENCES_PATH, encoding='utf-8') as f:
        data = json.load(f)
    for i, entry in enumerate(data):
        assert 'text'  in entry, f'Entry {i} missing "text"'
        assert 'score' in entry, f'Entry {i} missing "score"'
        assert isinstance(entry['text'],  str),   f'Entry {i} text not a string'
        assert isinstance(entry['score'], float), f'Entry {i} score not a float'
        assert 0.0 <= entry['score'] <= 1.0,      f'Entry {i} score out of range'
