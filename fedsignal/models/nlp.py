import json

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from fedsignal.config import MODEL_NAME, device, SENTENCES_PATH, SKIP_PATTERNS_US


class SentDataset(Dataset):
    def __init__(self, texts, scores, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        self.labels = torch.tensor(scores, dtype=torch.float32)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def train_nlp_model(sentences_path=None):
    if sentences_path is None:
        sentences_path = SENTENCES_PATH

    with open(sentences_path, encoding='utf-8') as f:
        training_data = json.load(f)

    texts  = [s['text']  for s in training_data]
    scores = [s['score'] for s in training_data]
    print(f'Training sentences: {len(texts)}')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model = model.to(device)

    dataset = SentDataset(texts, scores, tokenizer)
    training_args = TrainingArguments(
        output_dir='./fed_sentence_model',
        num_train_epochs=30,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        logging_steps=50,
        save_strategy='no',
        use_cpu=True,
    )
    Trainer(model=model, args=training_args, train_dataset=dataset).train()
    model.eval()
    print('DistilBERT training complete')
    return model, tokenizer


def score_statement(text, model, tokenizer, skip_patterns=None):
    if skip_patterns is None:
        skip_patterns = SKIP_PATTERNS_US

    sent_scores = []
    for sent in text.split('.'):
        sent = sent.strip()
        if len(sent) < 20 or any(p in sent for p in skip_patterns):
            continue
        inputs = tokenizer(sent, return_tensors='pt', truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        sent_scores.append(out.logits.cpu().numpy()[0][0])

    if not sent_scores:
        return 0.5, 0.0
    return float(np.mean(sent_scores)), float(np.std(sent_scores))


def score_fomc_df(fomc_df, model, tokenizer):
    print('\nScoring FOMC statements...')
    import pandas as pd
    for date, row in fomc_df.iterrows():
        if pd.notna(row.get('statement')):
            mean_s, disp_s = score_statement(row['statement'], model, tokenizer)
            fomc_df.loc[date, 'sent_level']      = mean_s
            fomc_df.loc[date, 'sent_dispersion'] = disp_s

    fomc_df['sent_level']         = fomc_df['sent_level'].astype(float)
    fomc_df['sent_dispersion']    = fomc_df['sent_dispersion'].astype(float)
    fomc_df['sent_level_demeaned'] = fomc_df['sent_level'] - fomc_df['sent_level'].mean()

    print(f'NLP scores range: [{fomc_df["sent_level"].min():.3f}, {fomc_df["sent_level"].max():.3f}]')
    print(f'NLP dispersion range: [{fomc_df["sent_dispersion"].min():.3f}, {fomc_df["sent_dispersion"].max():.3f}]')
    return fomc_df
