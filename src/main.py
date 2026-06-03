#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Broad Twitter Corpus NER homework solution.

Usage (from project root):

    python src/main.py --data-dir "D:/資料/大學/大三/下/自然語言處理與文件探勘/HW3" --use-opennlp False

This will train on eng.train, validate on eng.testa, and evaluate on eng.testb.
"""

import os
import argparse
from typing import List, Tuple

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score


def read_conll(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Read CoNLL-style file with at least token and NER tag in the last column.

    Returns:
        sentences_tokens: list of sentences, each a list of tokens
        sentences_tags:   list of sentences, each a list of ner tags
    """
    sentences_tokens: List[List[str]] = []
    sentences_tags: List[List[str]] = []
    tokens: List[str] = []
    tags: List[str] = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences_tokens.append(tokens)
                    sentences_tags.append(tags)
                    tokens, tags = [], []
                continue
            if line.startswith('-DOCSTART-'):
                # treat document boundary as sentence boundary
                if tokens:
                    sentences_tokens.append(tokens)
                    sentences_tags.append(tags)
                    tokens, tags = [], []
                continue

            parts = line.split()
            token = parts[0]
            ner_tag = parts[-1]
            tokens.append(token)
            tags.append(ner_tag)

    if tokens:
        sentences_tokens.append(tokens)
        sentences_tags.append(tags)

    return sentences_tokens, sentences_tags


def word2features(sent: List[str], i: int) -> dict:
    """Extract features for a single token in a sentence."""
    word = sent[i]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }

    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent: List[str]) -> List[dict]:
    return [word2features(sent, i) for i in range(len(sent))]


def train_and_eval(data_dir: str, use_opennlp: bool = False, c1: float = 0.1, c2: float = 0.1) -> None:
    """Train CRF model on train, tune on dev, and evaluate on test.

    Args:
        data_dir: directory containing eng.train/eng.testa/eng.testb (and optionally *.openNLP)
        use_opennlp: if True, use *.openNLP files instead of the default ones
        c1, c2: L1 and L2 regularization coefficients for CRF
    """
    suffix = '.openNLP' if use_opennlp else ''

    train_path = os.path.join(data_dir, f'eng.train{suffix}')
    dev_path = os.path.join(data_dir, f'eng.testa{suffix}')
    test_path = os.path.join(data_dir, f'eng.testb{suffix}')

    print(f"Loading data from: {train_path}, {dev_path}, {test_path}")

    X_train_tokens, y_train = read_conll(train_path)
    X_dev_tokens, y_dev = read_conll(dev_path)
    X_test_tokens, y_test = read_conll(test_path)

    X_train = [sent2features(s) for s in X_train_tokens]
    X_dev = [sent2features(s) for s in X_dev_tokens]
    X_test = [sent2features(s) for s in X_test_tokens]

    # Train CRF model
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=c1,
        c2=c2,
        max_iterations=100,
        all_possible_transitions=True,
    )

    print("Training CRF model...")
    crf.fit(X_train, y_train)

    # Evaluate on dev set
    y_dev_pred = crf.predict(X_dev)
    print("Dev set metrics:")
    print(classification_report(y_dev, y_dev_pred, digits=4))

    # Evaluate on test set
    y_test_pred = crf.predict(X_test)
    print("Test set metrics:")
    print(classification_report(y_test, y_test_pred, digits=4))

    # Also print overall scores
    p = precision_score(y_test, y_test_pred)
    r = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    acc = accuracy_score(y_test, y_test_pred)

    print("Overall test Precision: {:.4f}".format(p))
    print("Overall test Recall:    {:.4f}".format(r))
    print("Overall test F1:        {:.4f}".format(f1))
    print("Overall test Accuracy:  {:.4f}".format(acc))

    # Save model
    import pickle
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'crf_ner_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(crf, f)
    print(f"Saved model to {model_path}")

    # Save predictions for test set in CoNLL-like format
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    pred_path = os.path.join(results_dir, 'eng.testb.pred')
    with open(pred_path, 'w', encoding='utf-8') as out:
        for tokens, gold_tags, pred_tags in zip(X_test_tokens, y_test, y_test_pred):
            for tok, g, p_ in zip(tokens, gold_tags, pred_tags):
                out.write(f"{tok} {g} {p_}\n")
            out.write("\n")
    print(f"Saved test predictions to {pred_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Broad Twitter Corpus NER with CRF')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing eng.* files')
    parser.add_argument('--use-opennlp', action='store_true', help='Use *.openNLP files instead of default')
    parser.add_argument('--c1', type=float, default=0.1, help='L1 regularization')
    parser.add_argument('--c2', type=float, default=0.1, help='L2 regularization')
    args = parser.parse_args()

    train_and_eval(args.data_dir, use_opennlp=args.use_opennlp, c1=args.c1, c2=args.c2)
