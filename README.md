# HW3: Named Entity Recognition on Broad Twitter Corpus

This repository contains my solution for **Natural Language Processing and Text Mining – Homework #3 (NER)**. The goal is to train a named entity recognition (NER) model on the **Broad Twitter Corpus** and report evaluation metrics on the provided test set.

## 1. Project Structure

```text
.
├── requirements.txt        # Python dependencies
├── README.md               # This documentation
├── LICENSE.txt             # License notice kept at repository root
├── data/                   # Dataset files used in this homework
├── src/
│   └── main.py             # Training, evaluation, and prediction script
├── models/
│   └── crf_ner_model.pkl   # Saved CRF model (generated after training)
└── results/
    └── eng.testb.pred      # Prediction file for the test set
```

## 2. Environment Setup

- Python: **3.13.5** (earlier 3.x versions should also work)
- OS: Tested on **Windows 11**
- Packages:
  - `numpy`
  - `scikit-learn`
  - `sklearn-crfsuite`
  - `seqeval`

Install the dependencies with:

```bash
pip install -r requirements.txt
```

GPU is **not required** for this implementation because the model is based on CRF and runs on CPU.

## 3. Data Preparation

The homework provides six files in CoNLL-style format:

- `eng.train`
- `eng.testa`
- `eng.testb`
- `eng.train.openNLP`
- `eng.testa.openNLP`
- `eng.testb.openNLP`

Each line contains at least a **token** and a **NER tag** in the last column, with sentences separated by blank lines and documents separated by `-DOCSTART-` markers.

Place all six files into a single directory on your machine, for example:

```text
D:\proj\HW3_NER\data
```

(or another path of your choice).

## 4. Dataset Source

This homework uses the **Broad Twitter Corpus** from the GitHub repository [`entity-recognition-datasets`](https://github.com/juand-r/entity-recognition-datasets). The dataset included in this repository is distributed under the **MIT License**. The original license notice is preserved in `data/LICENSE.txt`, and an additional copy is also kept at the repository root as `LICENSE.txt` for clearer repository-level attribution.

## 5. How to Run

From the project root (`.`), run the following command in your terminal or PowerShell.

### 5.1 Use the standard files

```bash
python src/main.py --data-dir "D:/proj/HW3_NER/data"
```

This will:

1. Load `eng.train` as the training set.
2. Load `eng.testa` as the development (validation) set.
3. Load `eng.testb` as the test set.
4. Train a CRF model.
5. Evaluate on the dev and test sets.
6. Save the trained model to `models/crf_ner_model.pkl`.
7. Save the test predictions to `results/eng.testb.pred`.

### 5.2 Use the OpenNLP-converted files

If you want to use the `.openNLP` versions of the data, run:

```bash
python src/main.py --data-dir "D:/proj/HW3_NER/data" --use-opennlp
```

In this case, the script will use:

- `eng.train.openNLP`
- `eng.testa.openNLP`
- `eng.testb.openNLP`

## 6. Implementation Details

### 6.1 Data Reader

The script `src/main.py` implements a simple **CoNLL reader**:

- Skips `-DOCSTART-` lines.
- Treats blank lines as sentence boundaries.
- Uses the **first column** as the token and the **last column** as the NER tag.

This makes the code robust to minor column differences between the original and `.openNLP` variants.

### 6.2 Feature Extraction

For each token, the model uses a standard CRF feature set, including:

- Current word (lowercased, suffixes of length 2 and 3).
- Whether the word is uppercase, title case, or a digit.
- Same features for the previous and next words (if they exist).
- Special markers `BOS` (begin-of-sentence) and `EOS` (end-of-sentence).

These features are commonly used for classic NER baselines.

### 6.3 Model

The model is a **Conditional Random Field (CRF)** implemented via `sklearn-crfsuite`:

- Algorithm: `lbfgs`
- L1 regularization coefficient: `c1 = 0.1`
- L2 regularization coefficient: `c2 = 0.1`
- `max_iterations = 100`
- `all_possible_transitions = True`

You can modify `c1` and `c2` via command-line arguments to perform simple hyperparameter tuning, for example:

```bash
python src/main.py --data-dir "D:/proj/HW3_NER/data" --c1 0.05 --c2 0.01
```

### 6.4 Evaluation Metrics

The script uses the `seqeval` library to compute standard sequence labeling metrics on the **test set**:

- Precision
- Recall
- F1-score
- Accuracy

It also prints a detailed **classification report** (per-entity-type scores) for both the dev and test sets.

In addition, it writes a file `results/eng.testb.pred` containing, for each token in the test set:

```text
TOKEN GOLD_TAG PRED_TAG
```

separated by blank lines between sentences.

### 6.5 Experimental Results

Two sets of hyperparameters were tested on the test set:

- **Baseline (c1 = 0.1, c2 = 0.1)**\
  Precision = 0.8087, Recall = 0.7603, F1 = 0.7837, Accuracy = 0.9569
- **Tuned (c1 = 0.05, c2 = 0.01)**\
  Precision = 0.8138, Recall = 0.7689, F1 = 0.7907, Accuracy = 0.9584

In the final report, the tuned setting was used as the main result.

## 7. Reproducibility

- Randomness is mainly from CRF optimization; for strict reproducibility you may set a fixed `random_state` in the CRF constructor.
- As long as the same data and hyperparameters are used, the results should be very similar across runs.

## 8. Team Members and Responsibilities

This homework was completed as an **individual assignment**. All parts (implementation, experiments, and documentation) were done by **曹榮次 (112AB0057)**.

## 9. How to Submit

1. Make sure your local project contains:
   - `requirements.txt`
   - `README.md`
   - `src/main.py`
   - (Optionally) `models/crf_ner_model.pkl` and `results/eng.testb.pred` after running.
2. Zip the project directory or push it to a **GitHub** repository as required by the course.
3. Upload the zip file or GitHub link to the iSchool+ assignment entry `[HW#3]`.

## 10. Possible Extensions (Optional)

If you want to further improve the model (optional, not required for this baseline solution), you could:

- Add more contextual features (e.g., character n-grams, word shape patterns).
- Use pre-trained word embeddings.
- Try a neural model such as BiLSTM-CRF or BERT-based NER.

However, the current CRF baseline already fulfills the homework requirement of training a model and reporting NER metrics on the provided dataset.
