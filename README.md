# 📰 Fake News Detection using Machine Learning

> A Natural Language Processing (NLP) pipeline that classifies news articles as **Real** or **Fake** using four machine learning models — Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest — with TF-IDF text vectorization.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Highlights](#-project-highlights)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline Overview](#-pipeline-overview)
- [Text Preprocessing](#-text-preprocessing)
- [Models Used](#-models-used)
- [Results](#-results)
- [Manual Testing](#-manual-testing)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technologies Used](#-technologies-used)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧩 Problem Statement

The rapid spread of misinformation and fake news on social media and online platforms poses a serious threat to public discourse, democracy, and society. Manually identifying fake news is time-consuming and inconsistent. This project builds an **automated machine learning classifier** that can detect whether a given news article is real or fake based purely on its text content.

---

## ✨ Project Highlights

| Metric | Value |
|--------|-------|
| Total articles | **44,898** (23,481 Fake + 21,417 True) |
| Features used | Article text (TF-IDF vectors) |
| Models trained | **4** (LR · DT · GBC · RFC) |
| Vectorizer | TF-IDF (Term Frequency–Inverse Document Frequency) |
| Train / Test split | **75% / 25%** |
| Manual testing | ✅ Real-time user input classification |

---

## 📦 Dataset

The project uses two CSV files containing news articles:

| File | Label | Articles | Description |
|------|-------|----------|-------------|
| `Fake.csv` | `0` — Fake News | 23,481 | Fabricated or misleading news articles |
| `True.csv` | `1` — Real News | 21,417 | Verified real news articles |

### Columns in Raw Data

| Column | Used | Description |
|--------|------|-------------|
| `title` | ❌ Dropped | Headline of the article |
| `text` | ✅ Used | Full body text of the article |
| `subject` | ❌ Dropped | Topic category |
| `date` | ❌ Dropped | Publication date |
| `class` | ✅ Target | 0 = Fake, 1 = Real |

> **Note:** `title`, `subject`, and `date` columns are dropped — only the article `text` is used as the feature for classification, making the model generalisable beyond topic categories or time periods.

---

## 📁 Project Structure

```
fake-news-detection/
│
├── 📓 Fake_News_Detection_using_machine_learning.ipynb   ← Main notebook
│
├── data/
│   ├── Fake.csv                 ← Fake news articles (23,481 rows)
│   └── True.csv                 ← Real news articles (21,417 rows)
│
├── requirements.txt
└── README.md
```

---

## 🔄 Pipeline Overview

```
Fake.csv  ──┐
            ├──► Merge & Label ──► Shuffle ──► Drop Unused Cols
True.csv  ──┘         │
                       ▼
              Text Preprocessing (wordopt)
              ├── Lowercase
              ├── Remove URLs & links
              ├── Remove special characters
              ├── Remove punctuation
              └── Remove numbers
                       │
                       ▼
              TF-IDF Vectorization
              (fit on train, transform on test)
                       │
                       ▼
           ┌───────────────────────────┐
           │  4 Classifiers Trained    │
           ├───────────────────────────┤
           │  Logistic Regression      │
           │  Decision Tree            │
           │  Gradient Boosting        │
           │  Random Forest            │
           └───────────────────────────┘
                       │
                       ▼
          Evaluation: Accuracy + Classification Report
                       │
                       ▼
          Manual Testing (real-time user input)
```

---

## 🧹 Text Preprocessing

All article text is cleaned using a custom `wordopt()` function before training:

```python
def wordopt(text):
    text = text.lower()                              # Lowercase
    text = re.sub('\[.*?\]', '', text)               # Remove bracketed content
    text = re.sub("\\W", " ", text)                  # Remove non-word characters
    text = re.sub('https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub('<.*?>+', '', text)                # Remove HTML tags
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub('\w*\d\w*', '', text)              # Remove words containing digits
    return text
```

This produces clean, normalised text ready for TF-IDF vectorization.

---

## 🤖 Models Used

### 1. Logistic Regression
A fast, interpretable linear classifier well-suited for high-dimensional text data. Serves as the baseline model.

### 2. Decision Tree Classifier
A tree-based model that learns explicit if-else decision rules from TF-IDF features. Highly interpretable.

### 3. Gradient Boosting Classifier
An ensemble of weak learners trained sequentially, where each corrects the errors of the previous. Strong generalisation capability.

### 4. Random Forest Classifier
An ensemble of decision trees trained on random feature subsets. Reduces overfitting through averaging and typically achieves the best accuracy on text tasks.

---

## 📊 Results

All four models are evaluated on a held-out 25% test set using **accuracy score** and a full **classification report** (Precision, Recall, F1-score).

### Model Accuracy Comparison

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~98–99% |
| Decision Tree | ~99% |
| Gradient Boosting | ~99% |
| **Random Forest** | **~99%** |

> Exact scores depend on the random train/test split. Random Forest and Gradient Boosting typically achieve the highest F1 scores across both classes.

### Classification Report (Example — Logistic Regression)

```
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      5870
           1       0.99      0.99      0.99      5355

    accuracy                           0.99     11225
   macro avg       0.99      0.99      0.99     11225
weighted avg       0.99      0.99      0.99     11225
```

- **Class 0** = Fake News  
- **Class 1** = Real News

---

## 🧪 Manual Testing

The notebook includes a `manual_testing()` function that lets you type any news article text and get predictions from **all four models simultaneously**:

```python
def manual_testing(news):
    # Preprocess input
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    # Predict with all models
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)

    print("LR Prediction: {}".format(output_lable(pred_LR[0])))
    print("DT Prediction: {}".format(output_lable(pred_DT[0])))
    print("GBC Prediction: {}".format(output_lable(pred_GB[0])))
    print("RFC Prediction: {}".format(output_lable(pred_RF[0])))
```

**Example output:**
```
LR Prediction:  Not A Fake News
DT Prediction:  Not A Fake News
GBC Prediction: Not A Fake News
RFC Prediction: Not A Fake News
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

### 4. Add the dataset files

Place `Fake.csv` and `True.csv` in the project root directory (or a `data/` subfolder and update the paths in the notebook accordingly).

> The dataset is available publicly on [Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## 🚀 Usage

### Run the Jupyter Notebook

```bash
jupyter notebook Fake_News_Detection_using_machine_learning.ipynb
```

Run all cells in order: **Kernel → Restart & Run All**

At the end of the notebook, you will be prompted to enter any news article text manually and see predictions from all four models in real time.

### Notebook Steps

| Step | Description |
|------|-------------|
| 1 | Import libraries |
| 2 | Load `Fake.csv` and `True.csv`, assign class labels |
| 3 | Reserve 10 samples per class for manual testing |
| 4 | Merge, shuffle, and clean the dataset |
| 5 | Apply `wordopt()` text preprocessing |
| 6 | Split into train (75%) and test (25%) sets |
| 7 | Apply TF-IDF vectorization |
| 8 | Train Logistic Regression → evaluate |
| 9 | Train Decision Tree → evaluate |
| 10 | Train Gradient Boosting → evaluate |
| 11 | Train Random Forest → evaluate |
| 12 | Manual testing via user input |

---

## 🛠️ Technologies Used

| Category | Tool |
|----------|------|
| **Language** | Python 3.8+ |
| **Data manipulation** | pandas, NumPy |
| **Text vectorization** | scikit-learn `TfidfVectorizer` |
| **Machine learning** | scikit-learn (LR, DT, GBC, RFC) |
| **Text cleaning** | Python `re`, `string` |
| **Visualisation** | Matplotlib, Seaborn |
| **Notebook** | Jupyter |

---

## 🔮 Future Work

- [ ] **Deep learning** — fine-tune BERT or RoBERTa for higher accuracy
- [ ] **Include title + text** — concatenate headline and body for richer features
- [ ] **Explainability** — SHAP values to identify which words drive predictions
- [ ] **Web application** — Flask or Streamlit app for browser-based news checking
- [ ] **Multi-class classification** — detect satire, clickbait, and propaganda separately
- [ ] **Real-time detection** — integrate with a news API for live article classification
- [ ] **Model saving** — serialize trained models with `pickle` or `joblib` for deployment
- [ ] **Multilingual support** — extend to detect fake news in languages other than English

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add: your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---
