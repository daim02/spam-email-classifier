# 📧 Spam Email Classifier

A machine learning-based spam detection system that classifies email subject lines as **spam** or **ham** (not spam) using natural language processing (NLP), TF-IDF vectorization, and a Random Forest classifier.

---

## 🧠 Features

- Text preprocessing (stopword removal, stemming, punctuation stripping)
- TF-IDF feature extraction
- Random Forest model training
- Evaluation metrics with confusion matrix and ROC curve
- CLI for interactive testing of custom subject lines

---

## 🚀 Getting Started

### 1. Clone the repository:
```bash
git clone https://github.com/daim02/spam-text-classifier.git
cd spam-text-classifier
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the classifier:
Make sure the dataset is in ```./data/spam_ham_dataset.csv```, then run:
```bash
python src/spam_text_classifier.py
```
You'll see a menu with options to view model stats or test custom subject lines.

---

## 📌 Dataset Info
- Source: Public dataset of labeled email subject lines
- Classes: ```ham``` (not spam), ```spam```

---

## 🛠️ Built With
- Python
- scikit-learn
- NLTK
- pandas, NumPy
- matplotlib, seaborn
