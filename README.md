# 🏦 Customer Churn Detection ML Project

A full-stack machine learning project to predict and explain customer churn for a bank, featuring a Streamlit web app and multiple ML models.

---

## 🚀 Project Overview

This project helps banks **predict which customers are likely to leave (churn)** using advanced machine learning models. It provides:
- Accurate churn probability for each customer
- Explanations for predictions using AI (Groq LLM)
- Actionable business recommendations to retain customers

---

## 📁 Project Structure

```
churn-detectioin-ML-project/
│
├── banking_custoer_churn_detection.ipynb   # Jupyter notebook: data analysis, model training, evaluation, export
├── main.py                                # Streamlit app: user interface, prediction, AI explanations
├── churn.csv                              # Customer dataset (10,000 rows)
├── models/                                # Trained ML models (.pkl files)
│     ├── dt_model.pkl
│     ├── knn_model.pkl
│     ├── nb_model.pkl
│     ├── rf_model.pkl
│     ├── svm_model.pkl
│     ├── xgb_model.pkl
│     ├── xgboost_featureEng_model.pkl
│     ├── xgboost_SMOTE_model.pkl
│     └── voting_clf.pkl
├── requirements.txt                       # Python dependencies
├── .env                                   # API keys (not tracked in git)
└── ...
```

---

## 🧑‍💻 How It Works

### 1. **Data & Features**
- **Dataset:** 10,000 bank customers, 14 features (age, geography, balance, products, etc.)
- **Target:** `Exited` (1 = churned, 0 = stayed)

### 2. **Model Training (`.ipynb` notebook)**
- Data cleaning, feature engineering, scaling
- Trains 8+ ML models: Decision Tree, KNN, Naive Bayes, Random Forest, SVM, XGBoost, Voting Classifier, etc.
- Handles class imbalance with SMOTE
- Evaluates models (accuracy, recall, F1-score)
- Saves best models as `.pkl` files

### 3. **Web App (`main.py`)**
- **Select a customer** or adjust features to simulate scenarios
- **Predicts churn probability** using all models
- **Visualizes risk** (gauge, bar chart, color-coded risk)
- **Explains predictions** using Groq LLM (Llama 3)
- **Gives business recommendations** for retention

---

## 🏁 Getting Started

### 1. **Install dependencies**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. **Set up API Key**
- Create a `.env` file:
  ```
  GROQ_API_KEY=your_groq_api_key_here
  ```

### 3. **Run the app**
```bash
streamlit run main.py
```
- Open the link in your browser (usually http://localhost:8501)

---

## 🧠 Machine Learning Highlights

- **Ensemble Learning:** Combines multiple models for robust predictions
- **Class Imbalance:** Uses SMOTE to improve detection of rare churn cases
- **Feature Importance:** Identifies top factors driving churn
- **Explainability:** AI-generated, business-friendly explanations for every prediction

---
# ML-project-customer-churn-detection
