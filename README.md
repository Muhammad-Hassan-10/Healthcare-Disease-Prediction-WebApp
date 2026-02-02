# 🏥 Healthcare Disease Prediction System

A comprehensive machine learning system for predicting diseases based on textual symptom descriptions. This project implements and compares multiple ML algorithms to achieve high-accuracy disease classification.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)

## 🎯 Project Overview

This system uses Natural Language Processing (NLP) and Machine Learning to predict diseases from symptom descriptions. It analyzes text-based symptom inputs and classifies them into 24 different disease categories.

**Key Highlights:**

- **Dataset**: 1,200 symptom descriptions across 24 diseases
- **Approach**: Multi-class text classification using TF-IDF + ML algorithms
- **Best Model**: Achieves 95-98% accuracy on test data
- **Production-Ready**: Saved models with complete pipeline for deployment

## ✨ Features

- **Comprehensive EDA**: In-depth exploratory data analysis with visualizations
- **Text Preprocessing**: Advanced NLP techniques (tokenization, lemmatization, stopword removal)
- **Multiple Models**: 7 different ML algorithms trained and compared
- **Feature Engineering**: TF-IDF vectorization with n-grams
- **Model Evaluation**: Detailed metrics (Accuracy, Precision, Recall, F1-Score, Cross-validation)
- **Visual Analytics**: Confusion matrices, comparison charts, word clouds
- **Model Persistence**: All models saved for production use
- **Prediction Pipeline**: Ready-to-use inference code

## 📁 Project Structure

```
Healthcare-Disease-Prediction-System/
│
├── dataset/
│   └── Symptom2Disease.csv          # Raw dataset (1,200 samples)
│
├── notebook/
│   └── disease_prediction_analysis.ipynb  # Complete analysis notebook
│
├── models/                           # Saved models (generated after training)
│   ├── logistic_regression_model.pkl
│   ├── linear_svm_model.pkl
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── neural_network_model.pkl
│   ├── tfidf_vectorizer.pkl         # TF-IDF vectorizer
│   ├── label_info.pkl               # Label encoding info
│   ├── metadata.pkl                 # Model metadata
│   └── model_comparison.csv         # Performance comparison
│
└── README.md                         # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup Instructions

1. **Open the projet folder**
   
   ```bash
   git clone https://github.com/Muhammad-Hassan-10/Disease-Prediction-Model-Training.git
   cd Healthcare-Disease-Prediction-System
   ```

2. **Create virtual environment (recommended)**
   
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost nltk jupyter notebook ipykernel joblib wordcloud tqdm
   ```

4. **Download NLTK data**
   
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt_tab')"
   ```

## 💻 Usage

### Running the Jupyter Notebook

1. **Start Jupyter Notebook**
   
   ```bash
   jupyter notebook
   ```

2. **Open the analysis notebook**
   
   - Navigate to `notebook/disease_prediction_analysis.ipynb`
   - Run all cells sequentially

3. **Training Process**
   
   - The notebook will automatically:
     - Load and explore the dataset
     - Preprocess text data
     - Train 7 different models
     - Evaluate and compare performance
     - Save all models to `models/` directory

### Making Predictions

```python
import joblib
import pickle

# Load saved models
best_model = joblib.load('models/best_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
label_info = pickle.load(open('models/label_info.pkl', 'rb'))

# Preprocess and predict
def predict_disease(symptom_text):
    # Clean text (use the clean_text function from notebook)
    cleaned_text = clean_text(symptom_text)

    # Vectorize
    text_vector = vectorizer.transform([cleaned_text])

    # Predict
    prediction = best_model.predict(text_vector)[0]
    confidence = max(best_model.predict_proba(text_vector)[0]) * 100

    return prediction, confidence

# Example
symptom = "I have severe headache and fever with body aches"
disease, conf = predict_disease(symptom)
print(f"Predicted Disease: {disease} (Confidence: {conf:.2f}%)")
```

## 🤖 Models Implemented

### 1. **Logistic Regression**

- **Best for**: Text classification, high-dimensional data
- **Accuracy**: 94.58%
- **Speed**: Very Fast
- **Interpretability**: High

### 2. **Linear SVM**

- **Best for**: High-dimensional sparse features
- **Accuracy**: 96.67%
- **Speed**: Very Fast
- **Interpretability**: Medium

### 3. **XGBoost**

- **Best for**: Complex patterns, feature importance
- **Accuracy**: 88.75%
- **Speed**: Medium
- **Interpretability**: Medium

### 4. **Random Forest**

- **Best for**: Robust baseline, feature importance
- **Accuracy**: 95%
- **Speed**: Fast
- **Interpretability**: High

### 5. **Multinomial Naive Bayes**

- **Best for**: Fast baseline, text classification
- **Accuracy**: 94.17%
- **Speed**: Very Fast
- **Interpretability**: High

### 6. **Gradient Boosting**

- **Best for**: Sequential learning, complex patterns
- **Accuracy**: 82.92%
- **Speed**: Slow
- **Interpretability**: Medium

### 7. **Neural Network (MLP)**

- **Best for**: Non-linear patterns
- **Accuracy**: 96.67%
- **Speed**: Medium
- **Interpretability**: High

## 📊 Results

### Dataset Information

- **Total Samples**: 1,200
- **Number of Diseases**: 24
- **Samples per Disease**: 50 (perfectly balanced)
- **Train-Test Split**: 80-20 (960 train, 240 test)
- **TF-IDF Features**: 1,000 features with unigrams & bigrams

### Model Performance (Expected)

| Rank | Model               | Accuracy | Precision | Recall | F1-Score | Training Speed |
| ---- | ------------------- | -------- | --------- | ------ | -------- | -------------- |
| 🥇 1 | Neural Network      | 96.67%   | 96.81%    | 96.67% | 96.66%   | ⚡⚡             |
| 🥈 2 | Linear SVM          | 96.67%   | 96.95%    | 96.67% | 96.61%   | ⚡⚡⚡            |
| 🥉 3 | Random Forest       | 95.00%   | 95.39%    | 95.00% | 94.92%   | ⚡⚡⚡            |
| 4    | Logistic Regression | 94.58%   | 95.38%    | 94.58% | 94.44%   | ⚡⚡⚡            |
| 5    | Naive Bayes         | 94.17%   | 94.82%    | 94.17% | 94.01%   | ⚡⚡⚡            |
| 6    | XGBoost             | 88.75%   | 90.01%    | 88.75% | 88.79%   | ⚡⚡             |
| 7    | Gradient Boosting   | 82.92%   | 87.05%    | 82.92% | 83.93%   | ⚡              |

### Key Findings

- ✅ **Neural Network (MLP)** and **Linear SVM** achieve best performance
- ✅ Text-based features work excellently with linear models
- ✅ Balanced dataset ensures fair evaluation
- ✅ High accuracy indicates clear symptom-disease relationships
- ✅ Cross-validation confirms model robustness

## 🛠️ Technologies Used

### Core Libraries

- **Python 3.8+**
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting

### NLP & Text Processing

- **NLTK**: Natural language toolkit
- **TF-IDF**: Feature extraction

### Visualization

- **Matplotlib**: Static plots
- **Seaborn**: Statistical visualizations
- **WordCloud**: Text visualization

### Model Persistence

- **Joblib**: Model serialization
- **Pickle**: Object serialization

## 🔮 Future Enhancements

### Short-term

- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Ensemble voting classifier for higher accuracy
- [ ] Add more sophisticated text preprocessing
- [ ] Implement SHAP for model interpretability

### Medium-term

- [ ] REST API development (Flask/FastAPI)
- [ ] Web interface for user interaction
- [ ] Real-time prediction system
- [ ] Model monitoring and logging

### Long-term

- [ ] Deep learning models (BERT, BioBERT)
- [ ] Multi-language support
- [ ] Integration with medical knowledge bases
- [ ] Mobile application development
- [ ] Continuous learning pipeline

## 📈 Model Deployment Recommendations

### Production Checklist

1. **Model Selection**: Use MLP or Linear SVM (best performance + speed)
2. **Confidence Threshold**: Implement minimum confidence score for predictions
3. **Monitoring**: Track prediction accuracy and model drift
4. **Retraining**: Schedule regular model updates with new data
5. **Validation**: Medical expert review before production deployment
6. **Scaling**: Use batch prediction for large volumes
7. **Caching**: Cache vectorizer and model in memory

### API Deployment

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load models at startup
model = joblib.load('models/neural_network_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptom_text = data['symptoms']

    # Preprocess and predict
    cleaned_text = clean_text(symptom_text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]
    confidence = max(model.predict_proba(text_vector)[0]) * 100

    return jsonify({
        'disease': prediction,
        'confidence': f"{confidence:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=False)
```

## 📝 Notes

### Important Considerations

- This system is for **educational and research purposes** only
- **Not a substitute** for professional medical diagnosis
- Always consult healthcare professionals for medical advice
- Predictions should be validated by medical experts
- Consider ethical implications of healthcare AI systems

### Dataset Limitations

- Limited to 24 disease categories
- 50 samples per disease (relatively small)
- Text-based symptoms only (no numeric/clinical data)
- English language only

## 🙏 Acknowledgments

- Dataset source: https://www.kaggle.com/datasets/niyarrbarman/symptom2disease
- Inspired by medical NLP research
- Built with open-source ML libraries

---

*Last Updated: 30 January 2026*


