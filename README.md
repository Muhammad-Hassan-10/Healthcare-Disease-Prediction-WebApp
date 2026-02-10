# 🏥 Healthcare Disease Prediction System

A complete machine learning system with web application for predicting diseases based on symptom descriptions. Built with Python, Flask, and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Models & Results](#models--results)
- [Technology Stack](#technology-stack)
- [Disclaimer](#disclaimer)

---

## 🎯 Overview

This system uses **Natural Language Processing (NLP)** and **Machine Learning** to predict diseases from text-based symptom descriptions. It includes:

- **ML Training Pipeline**: Train and compare 7 different ML models
- **Web Application**: User-friendly Flask web app for predictions
- **Database System**: Track prediction history and statistics
- **Downloadable Reports**: Export predictions as Excel files

**Key Highlights:**
- 📊 **Dataset**: 1,200 symptom descriptions across 24 diseases
- 🎯 **Accuracy**: 95-98% prediction accuracy
- 🚀 **Production-Ready**: Saved models with complete deployment pipeline
- 🌐 **Web Interface**: Responsive Bootstrap 5 design

---

## ✨ Features

### Machine Learning
- ✅ Train 7 different ML algorithms
- ✅ Advanced NLP preprocessing (tokenization, lemmatization, TF-IDF)
- ✅ Comprehensive evaluation metrics
- ✅ Model comparison and selection
- ✅ Saved models for deployment

### Web Application
- ✅ AI-powered disease predictions
- ✅ Confidence scoring (percentage)
- ✅ Risk assessment (High/Medium/Low)
- ✅ Disease-specific precautions & recommendations
- ✅ Prediction history tracking
- ✅ Statistics dashboard
- ✅ Excel report downloads
- ✅ Fully responsive design

---

## 📁 Project Structure

```
Healthcare-Disease-Prediction-WebApp/
│
├── database
│   └── predictions.db             # Auto-generated
│
├── dataset/
│   └── Symptom2Disease.csv        # Raw dataset (1,200 samples)
│
├── models/                         # Saved models (generated after training)
│   └── (5+ models)
│
│   ├── tfidf_vectorizer.pkl       # TF-IDF vectorizer
│   ├── label_info.pkl             # Label encoding info
│   ├── metadata.pkl               # Model metadata
│   └── model_comparison.csv       # Performance comparison
│
├── notebook/
│   └── disease_prediction_analysis.ipynb  # Complete ML pipeline
│
├── static/
│   └── css/
│       └── style.css              # Web app styling
│
├── templates/                     # HTML templates
│   ├── landing.html
│   ├── home.html
│   ├── history.html
│   └── about.html
│
├── app.py                         # Main Flask application
├── .gitignore
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Open the projet folder**
   
   ```bash
   git clone https://github.com/hassancodebase/Healthcare-Disease-Prediction-WebApp
   cd Healthcare-Disease-Prediction-WebApp
   ```

2. **Create virtual environment (recommended)**
   
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt

# Or you can try these commands instead:
python -m pip install --upgrade pip setuptools wheel
pip install Flask==3.0.0 Werkzeug==3.0.1 scikit-learn>=1.3.0 xgboost>=1.7.5 joblib>=1.3.0 numpy>=1.24.0 pandas>=2.0.0 matplotlib seaborn wordcloud tqdm nltk>=3.8.1 openpyxl>=3.1.0 python-dateutil>=2.8.2 jupyter notebook ipykernel
```

### Step 4: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## 💻 Usage

### Option 1: Use Pre-trained Models (Quick Start)

If models are already trained and saved in `models/` folder:

```bash
# Start the web application
python app.py

# Open browser and go to:
http://localhost:5000
```

### Option 2: Train Models First

If you want to train models yourself:

1. **Train the models** (see [Model Training](#model-training) section)
2. **Run the web app** (see [Web Application](#web-application) section)

---

## 🤖 Model Training

### Running the Jupyter Notebook

1. **Start Jupyter Notebook**
   
   ```bash
   jupyter notebook
   ```

2. **Open the analysis notebook**
   
   Navigate to `notebook/disease_prediction_analysis.ipynb` and run all cells

3. **Training Process**
   
   The notebook will:
   - Load and explore the dataset (1,200 samples, 24 diseases)
   - Preprocess text data (cleaning, tokenization, lemmatization)
   - Create TF-IDF features (1000 features, unigrams + bigrams)
   - Train 7 different ML models
   - Evaluate and compare performance
   - Save all models to `models/` directory

### Models Trained

1. **Logistic Regression** - Fast, interpretable baseline
2. **Linear SVM** - Excellent for high-dimensional text data
3. **XGBoost** - Gradient boosting for complex patterns
4. **Random Forest** - Robust ensemble method
5. **Multinomial Naive Bayes** - Fast probabilistic classifier
6. **Gradient Boosting** - Sequential ensemble learning
7. **Neural Network (MLP)** - Captures non-linear patterns

### Expected Output

After training, the `models/` folder will contain:
- 7 trained model files (.pkl)
- TF-IDF vectorizer
- Label encoding information
- Model metadata
- Performance comparison CSV

---

## 🌐 Web Application

### Starting the Application

```bash
# Make sure you're in the project directory and venv is activated
python app.py
```

You should see:

```
================================================================================
🏥 Healthcare Disease Prediction System
================================================================================
🚀 Starting Flask application...
📍 Access the application at: http://localhost:5000
================================================================================
```

### Using the System

1. **Landing Page**
   - Welcome screen with feature highlights
   - Click "Get Started"

2. **Home Page**
   - Enter symptoms in natural language
   - Example: "I have severe headache and high fever for 3 days"
   - Click "Predict Disease"
   - View results with:
     - Predicted disease
     - Confidence score (circular meter)
     - Risk level (High/Medium/Low)
     - Precautions
     - Recommendations

3. **History Page**
   - View all past predictions
   - Statistics dashboard
   - Download individual reports (Excel)
   - Download complete history (Excel)
   - Delete predictions

4. **About Page**
   - System information
   - How it works
   - ML model details
   - Medical disclaimer

### API Endpoints

- `GET /` - Landing page
- `GET /home` - Main prediction page
- `POST /predict` - Make prediction (JSON API)
- `GET /history` - View prediction history
- `GET /about` - About page
- `GET /download_report/<id>` - Download single report
- `GET /download_history` - Download complete history
- `POST /delete_prediction/<id>` - Delete a prediction

---

## 📊 Models & Results

### Dataset Information

- **Total Samples**: 1,200
- **Number of Diseases**: 24
- **Samples per Disease**: 50 (perfectly balanced)
- **Train-Test Split**: 80-20 (960 train, 240 test)
- **TF-IDF Features**: 1,000 features

### Supported Diseases

1. Acne
2. Arthritis
3. Bronchial Asthma
4. Cervical Spondylosis
5. Chicken Pox
6. Common Cold
7. Dengue
8. Diabetes
9. Dimorphic Hemorrhoids
10. Fungal Infection
11. GERD
12. Hypertension
13. Impetigo
14. Jaundice
15. Malaria
16. Migraine
17. Peptic Ulcer Disease
18. Pneumonia
19. Psoriasis
20. Typhoid
21. Urinary Tract Infection
22. Varicose Veins
23. Allergy
24. Drug Reaction

### Model Performance

| Model               | Accuracy | Speed      |
|---------------------|----------|------------|
| Neural Network      | 96.67%   | ⚡⚡        |
| Linear SVM          | 96.67%   | ⚡⚡⚡       |
| Random Forest       | 95.00%   | ⚡⚡⚡       |
| Logistic Regression | 94.58%   | ⚡⚡⚡       |
| Naive Bayes         | 94.17%   | ⚡⚡⚡       |
| XGBoost             | 88.75%   | ⚡⚡        |
| Gradient Boosting   | 82.92%   | ⚡         |

**Best Models**: Neural Network (MLP) and Linear SVM (96.67% accuracy)

### Risk Level Criteria

- **High Risk**: Confidence ≥ 75%
- **Medium Risk**: 50% ≤ Confidence < 75%
- **Low Risk**: Confidence < 50%

---

## 🛠️ Technology Stack

### Backend
- **Flask 3.0.0** - Web framework
- **Python 3.8+** - Programming language
- **Scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting
- **NLTK** - Natural language processing
- **SQLite** - Database
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### Frontend
- **HTML5** - Markup
- **CSS3** - Styling
- **Bootstrap 5.3** - UI framework
- **JavaScript** - Interactivity
- **Font Awesome** - Icons
- **Google Fonts** - Typography (Poppins)

### Visualization
- **Matplotlib** - Plotting
- **Seaborn** - Statistical visualizations
- **WordCloud** - Text visualization

### Utilities
- **Joblib** - Model serialization
- **OpenPyXL** - Excel file generation

---

## 🔧 Configuration

### Change Port

Edit `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change port number
```

### Database Location

Edit `app.py`:

```python
DATABASE = 'database/predictions.db'  # Change path if needed
```

---

## 🐛 Troubleshooting

### Issue: "Models not loaded"
**Solution**: Train the models first using the Jupyter notebook

### Issue: "NLTK data not found"
**Solution**: Run the NLTK download command from installation steps

### Issue: "Module not found"
**Solution**: `pip install -r requirements.txt`

### Issue: "Port already in use"
**Solution**: Change the port in `app.py` or stop the process using that port

---

## 📚 Project Workflow

### Complete Development Flow

```
1. Data Collection
   ↓
2. Exploratory Data Analysis (Jupyter Notebook)
   ↓
3. Text Preprocessing (NLP Pipeline)
   ↓
4. Feature Engineering (TF-IDF)
   ↓
5. Model Training (7 algorithms)
   ↓
6. Model Evaluation & Comparison
   ↓
7. Model Selection & Saving
   ↓
8. Web Application Development (Flask)
   ↓
9. Frontend Design (Bootstrap)
   ↓
10. Database Integration (SQLite)
   ↓
11. Testing & Deployment
```

---

## 🎯 Future Enhancements

### Short-term
- User authentication system
- More disease categories (expand to 50+)
- Voice input for symptoms
- Multi-language support

### Medium-term
- Image-based diagnosis (skin conditions)
- AI chatbot for interactive diagnosis
- Mobile application (Android/iOS)
- Doctor recommendation system

### Long-term
- Integration with hospital systems
- Telemedicine capabilities
- Wearable device integration
- Clinical validation and certification

---

## ⚠️ Medical Disclaimer

**IMPORTANT NOTICE:**

This application is for **educational and research purposes only**. It is NOT:
- A substitute for professional medical diagnosis
- A tool for emergency medical situations
- Approved for clinical use
- A replacement for healthcare professionals

**Always consult a qualified healthcare provider** for proper medical diagnosis and treatment. In case of medical emergency, contact emergency services immediately.

---

## 📝 License

This project is for educational purposes. Please ensure you have appropriate permissions if using for commercial purposes.

---

## 🙏 Acknowledgments

- Dataset: [Symptom2Disease on Kaggle](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease)
- Built with open-source ML libraries
- Inspired by medical NLP research
- Auhor: **Muhammad Hassan**

---

## ✅ Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] NLTK data downloaded
- [ ] Models trained (or pre-trained models available)
- [ ] Application starts without errors
- [ ] Can access http://localhost:5000
- [ ] Predictions working correctly
- [ ] Reports downloadable

---

*Built with ❤️ using Flask, Bootstrap, and Machine Learning*

---

**Last Updated**: 09 Feburary 2024  
**Version**: 1.0.0  
**Status**: Production Ready
