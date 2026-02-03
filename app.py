"""
Healthcare Disease Prediction System - Flask Application
"""

from flask import Flask, render_template, request, jsonify, send_file
import sqlite3
import joblib
import pickle
import re
from datetime import datetime
import pandas as pd
import os
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key = os.urandom(24)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database configuration
DATABASE = 'database/predictions.db'

# Model paths
MODEL_PATH = 'models/neural_network_model.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'

# =============================================================================
# DATABASE SETUP
# =============================================================================

def init_db():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symptoms TEXT NOT NULL,
            predicted_disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create model_performance table for tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_predictions INTEGER DEFAULT 0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

# =============================================================================
# NLP PREPROCESSING (Same as training)
# =============================================================================

# Initialize NLP tools
try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Remove medical keywords from stopwords
    medical_keywords = {'pain', 'fever', 'cough', 'rash', 'itch', 'swell', 'bleed', 
                       'burn', 'ache', 'sore', 'nausea', 'dizzy', 'tired', 'weak'}
    stop_words = stop_words - medical_keywords
except:
    print("⚠️ NLTK data not found. Downloading...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    medical_keywords = {'pain', 'fever', 'cough', 'rash', 'itch', 'swell', 'bleed', 
                       'burn', 'ache', 'sore', 'nausea', 'dizzy', 'tired', 'weak'}
    stop_words = stop_words - medical_keywords

def clean_text(text):
    """
    Clean and preprocess text data (same as training pipeline)
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens 
                     if word not in stop_words and len(word) > 2]
    
    # Join back to string
    return ' '.join(cleaned_tokens)

# =============================================================================
# LOAD ML MODELS
# =============================================================================

def load_models():
    """Load the trained ML model and vectorizer"""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Models loaded successfully!")
        return model, vectorizer
    except Exception as e:
        print(f"Error loading models: {e}")
        print("⚠️ Please ensure trained models are in the 'models/' directory")
        return None, None

# Load models at startup
model, vectorizer = load_models()

# =============================================================================
# DISEASE INFORMATION DATABASE
# =============================================================================

DISEASE_INFO = {
    'Psoriasis': {
        'risk_factors': ['Genetic predisposition', 'Stress', 'Skin injury', 'Infections'],
        'precautions': [
            'Keep skin moisturized',
            'Avoid harsh soaps and hot water',
            'Manage stress levels',
            'Avoid skin injuries'
        ],
        'recommendations': [
            'Use prescribed topical treatments',
            'Consider phototherapy if recommended',
            'Maintain healthy lifestyle',
            'Consult dermatologist regularly'
        ]
    },
    'Arthritis': {
        'risk_factors': ['Age', 'Family history', 'Previous joint injury', 'Obesity'],
        'precautions': [
            'Maintain healthy weight',
            'Exercise regularly',
            'Protect joints from injury',
            'Avoid repetitive stress'
        ],
        'recommendations': [
            'Low-impact exercises like swimming',
            'Physical therapy',
            'Anti-inflammatory diet',
            'Consult rheumatologist'
        ]
    },
    'Common Cold': {
        'risk_factors': ['Weak immune system', 'Close contact with infected persons', 'Seasonal changes'],
        'precautions': [
            'Wash hands frequently',
            'Avoid touching face',
            'Stay away from sick people',
            'Get adequate rest'
        ],
        'recommendations': [
            'Drink plenty of fluids',
            'Rest and sleep well',
            'Use humidifier',
            'Take over-the-counter medications if needed'
        ]
    },
    'Diabetes': {
        'risk_factors': ['Family history', 'Obesity', 'Sedentary lifestyle', 'Age over 45'],
        'precautions': [
            'Monitor blood sugar regularly',
            'Maintain healthy diet',
            'Exercise regularly',
            'Take medications as prescribed'
        ],
        'recommendations': [
            'Follow diabetic diet plan',
            'Regular physical activity',
            'Regular check-ups with endocrinologist',
            'Foot care and eye exams'
        ]
    },
    'Hypertension': {
        'risk_factors': ['High salt intake', 'Obesity', 'Stress', 'Family history'],
        'precautions': [
            'Reduce sodium intake',
            'Maintain healthy weight',
            'Limit alcohol consumption',
            'Manage stress'
        ],
        'recommendations': [
            'DASH diet (Dietary Approaches to Stop Hypertension)',
            'Regular blood pressure monitoring',
            'Regular exercise',
            'Consult cardiologist'
        ]
    },
    # Add default info for other diseases
    'default': {
        'risk_factors': ['Various factors depending on condition'],
        'precautions': [
            'Maintain good hygiene',
            'Eat balanced diet',
            'Exercise regularly',
            'Get adequate sleep'
        ],
        'recommendations': [
            'Consult healthcare professional',
            'Follow prescribed treatment',
            'Monitor symptoms',
            'Maintain healthy lifestyle'
        ]
    }
}

def get_disease_info(disease):
    """Get disease information or return default"""
    return DISEASE_INFO.get(disease, DISEASE_INFO['default'])

# =============================================================================
# INPUT VALIDATION FUNCTIONS
# =============================================================================

def validate_symptoms_input(text):
    """
    Validate user input to detect invalid/gibberish symptoms
    Returns: (is_valid, error_message)
    """
    # Check if empty
    if not text or not text.strip():
        return False, "Please enter your symptoms"
    
    # Check minimum length
    if len(text.strip()) < 10:
        return False, "Please provide more detailed symptoms (at least 10 characters)"
    
    # Check maximum length
    if len(text) > 2000:
        return False, "Symptoms description is too long (maximum 2000 characters)"
    
    # Remove special characters and numbers for validation
    text_for_validation = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Check if text has actual words after cleaning
    if len(text_for_validation.strip()) < 5:
        return False, "Please enter valid symptoms using words (not just numbers or special characters)"
    
    # Tokenize into words
    words = text_for_validation.split()
    
    # Check for gibberish - words that are too long without vowels
    gibberish_count = 0
    valid_words = 0
    
    for word in words:
        if len(word) > 2:  # Only check words longer than 2 chars
            # Check if word has vowels (basic gibberish detection)
            vowel_count = sum(1 for char in word if char in 'aeiou')
            
            # If word is long but has no vowels, likely gibberish
            if len(word) > 4 and vowel_count == 0:
                gibberish_count += 1
            # If word has reasonable vowel ratio, it's valid
            elif vowel_count > 0:
                valid_words += 1
    
    # If too many gibberish words, reject input
    if gibberish_count > 5 or (len(words) > 0 and valid_words < 2):
        return False, "Please enter valid symptoms using real words (e.g., 'headache', 'fever', 'cough')"
    
    # Check for excessive repetition (like "aaaaaaa" or "111111")
    # Find the most common character
    if len(text) > 0:
        char_counts = {}
        for char in text.lower():
            if char.isalnum():
                char_counts[char] = char_counts.get(char, 0) + 1
        
        if char_counts:
            max_char_count = max(char_counts.values())
            # If any character appears more than 50% of the time, likely spam
            if max_char_count > len(text) * 0.5:
                return False, "Please enter meaningful symptoms (avoid repetitive characters)"
    
    # Check if input is mostly numbers
    digit_count = sum(1 for char in text if char.isdigit())
    if digit_count > len(text) * 0.5:
        return False, "Please describe symptoms using words, not just numbers"
    
    # Check if input is mostly special characters
    special_char_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
    if special_char_count > len(text) * 0.5:
        return False, "Please use proper words to describe your symptoms"
    
    # All validation passed
    return True, None

def is_meaningful_prediction(cleaned_text):
    """
    Check if cleaned text has enough meaningful content for prediction
    Returns: (is_meaningful, error_message)
    """
    # After cleaning, check if we have enough content
    if not cleaned_text or len(cleaned_text.strip()) < 3:
        return False, "Unable to extract meaningful symptoms from your input. Please use medical terms like 'headache', 'fever', 'pain', etc."
    
    # Check if we have at least some words
    words = cleaned_text.split()
    if len(words) < 2:
        return False, "Please provide more detailed symptoms for accurate prediction"
    
    return True, None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def determine_risk_level(confidence):
    """Determine risk level based on confidence score"""
    if confidence >= 75:
        return 'High Risk'
    elif confidence >= 50:
        return 'Medium Risk'
    else:
        return 'Low Risk'

def get_risk_color(risk_level):
    """Get color code for risk level"""
    colors = {
        'High Risk': '#dc3545',      # Red
        'Medium Risk': '#ffc107',    # Yellow
        'Low Risk': '#28a745'        # Green
    }
    return colors.get(risk_level, '#6c757d')

def save_prediction(symptoms, disease, confidence, risk_level):
    """Save prediction to database"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (symptoms, predicted_disease, confidence, risk_level)
            VALUES (?, ?, ?, ?)
        ''', (symptoms, disease, confidence, risk_level))
        
        # Update model performance
        cursor.execute('''
            INSERT INTO model_performance (total_predictions)
            VALUES (1)
            ON CONFLICT(id) DO UPDATE SET
            total_predictions = total_predictions + 1,
            last_updated = CURRENT_TIMESTAMP
        ''')
        
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        
        return prediction_id
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return None

def get_prediction_history():
    """Retrieve all predictions from database"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, symptoms, predicted_disease, confidence, risk_level, timestamp
            FROM predictions
            ORDER BY timestamp DESC
        ''')
        
        predictions = cursor.fetchall()
        conn.close()
        
        return predictions
    except Exception as e:
        print(f"Error retrieving history: {e}")
        return []

def get_prediction_stats():
    """Get prediction statistics"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total = cursor.fetchone()[0]
        
        # Most common diseases
        cursor.execute('''
            SELECT predicted_disease, COUNT(*) as count
            FROM predictions
            GROUP BY predicted_disease
            ORDER BY count DESC
            LIMIT 5
        ''')
        top_diseases = cursor.fetchall()
        
        # Average confidence
        cursor.execute('SELECT AVG(confidence) FROM predictions')
        avg_confidence = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total': total,
            'top_diseases': top_diseases,
            'avg_confidence': round(avg_confidence, 2)
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {'total': 0, 'top_diseases': [], 'avg_confidence': 0}

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def landing():
    """Landing page"""
    return render_template('landing.html')

@app.route('/home')
def home():
    """Home page with prediction form"""
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle disease prediction"""
    try:
        # Check if models are loaded
        if model is None or vectorizer is None:
            return jsonify({
                'success': False,
                'error': 'ML models not loaded. Please check server configuration.'
            })
        
        # Get symptoms from request
        data = request.get_json()
        symptoms = data.get('symptoms', '').strip()
        
        # Validate input
        is_valid, error_msg = validate_symptoms_input(symptoms)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            })
        
        # Preprocess symptoms (same as training)
        cleaned_symptoms = clean_text(symptoms)
        
        # Check if cleaning resulted in meaningful text
        is_meaningful, meaning_error = is_meaningful_prediction(cleaned_symptoms)
        if not is_meaningful:
            return jsonify({
                'success': False,
                'error': meaning_error,
                'suggestion': 'Try describing specific symptoms like: "I have severe headache, high fever, and body aches"'
            })
        
        # Vectorize
        symptoms_vector = vectorizer.transform([cleaned_symptoms])
        
        # Make prediction
        prediction = model.predict(symptoms_vector)[0]
        
        # Get confidence
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(symptoms_vector)[0]
            confidence = round(max(probabilities) * 100, 2)
        else:
            confidence = 85.0  # Default confidence for models without predict_proba
        
        # Determine risk level
        risk_level = determine_risk_level(confidence)
        
        # Get disease information
        disease_info = get_disease_info(prediction)
        
        # Save to database
        prediction_id = save_prediction(symptoms, prediction, confidence, risk_level)
        
        # Prepare response
        response = {
            'success': True,
            'prediction_id': prediction_id,
            'disease': prediction,
            'confidence': confidence,
            'risk_level': risk_level,
            'risk_color': get_risk_color(risk_level),
            'precautions': disease_info['precautions'],
            'recommendations': disease_info['recommendations'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred during prediction. Please try again with different symptoms.',
            'suggestion': 'Make sure to describe your symptoms clearly using medical terms.'
        })

@app.route('/history')
def history():
    """View prediction history"""
    predictions = get_prediction_history()
    stats = get_prediction_stats()
    return render_template('history.html', predictions=predictions, stats=stats)

@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
def delete_prediction(prediction_id):
    """Delete a prediction from history"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM predictions WHERE id = ?', (prediction_id,))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_report/<int:prediction_id>')
def download_report(prediction_id):
    """Download individual prediction report"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symptoms, predicted_disease, confidence, risk_level, timestamp
            FROM predictions
            WHERE id = ?
        ''', (prediction_id,))
        
        prediction = cursor.fetchone()
        conn.close()
        
        if not prediction:
            return "Prediction not found", 404
        
        # Create Excel file
        df = pd.DataFrame([{
            'Symptoms': prediction[0],
            'Predicted Disease': prediction[1],
            'Confidence (%)': prediction[2],
            'Risk Level': prediction[3],
            'Timestamp': prediction[4]
        }])
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Prediction Report')
        
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'prediction_report_{prediction_id}.xlsx'
        )
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return f"Error: {str(e)}", 500

@app.route('/download_history')
def download_history():
    """Download complete prediction history as Excel"""
    try:
        predictions = get_prediction_history()
        
        df = pd.DataFrame(predictions, columns=[
            'ID', 'Symptoms', 'Predicted Disease', 'Confidence (%)', 'Risk Level', 'Timestamp'
        ])
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Prediction History')
        
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='prediction_history.xlsx'
        )
        
    except Exception as e:
        print(f"Error generating history: {e}")
        return f"Error: {str(e)}", 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    stats = get_prediction_stats()
    return jsonify(stats)

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Run the app
    print("\n" + "="*80)
    print("Healthcare Disease Prediction System")
    print("="*80)
    print("🚀 Starting Flask application...")
    print("Access the application at: http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)