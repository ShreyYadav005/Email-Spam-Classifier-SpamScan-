"""
Spam Classifier - Flask Backend API
Serves predictions from the trained Naive Bayes model
"""

from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__, static_folder='static', static_url_path='')

# -----------------------------------------------
# Load trained model and vectorizer
# -----------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

print("✅ Model and vectorizer loaded successfully")


# -----------------------------------------------
# Routes
# -----------------------------------------------

@app.route('/')
def index():
    """Serve the frontend"""
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if an email is spam or ham.
    
    Request body (JSON):
        { "email": "your email text here" }
    
    Response (JSON):
        {
            "prediction": "spam" | "ham",
            "confidence": 0.97,
            "spam_probability": 0.97,
            "ham_probability": 0.03
        }
    """
    data = request.get_json()

    if not data or 'email' not in data:
        return jsonify({'error': 'Please provide an email text in the request body'}), 400

    email_text = data['email'].strip()

    if not email_text:
        return jsonify({'error': 'Email text cannot be empty'}), 400

    # Transform text using the fitted vectorizer
    text_tfidf = vectorizer.transform([email_text])

    # Get prediction
    prediction = model.predict(text_tfidf)[0]

    # Get probabilities for both classes
    probabilities = model.predict_proba(text_tfidf)[0]
    classes = model.classes_  # ['ham', 'spam'] or ['spam', 'ham'] depending on order

    prob_dict = dict(zip(classes, probabilities))
    spam_prob = prob_dict.get('spam', 0)
    ham_prob = prob_dict.get('ham', 0)

    confidence = max(spam_prob, ham_prob)

    return jsonify({
        'prediction': prediction,
        'confidence': round(float(confidence) * 100, 1),
        'spam_probability': round(float(spam_prob) * 100, 1),
        'ham_probability': round(float(ham_prob) * 100, 1),
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model': 'Naive Bayes Spam Classifier'})


# -----------------------------------------------
# Run the server
# -----------------------------------------------
if __name__ == '__main__':
    print("\n🚀 Starting Spam Classifier API...")
    print("   Frontend: http://localhost:5000")
    print("   API:      http://localhost:5000/predict")
    print("   Health:   http://localhost:5000/health\n")
    app.run(debug=True, port=5000)
