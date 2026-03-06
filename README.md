# SpamScan — Email Spam Classifier

A beginner-friendly ML project that classifies emails as **spam** or **ham** (legitimate).

## 📁 Project Structure

```
spam-classifier/
│
├── train_model.py      # ML model training script
├── app.py              # Flask backend API
├── model.pkl           # Saved trained model (generated after training)
├── vectorizer.pkl      # Saved TF-IDF vectorizer (generated after training)
├── requirements.txt    # Python dependencies
├── static/
│   └── index.html      # Frontend UI
└── README.md
```

## 🧠 How It Works

1. **TF-IDF Vectorization** — Converts email text into numerical feature vectors. Words that appear frequently in spam but rarely overall get higher weights.

2. **Multinomial Naive Bayes** — A probabilistic classifier that applies Bayes' theorem. It calculates the probability that a given email belongs to the "spam" or "ham" class.

3. **Bigrams** — The model uses both single words and word pairs (e.g., "free money", "click here") as features, improving accuracy.

## ⚙️ Setup & Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the model
```bash
python train_model.py
```
This creates `model.pkl` and `vectorizer.pkl`.

### Step 3: Start the Flask server
```bash
python app.py
``` 

### Step 4: Open the app
Go to **http://localhost:5000** in your browser.

## 🔌 API Reference

### POST `/predict`
```json
// Request
{ "email": "Congratulations! You've won a prize..." }

// Response
{
  "prediction": "spam",
  "confidence": 97.3,
  "spam_probability": 97.3,
  "ham_probability": 2.7
}
```

### GET `/health`
Returns server status.

## 📈 Improving the Model

To get better accuracy on real-world data:
- Use the **SMS Spam Collection** dataset from Kaggle (5,572 labeled messages)
- Try `LogisticRegression` or `SVC` as alternative classifiers
- Add preprocessing: lowercase, remove punctuation, lemmatization
- Use cross-validation for a more reliable accuracy estimate

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model  | Scikit-learn (MultinomialNB) |
| Features  | TF-IDF Vectorizer |
| Backend   | Flask (Python) |
| Frontend  | HTML / CSS / Vanilla JS |
