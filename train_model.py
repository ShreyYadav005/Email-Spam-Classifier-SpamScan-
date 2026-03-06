import pickle
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the Kaggle dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only the relevant columns and rename them
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

print(f"Total samples: {len(df)}")
print(df['label'].value_counts())

# Features and labels
texts  = df['text'].tolist()
labels = df['label'].tolist()  # 'spam' or 'ham'

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=10000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# Train Model
model = MultinomialNB(alpha=0.1)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred   = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ model.pkl and vectorizer.pkl saved!")