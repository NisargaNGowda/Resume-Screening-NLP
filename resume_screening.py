import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("resume_dataset.csv")

# Input and Output
X = data["Resume_str"]
y = data["Category"]

# Convert text into numbers
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_tfidf = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Read resume from file
with open("sample_resume.txt", "r", encoding="utf-8") as file:
    resume_text = file.read()

# Convert resume to TF-IDF
resume_vector = vectorizer.transform([resume_text])

# Predict category
prediction = model.predict(resume_vector)

print("\n" + "=" * 40)
print("RESUME SCREENING RESULT")
print("=" * 40)

print("\nPredicted Category:")
print(prediction[0])

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")