import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("review_data.csv")

# Split data
X = df["Review"]
y = df["Sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
print("Accuracy:", model.score(X_test_vec, y_test))

# Test on new input
while True:
    msg = input("Enter a review (or 'q' to quit): ")
    if msg.lower() == 'q':
        break
    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    print(f"'{msg}' → {pred}")
