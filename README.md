# Task-3
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Load dataset
df = pd.read_csv('amazon_reviews.csv', nrows=10000)  # Load first 10k rows for speed
print(df.head())

# Clean data
df = df[['Text', 'Score']]  # Keep only review text and rating
df['Sentiment'] = df['Score'].apply(lambda x: 'positive' if x > 3 else 'negative')
print("\nSentiment Distribution:")
print(df['Sentiment'].value_counts())

# Visualize
plt.figure(figsize=(8,5))
df['Sentiment'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title("Positive vs Negative Reviews")
plt.show()

# Prepare data for ML
X = df['Text']
y = df['Sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numbers
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Test model
predictions = model.predict(X_test_vec)
print("\nAccuracy:", accuracy_score(y_test, predictions))

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
