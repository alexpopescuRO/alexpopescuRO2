import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = LogisticRegression()

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def preprocess_data(self, data):
        X = data['text']
        y = data['label']

        # Convert text data into numerical feature vectors
        X = self.vectorizer.fit_transform(X)

        return X, y

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        # Evaluate model on test data
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    def predict_sentiment(self, text):
        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]
        return prediction

# Example usage:
file_path = 'path/to/your/data.csv'

sentiment_analyzer = SentimentAnalyzer()

data = sentiment_analyzer.load_data(file_path)
X, y = sentiment_analyzer.preprocess_data(data)

accuracy = sentiment_analyzer.train_model(X, y)
print("Model accuracy:", accuracy)

text = "This movie is great!"
prediction = sentiment_analyzer.predict_sentiment(text)
print("Sentiment prediction:", prediction)
