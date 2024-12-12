import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')

# Load the dataset
data_path = '/home/apiiit123/Desktop/projects/Spam classification/spam_text_large_dataset.csv'
data = pd.read_csv(data_path)

# Rename columns for consistency
data.rename(columns={'Message': 'message', 'Prediction': 'label'}, inplace=True)

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"\W", " ", text)  # Remove non-alphanumeric characters
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"^\s+|\s+$", "", text)  # Remove leading/trailing spaces
    text = text.lower()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

# Preprocess the dataset
data['message'] = data['message'].apply(preprocess_text)

# Features and labels
X = data['message']
y = data['label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_transformed = vectorizer.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict individual emails based on user input
def predict_email():
    while True:
        try:
            email_index = int(input(f"Enter the email number (0 to {len(data) - 1}): "))
            if 0 <= email_index < len(data):
                email = data.iloc[email_index]['message']
                label = data.iloc[email_index]['label']
                email_transformed = vectorizer.transform([email])
                prediction = model.predict(email_transformed)[0]
                print(f"\nEmail: {email}")
                print(f"Actual Label: {'Spam' if label == 1 else 'Not Spam'}")
                print(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}\n")
            else:
                print("Invalid index. Try again.")
        except ValueError:
            print("Invalid input. Enter a valid number.")

# Start prediction
predict_email()

