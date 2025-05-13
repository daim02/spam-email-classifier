import os
import pandas as pd
import numpy as np
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from nltk.stem import PorterStemmer
import random

# Download all NLTK resources
nltk.download('stopwords')

# Load the dataset
dataset_path = 'Put File Path Here' # Replace with file path when testing
data = pd.read_csv(dataset_path)

# Clean the dataset by removing newline characters
data['text'] = data['text'].replace({r'\r\n': ' '}, regex=True)

# Check for any missing values
missing_values = data.isna().sum()
print(f"Missing Values:\n{missing_values}")

# Remove duplicates if any ar e present
data.drop_duplicates(inplace=True)

# Stemming tool and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words.discard('not')

# Prepare the corpus - process each text sample
processed_texts = []

for index, row in data.iterrows():
    # Convert text to lowercase, remove punctuation, and split into words - CLEAN data
    cleaned_text = row['text'].lower().translate(str.maketrans('', '', string.punctuation))
    words = cleaned_text.split()

    # Remove stopwords and apply stemming
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]

    # Rejoin the words into a single string
    processed_texts.append(' '.join(filtered_words))

# TFID Vectorizer
vectorizer = TfidfVectorizer(max_features=10000)  # Using TF-IDF here
X = vectorizer.fit_transform(processed_texts).toarray()  # Vectorize the processed texts
y = data['label_num']

# Split dataset into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Function to calculate and display the evaluation metrics
def evaluate_model(true_labels, predicted_labels):
    # Accuracy (for the whole dataset)
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Precision and Recall for both classes (Ham and Spam)
    precision_0 = precision_score(true_labels, predicted_labels, pos_label=0)  # For Ham (class 0)
    precision_1 = precision_score(true_labels, predicted_labels, pos_label=1)  # For Spam (class 1)

    recall_0 = recall_score(true_labels, predicted_labels, pos_label=0)  # For Ham (class 0)
    recall_1 = recall_score(true_labels, predicted_labels, pos_label=1)  # For Spam (class 1)

    # F1-Score for both classes
    f1_0 = f1_score(true_labels, predicted_labels, pos_label=0)  # For Ham (class 0)
    f1_1 = f1_score(true_labels, predicted_labels, pos_label=1)  # For Spam (class 1)

    # Average Score
    avg_score = np.mean([accuracy, precision_1, recall_1, f1_1])

    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print(f"\nFor Ham (Class 0):")
    print(f"  Precision: {precision_0:.4f} ({precision_0*100:.2f}%)")
    print(f"  Recall: {recall_0:.4f} ({recall_0*100:.2f}%)")
    print(f"  F1-Score: {f1_0:.4f} ({f1_0*100:.2f}%)")

    print(f"\nFor Spam (Class 1):")
    print(f"  Precision: {precision_1:.4f} ({precision_1*100:.2f}%)")
    print(f"  Recall: {recall_1:.4f} ({recall_1*100:.2f}%)")
    print(f"  F1-Score: {f1_1:.4f} ({f1_1*100:.2f}%)")

    print(f"Average Score: {avg_score:.4f} ({avg_score*100:.2f}%)")

    # Confusion Matrix output
    cm = confusion_matrix(true_labels, predicted_labels)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")

    # Plot the Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # ROC Curve ouput
    fpr, tpr, _ = roc_curve(true_labels, rf_classifier.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


# Function to predict if an input text is spam or ham via user input
def test_model():
    print("Please enter the text you want to classify:")
    user_input = input()

    # Preprocess the input
    cleaned_input = user_input.lower().translate(str.maketrans('', '', string.punctuation))
    words = cleaned_input.split()
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]
    processed_input = ' '.join(filtered_words)

    # Vectorize the input text
    input_vector = vectorizer.transform([processed_input]).toarray()

    # Predict using the trained RF model
    prediction = rf_classifier.predict(input_vector)

    if prediction == 0:
        print("The text is classified as: Ham")
    else:
        print("The text is classified as: Spam")

# Menu for User
def menu():
    while True:
        print("\nChoose an option:")
        print("1. View Model Stats and Performance")
        print("2. Test Model")
        print("3. Exit")

        choice = input("Enter the number of your choice: ")

        if choice == '1':
            predictions = rf_classifier.predict(X_test)
            evaluate_model(y_test, predictions)
        elif choice == '2':
            test_model()
        elif choice == '3':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice, please try again.")

# Run the menu function
menu()
